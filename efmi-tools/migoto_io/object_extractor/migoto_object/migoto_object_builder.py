import copy
import numpy

from dataclasses import dataclass
from textwrap import dedent

from ...data_model.byte_buffer import BufferLayout, BufferSemantic, AbstractSemantic, Semantic, NumpyBuffer
from ...data_model.dxgi_format import DXGIFormat

from ...migoto_model.types import SlotType
from ...migoto_model.frame_model.resources import IndexBuffer, VertexBuffer
from ...migoto_model.frame_model.calls import ShaderCall
from ...migoto_model.migoto_mesh import MigotoMesh, WeightingType

from ..raw_object.raw_object_extractor import RawObject, RawComponent

from .migoto_object import MigotoObject, MigotoComponent


@dataclass
class SemanticVertexData:
    semantic: BufferSemantic
    layout: BufferLayout
    resource: VertexBuffer
    shader_call: ShaderCall


@dataclass
class MigotoObjectFilter:
    skip_static_objects: bool = False
    # skip_weighted_objects: bool = False
    # skip_character_objects: bool = False
    ignore_errors: bool = False


@dataclass
class MigotoObjectBuilder:
    migoto_object_filter: MigotoObjectFilter
    verbose_logging: bool = False

    semantic_remap = {
        BufferSemantic(
            AbstractSemantic(Semantic.Normal, 0), format=DXGIFormat.R32_FLOAT, input_slot=0
        ): BufferSemantic(
            AbstractSemantic(Semantic.EncodedData, 0), format=DXGIFormat.R32_UINT, input_slot=0
        ),
        # Characters
        BufferSemantic(
            AbstractSemantic(Semantic.TexCoord, 4), format=DXGIFormat.R8G8B8A8_SNORM, input_slot=1
        ): BufferSemantic(
            AbstractSemantic(Semantic.Color, 0), format=DXGIFormat.R8G8B8A8_SNORM, input_slot=1
        ),
        # Factory buildings
        BufferSemantic(
            AbstractSemantic(Semantic.TexCoord, 3), format=DXGIFormat.R8G8B8A8_SNORM, input_slot=2
        ): BufferSemantic(
            AbstractSemantic(Semantic.Color, 2), format=DXGIFormat.R8G8B8A8_SNORM, input_slot=2
        ),
        # Rossi
        BufferSemantic(
            AbstractSemantic(Semantic.Color, 0), format=DXGIFormat.R8G8B8A8_UNORM, input_slot=1
        ): BufferSemantic(
            AbstractSemantic(Semantic.Color, 3), format=DXGIFormat.R8G8B8A8_UNORM, input_slot=1
        ),

        BufferSemantic(
            AbstractSemantic(Semantic.TexCoord, 4), format=DXGIFormat.R8G8B8A8_SNORM, input_slot=2
        ): BufferSemantic(
            AbstractSemantic(Semantic.Color, 1), format=DXGIFormat.R8G8B8A8_SNORM, input_slot=2
        ),
    }

    vb_data_import_slots = [0, 1, 2]

    def collect_index_data(self, raw_component: RawComponent) -> IndexBuffer:

        component_ib_data: IndexBuffer | None = None

        for shader_call in raw_component.shader_calls:

            ib = shader_call.resources.get_by_slot("ib")

            if not isinstance(ib, IndexBuffer):
                raise ValueError(f"{ib} is not an IndexBuffer")

            if component_ib_data is None:
                component_ib_data = ib
            elif ib.hash != component_ib_data.hash:
                raise ValueError(f"IB mismatch across draw calls")

        return component_ib_data

    def collect_vertex_data(self, raw_component: RawComponent) -> list[SemanticVertexData]:
        """
        Collects sources of data for each abstract semantic used by draw calls of given component.

        Since D3D11 does no sanity checks for VB bindings, layout consistency across draw calls is not guaranteed:
        - Same semantic bytes may be mapped to different semantics in the same layout (e.g. to COLOR_0 and COLOR_4).
        - Some semantics may be mapped to non-representative abstractions (e.g. COLOR defined as TEXCOORD).
        - Same semantic bytes may be mapped in different layouts, or even mapped to duplicate resources.

        So the end goal of this function is to disambiguate the source of each semantic bytes and ensure consistency.
        """

        component_vb_data: dict[AbstractSemantic, SemanticVertexData] = {}
        component_vb_data_usage: dict[tuple[str, int], SemanticVertexData] = {}

        unknown_semantic_index_offset = 0

        for shader_call in raw_component.shader_calls:

            for resource_slot, resource in shader_call.resources.vertex_buffers.items():

                if not isinstance(resource, VertexBuffer):
                    raise ValueError(f"{resource} is not an VertexBuffer")
                
                if resource_slot.slot_id not in self.vb_data_import_slots:
                    continue

                if not resource.data_descriptor:
                    continue

                resource.load_format(from_file=True)

                vb_layout = BufferLayout(
                    semantics=copy.deepcopy(resource.migoto_format.vb_layout.get_elements_in_slot(resource_slot.slot_id)),
                    auto_offsets=False,
                    auto_stride=False,
                )
                vb_layout.stride = resource.migoto_format.stride

                ib_hash = next(iter(shader_call.resources.index_buffer.values())).hash

                if vb_layout.stride == 0:
                    print(f"WARNING! [{resource.usage_descriptor.call_id:06d}][IB={ib_hash}][VB{resource_slot.slot_id}={resource.hash}]: Skipped VB with zero-stride layout: {vb_layout}")
                    continue

                vb_layout.sort()
                vb_layout.remove_data_views()

                if self.semantic_remap:
                    remapped_semantics = vb_layout.remap_semantics(self.semantic_remap)
                    for (map_from_semantic, map_to_semantic) in remapped_semantics:
                        if self.verbose_logging:
                            print(f"[{resource.usage_descriptor.call_id:06d}][IB={ib_hash}][VB{resource_slot.slot_id}={resource.hash}]: Remapped {map_from_semantic} to {map_to_semantic}")

                vb_layout.dedupe_semantics()

                # Layout sent to InputAssembler can have gaps (byte strides of unknown semantics).
                missing_semantics = vb_layout.fill_missing_semantics(unknown_semantic_index_offset)
                if missing_semantics:
                    unknown_semantic_index_offset += len(missing_semantics)
                    if self.verbose_logging:
                        print(f"[{resource.usage_descriptor.call_id:06d}][IB={ib_hash}][VB{resource_slot.slot_id}={resource.hash}]: Filled missing semantics: {vb_layout}")

                for buffer_semantic in vb_layout.semantics:
                    vb_data = component_vb_data.get(buffer_semantic.abstract, None)

                    # Ensure consistent buffer semantic for current buffer region across component's draw calls
                    # If same resource pointer + semantic byte offset is mapped differently, semantic_remap must be used
                    sematic_key = (resource.hash, buffer_semantic.offset)
                    
                    # Here we track how semantic for current sematic_key was previously defined
                    vb_data_usage = component_vb_data_usage.get(sematic_key, None)
                    
                    if vb_data_usage is not None:
                        
                        try:
                            sematic_key_defines_unknown = vb_data_usage.semantic.abstract.enum == Semantic.Unknown
                            buffer_semantic_defines_unknown = buffer_semantic.abstract.enum == Semantic.Unknown

                            if sematic_key_defines_unknown:
                                # Semantic for current sematic_key was previously defined as unknown
                                if buffer_semantic_defines_unknown:
                                    # Current VB draw also defines semantic as unknown
                                    # Skip processing this buffer_semantic entirely, since we won't get anything new
                                    continue
                                else:
                                    # Current VB draw defines semantic that was previously unknown
                                    # Remove this abstract semantic (e.g. UNKNOWN_0) from consistency tracking
                                    print(f"Identified {vb_data_usage.semantic} as {buffer_semantic}")
                                    del component_vb_data[vb_data_usage.semantic.abstract]
                            else:
                                # Semantic for current sematic_key is already known, lets handle re-definition attempt
                                if buffer_semantic_defines_unknown:
                                    # Current VB draw defines semantic as unknown, but we've already found proper definiton
                                    # Skip processing this buffer_semantic entirely, since we won't get anything new from it
                                    continue
                                else:
                                    # Current VB draw defines semantic that is already known
                                    # Do nothing and let the consistency checks below handle the rest
                                    pass

                            if buffer_semantic.abstract != vb_data_usage.semantic.abstract:
                                # Current VB draw re-defines semantic for current sematic_key as different semantic
                                raise ValueError(dedent(f"""
                                    Ambiguous buffer semantics across draw calls (missing remap): 
                                    - [{vb_data_usage.resource.usage_descriptor.call_id:06d}]: {vb_data_usage.semantic} (layout: {vb_data_usage.layout})
                                    - [{resource.usage_descriptor.call_id:06d}]: {buffer_semantic} (layout: {vb_layout})
                                """))
                            
                            if buffer_semantic.stride != vb_data_usage.semantic.stride:
                                # Current VB draw re-defines semantic stride for current sematic_key
                                raise ValueError(dedent(f"""
                                    Inconsistent buffer semantics stride across draw calls:
                                    - [{vb_data.resource.usage_descriptor.call_id:06d}]: {vb_data_usage.semantic} (layout: {vb_data_usage.layout})
                                    - [{resource.usage_descriptor.call_id:06d}]: {buffer_semantic} (layout: {vb_layout})
                                """))
                            
                        except Exception as e:
                            print("WARNING! " + str(e).strip())
                            print(f"Solved conflict by selecting first seen semantic {vb_data_usage.semantic}.")
                            continue

                    if vb_data is None:
                        component_vb_data[buffer_semantic.abstract] = SemanticVertexData(buffer_semantic, vb_layout, resource, shader_call)
                        component_vb_data_usage[sematic_key] = component_vb_data[buffer_semantic.abstract]
                    else:
                        if vb_data.resource.hash != resource.hash:
                            # Fetch numpy data for current abstract semantic from VB of previous draw
                            vb_data_migoto_format = copy.deepcopy(vb_data.resource.migoto_format)
                            vb_data_migoto_format.vb_layout = vb_data.layout
                            vb_data_migoto_format.vb_layout.fill_missing_semantics()
                            vb_data.resource.build_numpy_buffer(migoto_format=vb_data_migoto_format)
                            vb_data_semantic_data = vb_data.resource.buffer.get_field(buffer_semantic.abstract)

                            # Fetch numpy data for current abstract semantic from VB of current draw
                            migoto_format = copy.deepcopy(resource.migoto_format)
                            migoto_format.vb_layout = vb_layout
                            migoto_format.vb_layout.fill_missing_semantics()
                            resource.build_numpy_buffer(migoto_format=migoto_format)
                            resource_semantic_data = resource.buffer.get_field(vb_data.semantic.abstract)

                            # Semantic data mismatch means that game defines different entities using the same semantic
                            if not numpy.array_equal(vb_data_semantic_data, resource_semantic_data):

                                ib_hash = next(iter(shader_call.resources.index_buffer.values())).hash

                                if buffer_semantic.abstract in [AbstractSemantic(Semantic.Blendindices), AbstractSemantic(Semantic.Blendweights)]:
                                    print(f"WARNING! [IB={ib_hash}]: {buffer_semantic.abstract} data mismatch: {vb_data.resource.usage_descriptor.call_id:06d}=VB{vb_data.semantic.input_slot}={vb_data.resource.hash} vs {resource.usage_descriptor.call_id:06d}=VB{buffer_semantic.input_slot}={resource.hash}")
                                # if buffer_semantic.abstract == AbstractSemantic(Semantic.Blendindices):
                                #     buffer_semantic.abstract = AbstractSemantic(Semantic.Blendindices, 1)
                                #     component_vb_data[buffer_semantic.abstract] = SemanticVertexData(buffer_semantic, vb_layout, resource)
                                # elif buffer_semantic.abstract == AbstractSemantic(Semantic.Blendweights):
                                #     buffer_semantic.abstract = AbstractSemantic(Semantic.Blendweights, 1)
                                #     component_vb_data[buffer_semantic.abstract] = SemanticVertexData(buffer_semantic, vb_layout, resource)
                                else:
                                    raise ValueError(dedent(f"""
                                        Inconsistent data for VB{buffer_semantic.input_slot} semantic {buffer_semantic.abstract} across draw calls (missing remap?): 
                                        - [{vb_data.resource.usage_descriptor.call_id:06d}][IB={ib_hash}][VB{vb_data.semantic.input_slot}={vb_data.resource.hash}]: {vb_data.layout}
                                        - [{resource.usage_descriptor.call_id:06d}][IB={ib_hash}][VB{buffer_semantic.input_slot}={resource.hash}]: {vb_layout}
                                    """))

        return list(component_vb_data.values())

    def build_index_buffer(self, raw_component: RawComponent) -> IndexBuffer:
        index_buffer = self.collect_index_data(raw_component)
        if index_buffer.buffer is None:
            index_buffer.build_numpy_buffer()
        return index_buffer

    def build_vertex_buffer(self, raw_component: RawComponent) -> NumpyBuffer:

        component_vertex_data = self.collect_vertex_data(raw_component)

        layout = BufferLayout([])

        for vb_data in component_vertex_data:

            layout.add_element(copy.deepcopy(vb_data.semantic))

            if vb_data.resource.buffer is None:
                # Here we copy migoto format to avoid mutating the original one.
                migoto_format = copy.deepcopy(vb_data.resource.migoto_format)
                migoto_format.vb_layout = vb_data.layout

                # MigotoComponent must contain data only for vertices that are actually used for RawComponent draws.
                # RawComponent object has vertex_offset and vertex_count calculated from vertex indices values of IB.
                # Thus, whenever vertex count from draw (migoto_format) is different from RawComponent, we should
                # override migoto_format to ensure VB is built only from effective vertex range.
                if migoto_format.vertex_count != raw_component.vertex_count:
                    migoto_format.first_vertex = raw_component.vertex_offset
                    migoto_format.vertex_count = raw_component.vertex_count

                # Load VB data to memory.
                vb_data.resource.build_numpy_buffer(migoto_format=migoto_format)

            # if vb_data.resource.buffer.layout.get_element(AbstractSemantic(Semantic.Blendindices, 1)):
            #     migoto_format = copy.deepcopy(vb_data.resource.migoto_format)
            #     migoto_format.vb_layout = vb_data.layout
            #     vb_data.layout.fill_missing_semantics()
            #     vb_data.resource.build_numpy_buffer(migoto_format=migoto_format)

        vertex_buffer = NumpyBuffer(layout, size=raw_component.vertex_count)

        for vb_data in component_vertex_data:
            vertex_buffer.import_data(data=vb_data.resource.buffer, semantic_converters={}, format_converters={})

        return vertex_buffer

    def build_migoto_component(self, raw_component: RawComponent) -> MigotoComponent:

        index_buffer = self.build_index_buffer(raw_component)

        mesh = MigotoMesh.from_numpy_buffers(
            index_buffer=index_buffer.buffer,
            vertex_buffer=self.build_vertex_buffer(raw_component),
            topology=index_buffer.data_descriptor.topology,
        )

        component = MigotoComponent(
            mesh=mesh,
            textures=raw_component.get_resources(SlotType.Texture),
            raw_data=raw_component,
        )

        return component
    
    def filter_components(self, migoto_object: MigotoObject) -> None:
        # Remove components without weights from weighted objects.
        if any(component.mesh.get_weighting_type() != WeightingType.NoWeights for component in migoto_object.components):
            filtered_components = []
            for component in migoto_object.components:
                if component.mesh.get_weighting_type() != WeightingType.NoWeights:
                    filtered_components.append(component)
                else:
                    print(f"Skipped component without weights from weighted object: {component}")
            migoto_object.components = filtered_components
    
    def sort_components(self, migoto_object: MigotoObject) -> None:
        migoto_object.components.sort(
            key=lambda component: float(component.mesh.vertex_buffer.get_field(Semantic.Position)[:, 2].max()),
            reverse=True,
        )
    
    def label_object(self, migoto_object: MigotoObject) -> None:
        vertex_count = sum(component.mesh.format.vertex_count for component in migoto_object.components)
        is_weighted = any(component.mesh.get_weighting_type() != WeightingType.NoWeights for component in migoto_object.components)
        has_vertex_offset = any(component.raw_data.vertex_offset != 0 for component in migoto_object.components)

        if is_weighted:
            if len(migoto_object.components) > 6:
                migoto_object.id = f"Character {vertex_count}"
            else:
                is_implicit_weighted = all(component.mesh.get_weighting_type() == WeightingType.Implicit for component in migoto_object.components)
                if is_implicit_weighted:
                    migoto_object.id = f"ImpliedWeighted {vertex_count}"
                else:
                    migoto_object.id = f"Weighted {vertex_count}"

        else:

            has_texcoord_3 = any(component.mesh.vertex_buffer.layout.get_element(AbstractSemantic(Semantic.TexCoord, 3)) for component in migoto_object.components)
            has_color_2 = any(component.mesh.vertex_buffer.layout.get_element(AbstractSemantic(Semantic.Color, 2)) for component in migoto_object.components)

            if not has_vertex_offset and (has_texcoord_3 or has_color_2):
                migoto_object.id = f"Factory {vertex_count}"
            else:
                migoto_object.id = f"Static {vertex_count}"

        if migoto_object.id.startswith("Character"):
            for component in migoto_object.components:
                component.mesh.cpu_posed = not component.raw_data.gpu_posed

    def filter_object(self, migoto_object: MigotoObject) -> bool:

        if len(migoto_object.components) == 0:
            print(f"Skipped object without valid components: {migoto_object}")
            return False

        if self.migoto_object_filter.skip_static_objects:
            is_weighted = any(component.mesh.get_weighting_type() != WeightingType.NoWeights for component in migoto_object.components)
            if not is_weighted:
                return False

        return True

    def build(self, raw_objects: dict[str, RawObject]) -> list[MigotoObject]:

        migoto_objects = []

        for raw_object in raw_objects.values():

            try:

                migoto_object = MigotoObject(
                    id=raw_object.id,
                )

                for raw_component in raw_object.components.values():

                    migoto_component = self.build_migoto_component(raw_component)

                    migoto_object.components.append(migoto_component)

                self.filter_components(migoto_object)

                if not self.filter_object(migoto_object):
                    continue

                self.sort_components(migoto_object)
                self.label_object(migoto_object)

                migoto_object.build_metadata()

                migoto_objects.append(migoto_object)

            except Exception as e:
                if self.migoto_object_filter.ignore_errors:
                    print(dedent(f"""
                        Failed to build object {raw_object.id} with {len(raw_object.components)} components:
                        Error: {e}
                        RawObject: {raw_object}
                    """).strip())
                    continue
                raise e

        return migoto_objects
