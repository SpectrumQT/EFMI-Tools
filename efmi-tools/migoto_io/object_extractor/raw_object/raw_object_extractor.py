import numpy
import hashlib
import re

from dataclasses import dataclass, field

from ...data_model.byte_buffer import BufferLayout, BufferSemantic, AbstractSemantic, Semantic, NumpyBuffer
from ...data_model.dxgi_format import DXGIFormat

from ...migoto_model.migoto_format import MigotoFormat
from ...migoto_model.types import SlotType, ShaderType, ResourceSlot
from ...migoto_model.frame_model.calls import ShaderCall
from ...migoto_model.frame_model.frame_model import DumpModel, ParseDumpModelConfig
from ...migoto_model.frame_model.resources import Resource, ConstantBuffer, IndexBuffer, VertexBuffer, ResourceStorage
from ...migoto_model.frame_model.api_calls.draw_calls import DrawCall, DrawIndexedInstanced

from .raw_object import RawObject, RawComponent


class DrawCallFilter:

    def collect(self, model: DumpModel) -> list[ShaderCall]:

        calls = []

        for shader_call in model.calls:

            if not isinstance(shader_call.draw_call, DrawIndexedInstanced):
                continue

            # Resource may be missing "real" in-game hash if it's malformed or loaded from local files.
            # In either case there's no way to produce a healthy mod due to missing metadata.
            if shader_call.has_unknown_resource:
                continue

            buffers = shader_call.resources

            vb0 = buffers.get_by_slot(ResourceSlot(ShaderType.Any, SlotType.VertexBuffer, 0))
            if vb0 is None:
                continue
            if not isinstance(vb0, VertexBuffer):
                raise ValueError

            if not vb0.data_descriptor:
                continue

            # if vb0.data_descriptor.byte_offset:
            #     continue

            vs_t0 = buffers.get_by_slot(ResourceSlot(ShaderType.Vertex, SlotType.Texture, 0))
            if vs_t0 is None:
                continue

            vs_cb0 = buffers.get_by_slot(ResourceSlot(ShaderType.Vertex, SlotType.ConstantBuffer, 0))
            if vs_cb0 is None:
                continue

            calls.append(shader_call)

        return calls


@dataclass
class RawObjectIdentifier:

    @staticmethod
    def get_object_id(shader_call: ShaderCall) -> tuple[str, bool]:

        dynamic_cb = None

        for slot, cb in shader_call.resources.constant_buffers.items():

            if slot.shader_type != ShaderType.Vertex:
                continue

            if cb.num_constants == 4096:
                dynamic_cb = cb
                break

        if dynamic_cb is None:
            return "0x0000000000000000"

        dynamic_cb.build_numpy_buffer(MigotoFormat(vb_layout=BufferLayout([
            BufferSemantic(AbstractSemantic(Semantic.RawData, 0), DXGIFormat.R32G32B32A32_FLOAT, input_slot=0),
        ])))

        offset = dynamic_cb.first_constant

        data = dynamic_cb.buffer.get_field(0)

        fragment = data[offset:offset + 4]

        # Compute 64-bit hash
        h = hashlib.blake2b(fragment.view(numpy.uint8), digest_size=8)
        fragment_hash = int.from_bytes(h.digest(), 'little')

        object_id = f"{fragment_hash:016x}"

        gpu_posed = numpy.bitwise_and(numpy.int32(-17), data[offset + 4, 3].view(numpy.int32)) != 0

        return object_id, gpu_posed


@dataclass
class RawObjectFilter:
    min_component_count: int = 0
    min_texture_count: int = 0
    lookup_resource_hashes: str = ""

    _lookup_resource_hashes: set[str] = field(init=False)

    def __post_init__(self):
        self._lookup_resource_hashes = set([x for x in re.split(r"[,; ]", self.lookup_resource_hashes) if x])

    def is_valid_object(self, raw_object: RawObject) -> bool:

        if self.min_component_count:
            if len(raw_object.components) < self.min_component_count:
                return False

        if self.min_texture_count:
            has_ps_textures = any(
                len(component.get_resources(SlotType.Texture, ShaderType.Pixel)) > 0
                for component in raw_object.components.values()
            )
            if not has_ps_textures:
                return False

        if self.lookup_resource_hashes:
            resource_found = False
            for component in raw_object.components.values():
                for shader_call in component.shader_calls:
                    for lookup_hash in self._lookup_resource_hashes:
                        if shader_call.resources.get_by_hash(lookup_hash):
                            resource_found = True
                            break
            if not resource_found:
                return False
            
        return True


@dataclass
class RawObjectExtractor:
    draw_call_filter: DrawCallFilter
    identifier: RawObjectIdentifier
    raw_object_filter: RawObjectFilter

    def register_shader_call(self, extracted_object: RawObject, shader_call: ShaderCall, gpu_posed: bool):
        ib: IndexBuffer = shader_call.resources.get_by_slot(ResourceSlot(ShaderType.Any, SlotType.IndexBuffer, 0))
        ib.build_numpy_buffer()

        vb0: VertexBuffer = shader_call.resources.get_by_slot(ResourceSlot(ShaderType.Any, SlotType.VertexBuffer, 0))

        # Here we calculate bounds of VB segment addressed by IB, where:
        # - Start: vertex_offset
        # - End: vertex_offset + vertex_count
        # If segment spans across entire buffer, vertex_offset is 0 and vertex_count is total number of buffer elements.
        vertex_indices = ib.buffer.get_field(Semantic.Index).flatten()
        vertex_offset = int(min(vertex_indices))
        vertex_count = int(max(vertex_indices) - vertex_offset + 1)

        # Ensure that IB is addressing only existing VB0 vertices.
        if vb0.migoto_format.vertex_count < vertex_count and not vb0.usage_descriptor.resource_hash.startswith("UNKNOWN_"):
            raise ValueError(f"BUG: Call {shader_call.id:06d} IB-addressed vertex count {vertex_count} is above VB0 vertex count {vb0.migoto_format.vertex_count}!")

        # When lower bound of IB-addressed VB segment is non-zero, vertex indices in IB should be decremented
        # by its value, otherwise we'll have to import unused vertices (aka garbage data) to keep IB valid.
        if vertex_offset > 0:
            if not ib.buffer.data.flags.writeable:
                ib.buffer.data = ib.buffer.data.copy()
            index_data = ib.buffer.get_field(Semantic.Index)
            index_data -= vertex_offset

        # For indexed draws MigotoFormat's first_vertex is derived from BaseVertexLocation, which is a value
        # automatically added to each IB index on draw. While RawComponent's vertex_offset is basically a
        # value that was already added to each IB index (and is usually present when draw call doesn't have
        # BaseVertexLocation defined). But technically we can have BOTH BaseVertexLocation and "shifted" IB
        # co-exist in the same draw call, so we should combine both values, not override:
        if vb0.migoto_format.first_vertex > 0:
            vertex_offset += vb0.migoto_format.first_vertex

        draw_key = (ib.hash, vertex_offset, vertex_count)

        component = extracted_object.components.get(draw_key, None)

        if component is None:
            component = RawComponent(
                vertex_offset=vertex_offset,
                vertex_count=vertex_count,
                gpu_posed=gpu_posed,
            )
            extracted_object.components[draw_key] = component

        component.shader_calls.append(shader_call)

    def extract(self, model: DumpModel) -> dict[str, RawObject]:
        
        shader_calls = self.draw_call_filter.collect(model)

        raw_objects: dict[str, RawObject] = {}
        for shader_call in shader_calls:
                
            object_id, gpu_posed = self.identifier.get_object_id(shader_call)

            extracted_object = raw_objects.get(object_id, None)

            if extracted_object is None:
                extracted_object = RawObject(
                    id=object_id
                )
                raw_objects[object_id] = extracted_object

            self.register_shader_call(extracted_object, shader_call, gpu_posed)

        filtered_objects = {}
        for object_id, extracted_object in raw_objects.items():
            if self.raw_object_filter.is_valid_object(extracted_object):
                filtered_objects[object_id] = extracted_object

        raw_objects = dict(
            sorted(
                filtered_objects.items(),
                key=lambda item: len(item[1].components),
                reverse=True
            )
        )

        return raw_objects
