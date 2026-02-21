import time

from dataclasses import dataclass, field
from typing import Union, List, Dict, Set
from enum import Enum
from pathlib import Path

from ..migoto_io.data_model.dxgi_format import DXGIFormat
from ..migoto_io.data_model.byte_buffer import ByteBuffer, IndexBuffer, BufferLayout, BufferSemantic, AbstractSemantic, Semantic, NumpyBuffer, MigotoFormat
from ..migoto_io.dump_parser.log_parser import CallParameters
from ..migoto_io.dump_parser.filename_parser import ResourceDescriptor, SlotType
from ..migoto_io.dump_parser.resource_collector import ShaderCallBranch, WrappedResource, ResourceConflict


class PoseConstantBufferFormat(Enum):
    static = 1
    animated = 2


@dataclass(frozen=True)
class ShapeKeyData:
    shapekey_hash: str
    shapekey_scale_hash: str
    dispatch_y: int
    shapekey_offset_buffer: ByteBuffer
    shapekey_vertex_id_buffer: ByteBuffer
    shapekey_vertex_offset_buffer: ByteBuffer


@dataclass
class DrawData:
    object_id: str
    vertex_offset: int
    vertex_count: int
    index_offset: int
    index_count: int
    buffers: Dict[str, WrappedResource]
    textures: Set[ResourceDescriptor]

    def get_buffer_hash(self, buffer_name) -> Union[str, None]:
        buffer = self.buffers.get(buffer_name, None)
        return buffer.hash if buffer else None

    @property
    def ib_hash(self) -> Union[str, None]:
        return self.get_buffer_hash('IB')

    @property
    def vb0_hash(self) -> Union[str, None]:
        return self.get_buffer_hash('VB0')

    @property
    def cb3_hash(self) -> Union[str, None]:
        return self.get_buffer_hash('CB3')

    @property
    def cb4_hash(self) -> Union[str, None]:
        return self.get_buffer_hash('CB4')

    @property
    def shapekey_hash(self) -> Union[str, None]:
        return self.get_buffer_hash('SK')


@dataclass
class DataExtractor:
    # Input
    call_branches: Dict[str, ShaderCallBranch]
    # Output
    shader_hashes: Dict[str, str] = field(init=False)
    shape_key_data: Dict[str, ShapeKeyData] = field(init=False)
    draw_data: Dict[tuple, DrawData] = field(init=False)

    def __post_init__(self):
        self.shader_hashes = {}
        self.shape_key_data = {}
        self.draw_data = {}

        self.imported_components = []
        self.skipped_components = []

        self.object_id = f'{int(time.time())}'

        self.semantic_remap = {
            BufferSemantic(
                AbstractSemantic(Semantic.Normal, 0), format=DXGIFormat.R32_FLOAT, input_slot=0
                ): BufferSemantic(
                    AbstractSemantic(Semantic.EncodedData, 0), format=DXGIFormat.R32_UINT, input_slot=0
            ),
            BufferSemantic(
                AbstractSemantic(Semantic.TexCoord, 4), format=DXGIFormat.R8G8B8A8_SNORM, input_slot=1
                ): BufferSemantic(
                    AbstractSemantic(Semantic.Color, 0), format=DXGIFormat.R8G8B8A8_SNORM, input_slot=1
            ),
            BufferSemantic(
                AbstractSemantic(Semantic.TexCoord, 3), format=DXGIFormat.R8G8B8A8_SNORM, input_slot=2
                ): BufferSemantic(
                    AbstractSemantic(Semantic.Color, 2), format=DXGIFormat.R8G8B8A8_SNORM, input_slot=2
            ),
            BufferSemantic(
                AbstractSemantic(Semantic.TexCoord, 4), format=DXGIFormat.R8G8B8A8_SNORM, input_slot=2
                ): BufferSemantic(
                    AbstractSemantic(Semantic.Color, 1), format=DXGIFormat.R8G8B8A8_SNORM, input_slot=2
            ),
        }
        self.expected_input_slots = {
            AbstractSemantic(Semantic.Position, 0): [0],
            AbstractSemantic(Semantic.Normal, 0): [0],
            AbstractSemantic(Semantic.Tangent, 0): [0],
            AbstractSemantic(Semantic.EncodedData, 0): [0],
            AbstractSemantic(Semantic.TexCoord, 0): [1],
            AbstractSemantic(Semantic.TexCoord, 1): [1],
            # AbstractSemantic(Semantic.TexCoord, 4): [1],
            AbstractSemantic(Semantic.Color, 0): [1],
            AbstractSemantic(Semantic.Blendindices, 0): [2],
            AbstractSemantic(Semantic.Blendweight, 0): [2],
            AbstractSemantic(Semantic.TexCoord, 5): [2],  # Ember Weapon
            AbstractSemantic(Semantic.TexCoord, 4): [1], 
            AbstractSemantic(Semantic.Color, 1): [2], # Unmapped semantic in VB2 with Texcoord 3 (Factorio static)
            AbstractSemantic(Semantic.Color, 2): [2], # Texcoor 3 in VB2
        }
        self.allowed_missing_input_slots = {
            AbstractSemantic(Semantic.Normal, 0): [0],
            AbstractSemantic(Semantic.Tangent, 0): [0],
            AbstractSemantic(Semantic.EncodedData, 0): [0],
            AbstractSemantic(Semantic.TexCoord, 1): [1],
            # AbstractSemantic(Semantic.TexCoord, 4): [1],
            AbstractSemantic(Semantic.Blendweight, 0): [2],
            AbstractSemantic(Semantic.Color, 0): [1],
            AbstractSemantic(Semantic.Blendindices, 0): [2],
            AbstractSemantic(Semantic.TexCoord, 5): [2],
            AbstractSemantic(Semantic.TexCoord, 4): [1], 
            AbstractSemantic(Semantic.Color, 1): [2], # Unmapped semantic in VB2 with Texcoord 3 (Factorio static)
            AbstractSemantic(Semantic.Color, 2): [2],  # Texcoor 3 in VB2
        }
        self.allowed_unmapped_resources = {
            BufferSemantic(AbstractSemantic(Semantic.EncodedData, 0), format=DXGIFormat.R32_UINT, input_slot=0, offset=12): 16,
            BufferSemantic(AbstractSemantic(Semantic.Color, 0), format=DXGIFormat.R8G8B8A8_SNORM, input_slot=1, offset=8): 12,
            # BufferSemantic(AbstractSemantic(Semantic.Tangent, 0), format=DXGIFormat.R32G32B32A32_FLOAT, input_slot=0, offset=24): 40,  # Levi brows
            BufferSemantic(AbstractSemantic(Semantic.TexCoord, 1), format=DXGIFormat.R32G32_FLOAT, input_slot=1, offset=8): 20,  # Levi
            BufferSemantic(AbstractSemantic(Semantic.Color, 1), format=DXGIFormat.R8G8B8A8_SNORM, input_slot=2, offset=0): 8,
            BufferSemantic(AbstractSemantic(Semantic.Color, 2), format=DXGIFormat.R8G8B8A8_SNORM, input_slot=2, offset=4): 8,
        }

        self.handle_shapekey_cs_0(list(self.call_branches.values()))
        self.handle_draw_vs(list(self.call_branches.values()))

    def handle_shapekey_cs_0(self, call_branches):
        for call_branch in call_branches:
            if call_branch.shader_id != 'SKELETON_CS_0':
                continue
            for branch_call in call_branch.calls:
                self.verify_shader_hash(branch_call.call, call_branch.shader_id, 1)
            # We don't need any data from this call, lets go deeper
            # self.handle_shapekey_cs_1(call_branch.nested_branches)
            self.handle_draw_vs(call_branch.nested_branches)

    def handle_shapekey_cs_1(self, call_branches):
        for call_branch in call_branches:
            if call_branch.shader_id != 'DRAW_VS':
                continue

    def verify_semantics(self, vb_layout: BufferLayout, input_slot: int):
        mapped = {
            semantic.abstract
            for semantic in vb_layout.semantics
            if semantic.input_slot == input_slot
        }

        expected = {
            semantic
            for semantic, input_slots in self.expected_input_slots.items()
            if input_slot in input_slots
        }

        missing = expected - mapped
        if missing:
            allowed_missing = {
                semantic
                for semantic, input_slots in self.allowed_missing_input_slots.items()
                if input_slot in input_slots
            }
            missing -= allowed_missing

        unexpected = mapped - expected

        return list(missing), list(unexpected)
    
    def warn(self, call_id, vb0_hash, msg):
        print(f'[WARN][{call_id}][{vb0_hash}]: {msg}') 

    def info(self, call_id, vb0_hash, msg):
        # print(f'[INFO][{call_id}][{vb0_hash}]: {msg}')
        pass

    def handle_draw_vs(self, call_branches):
        for call_branch in call_branches:
            
            if call_branch.shader_id != 'DRAW_VS':
                continue

            for branch_call in call_branch.calls:
                call_id = branch_call.call.id
            
                draw_indexed = branch_call.call.parameters.get(CallParameters.DrawIndexedInstanced, None)
                if draw_indexed is None:
                    draw_indexed = branch_call.call.parameters.get(CallParameters.DrawIndexed, None)
                if draw_indexed is None:
                    self.warn(call_id, 'N/A', f'Skipping call without DrawIndexed/Instanced')
                    continue

                index_offset = draw_indexed.StartIndexLocation
                index_count = draw_indexed.IndexCount

                vb0 = branch_call.resources.get('VB0', None)
                if vb0 is None:
                    self.info(call_id, 'N/A', f'Skipping call without VB0')
                    continue
                vb0_hash = vb0.hash

                is_static_object = branch_call.resources.get('VB2', None) is None
                if is_static_object:
                    # self.info(call_id, 'vb0_hash', f'Skipping static object (not supported yet)')
                    # self.skipped_components.append(vb0_hash)
                    continue

                # Read buffer layouts
                buffers: Dict[str, WrappedResource] = {
                    'IB': branch_call.resources.get('IB', None),
                    'VB0': vb0,
                }
                    
                ib = buffers['IB']
                ib_format = ib.get_format(call_id)
                if ib_format.ib_layout is None:
                    for descriptor in ib.descriptors:
                        if descriptor.call_id == call_id:
                            if descriptor.slot_type == SlotType.IndexBuffer:
                                path = Path(descriptor.data.path).with_suffix('.txt')
                                with open(path, 'r') as f:
                                    ib_format = MigotoFormat.from_txt_file(f)  
                                break

                ib_layout = ib_format.ib_layout
                index_semantic_name = AbstractSemantic(Semantic.Index).get_name()

                if is_static_object:
                    object_id = f'Static {ib.hash}'
                else:
                    object_id = self.object_id
                    vb = branch_call.resources.get('VB2', None)
                    if vb:
                        vb_format = vb.get_format(call_id)
                        vb_layout = vb_format.vb_layout
                        if vb_layout.get_element(AbstractSemantic(Semantic.TexCoord, 3)) or vb_layout.get_element(AbstractSemantic(Semantic.Color, 2)):
                            object_id = f'Factory {vb0_hash}'

                if is_static_object:
                    # Load unique IB for static object from TXT
                    for descriptor in ib.descriptors:
                        if descriptor.call_id == call_id and descriptor.slot_type == SlotType.IndexBuffer:
                            ib = WrappedResource(descriptor, load_header=True)
                            with open(Path(descriptor.path).with_suffix('.txt'), 'r') as f:
                                ib_container = IndexBuffer(ib_layout, f, load_indices=True)
                            ib_data = ib_container.get_numpy_array()
                            ib.buffer = NumpyBuffer(ib_layout, size=len(ib_data))
                            ib.buffer.set_field(index_semantic_name, ib_data)
                            buffers['IB'] = ib
                            break
                else:
                    # Load last used IB from BUF
                    try:
                        ib.load_buffer(ib_layout)
                    except Exception as e:
                        raise e

                ib = buffers['IB']
                vertex_indices = ib.buffer.get_field(index_semantic_name).flatten()

                vertex_offset = int(min(vertex_indices))
                vertex_count = int(max(vertex_indices) - vertex_offset + 1)

                draw_guid = (object_id, vb0_hash, vertex_offset, vertex_count)

                textures = set(branch_call.textures)

                # if vb0_hash in self.skipped_components:
                #     print(f'Warning: Skipping already skipped object {vb0_hash} for call {branch_call.call}')
                #     continue
                if draw_guid in self.imported_components:
                    cached_draw_data = self.draw_data.get(draw_guid, None)
                    if cached_draw_data is not None:
                        cached_draw_data.textures.update(textures)
                    # print(f'Warning: Skipping already imported object {vb0_hash} for call {branch_call.call}')
                    continue

                # Fetch remaining VB buffers
                for vb_name in ['VB1', 'VB2', 'VB3', 'VB4']:
                    vb = branch_call.resources.get(vb_name, None)
                    if vb is not None:
                        buffers[vb_name] = vb

                # Skip resource if any VB hash matches IB
                vb_buffers = [vb for vb_name, vb in buffers.items() if vb_name.startswith('VB')]
                if any(SlotType.IndexBuffer in vb.slot_types for vb in vb_buffers):
                    self.warn(call_id, vb0_hash, f'Object skipped due to IB with same hash found')
                    self.skipped_components.append(vb0_hash)
                    continue

                # Split VB resource if hash matches IB
                # for vb_name, vb in buffers.items():
                #     if not vb_name.startswith('VB'):
                #         continue
                #     if SlotType.IndexBuffer not in vb.slot_types:
                #         continue
                #     descriptors = [x for x in vb.descriptors if x.slot_type == SlotType.VertexBuffer]
                #     if len(descriptors) == 0:
                #         raise ValueError(f'No VertexBuffer descriptors found for VB resource!')
                #     split_vb = WrappedResource(descriptors[0], load_header=True)
                #     for descriptor in descriptors[1:]:
                #         split_vb.bind_descriptor(descriptor, allow_conflicts=[
                #             ResourceConflict.OldHash, ResourceConflict.SlotId, ResourceConflict.SlotShaderType, ResourceConflict.SlotType
                #         ], error_on_conflict=True, load_header=True)
                #     buffers[vb_name] = split_vb

                # Skip if no VB0 or VB1
                missing_vbs = [vb_name for vb_name in ['VB0', 'VB1'] if vb_name not in buffers.keys()]
                if missing_vbs:
                    self.warn(call_id, vb0_hash, f'Object skipped due to no {missing_vbs} found')
                    self.skipped_components.append(vb0_hash)
                    continue
                
                missing_semantics = []
                unexpected_semantics = []
                stride_mismatch = None
                allow_mismatch = False

                for input_slot in [0, 1, 2]:
                    vb_id = f'VB{input_slot}'

                    vb = buffers.get(vb_id, None)

                    if vb is None:
                        break

                    vb_format = vb.get_format(call_id)
                    vb_layout = vb_format.vb_layout

                    if any(semantic.abstract == AbstractSemantic(Semantic.TexCoord, 5) for semantic in vb_layout.get_elements_in_slot(input_slot)):
                        continue

                    # Remove duplicate semantic definitions for same bytes
                    vb_layout.sort()
                    vb_layout.remove_data_views()

                    # Remap semantics
                    remapped_semantics = {}
                    for buffer_semantic in vb_layout.get_elements_in_slot(input_slot):
                        for remap_from, remap_to in self.semantic_remap.items():
                            if remap_from.input_slot != input_slot:
                                continue
                            if remap_from.abstract == buffer_semantic.abstract:
                                if remap_from.format != buffer_semantic.format:
                                    self.warn(call_id, vb0_hash, f'Failed to remap {buffer_semantic.abstract}->{remap_to.abstract} (expected {remap_from.format}, received {buffer_semantic.format}) for semantic {buffer_semantic}')
                                    continue
                                remapped_semantics[remap_to.abstract] = buffer_semantic.abstract
                                buffer_semantic.abstract = remap_to.abstract
                                buffer_semantic.input_slot = remap_to.input_slot
                                buffer_semantic.format = remap_to.format

                    # Map unmapped semantics and fix wrongly mapped ones
                    for expected_semantic, expected_stride in self.allowed_unmapped_resources.items():
                        if expected_semantic.input_slot != input_slot:
                            continue
                        buffer_semantic = vb_layout.get_element(expected_semantic.abstract)
                        if buffer_semantic is None:
                            if vb_format.stride != expected_stride:
                                continue
                            self.warn(call_id, vb0_hash, f'Added missing semantic {expected_semantic}')
                            vb_layout.add_element(expected_semantic)
                            vb_layout.sort()
                        else:
                            if buffer_semantic.offset == expected_semantic.offset:
                                continue
                            if vb_format.stride != expected_stride:
                                continue
                            self.warn(call_id, vb0_hash, f'Fixed wrong stride {buffer_semantic.offset}->{expected_semantic.offset} for semantic {buffer_semantic}')
                            buffer_semantic.offset = expected_semantic.offset

                    missing, unexpected = self.verify_semantics(vb_layout, input_slot)

                    missing_semantics += missing
                    unexpected_semantics += unexpected

                    if missing or unexpected:
                        continue

                    output_vb_layout = BufferLayout([])

                    for buffer_semantic in vb_layout.semantics:
                        if buffer_semantic.input_slot > 2:
                            if buffer_semantic.input_slot == 4:
                                self.info(call_id, vb0_hash, f'Skipped implicit semantic {buffer_semantic} (input slot 4)')
                            else:
                                self.info(call_id, vb0_hash, f'Skipped unknown semantic {buffer_semantic} (input slot above 2)')
                            continue
                        if buffer_semantic.input_slot != input_slot:
                            continue
                        if buffer_semantic.offset != output_vb_layout.stride:
                            self.warn(call_id, vb0_hash, f'Skipped semantic {buffer_semantic} extraction (offset {buffer_semantic.offset} differs from current layout stride {output_vb_layout.stride})')
                            continue

                        output_vb_layout.add_element(buffer_semantic)
                    
                    # Verify output layout stride
                    if vb_format.stride != output_vb_layout.calculate_stride():
                        stride_mismatch = (output_vb_layout, vb_format)
                        self.warn(call_id, vb0_hash, f'Skipped {vb_id} buffer due to layout stride mismatch (input={vb_format.stride} != output={output_vb_layout.calculate_stride()}), semantics: input={vb_format.vb_layout.get_elements_in_slot(input_slot)} vs output={output_vb_layout.get_elements_in_slot(input_slot)}')
                        break
                    
                    # Load buffer data
                    try:
                        buffers[vb_id].load_buffer(output_vb_layout)
                        
                        try:

                            path = Path(buffers[vb_id].data.path).with_suffix('.txt')
                            # layout = BufferLayout([])

                            # for semantic in vb_layout.get_elements_in_slot(input_slot):
                            #     layout.add_element(semantic)

                            # layout = output_vb_layout

                            # buffer = NumpyBuffer(layout, size=vb_format.vertex_count)
                            # with open(path, 'r') as f:
                            #     data = f.read()
                            #     buffer.import_txt_data(data, remapped_semantics)
                            #     print(1)

                            # if len(buffer.data) != len(buffers[vb_id].buffer.data):
                            #     raise ValueError


                        except Exception as e:

                            raise e
                        
                    except Exception as e:

                        self.warn(call_id, vb0_hash, f'Failed to parse buffer file: {buffers[vb_id].data.path}')

                        try:
                            
                            path = Path(buffers[vb_id].data.path).with_suffix('.txt')
                            
                            with open(path, 'r') as f:
                                fmt = MigotoFormat.from_txt_file(f)  
                            
                            layout = BufferLayout([])

                            for semantic in output_vb_layout.get_elements_in_slot(input_slot):

                                if not remapped_semantics.get(semantic.abstract, semantic.abstract) in [x.abstract for x in fmt.vb_layout.semantics]:
                                    continue

                                layout.add_element(semantic)

                            buffer = NumpyBuffer(layout, size=vb_format.vertex_count)

                            with open(path, 'r') as f:
                                data = f.read()
                                buffer.import_txt_data(data, remapped_semantics)
                                print(1)

                            buffers[vb_id].buffer = buffer

                            # allow_mismatch = True

                            self.warn(call_id, vb0_hash, f'Recovered by parsing paired buffer file: {path}')
                        
                        except Exception as e:

                            raise e
                    
                if stride_mismatch and not allow_mismatch:
                    output_vb_layout, vb_format = stride_mismatch
                    self.warn(call_id, vb0_hash, f'Object skipped due to layout stride mismatch!')
                    self.skipped_components.append(vb0_hash)
                    continue
                
                if len(missing_semantics) > 0:
                    self.warn(call_id, vb0_hash, f'Object skipped due to missing semantics {missing_semantics}')
                    self.skipped_components.append(vb0_hash)
                    continue

                if len(unexpected_semantics) > 0:
                    self.warn(call_id, vb0_hash, f'Object skipped due to unexpected semantics {unexpected_semantics}')
                    self.skipped_components.append(vb0_hash)
                    continue

                # vb0_format = vb0.get_format(call_id)
                # try:
                #     assert(vb0_format.first_vertex == vertex_offset)
                #     assert(vb0_format.vertex_count == vertex_count)
                # except Exception as e:
                #     raise e

                # if is_static_object:
                #     for buffer_name, vb in buffers.items():
                #         if not buffer_name.startswith('VB') or vb.buffer is None:
                #             continue
                #         data = vb.buffer.get_data()[vertex_offset : vertex_offset + vertex_count]
                #         try:
                #             assert(len(data) == vertex_count)
                #         except Exception as e:
                #             raise e
                #         buffers[buffer_name].buffer = NumpyBuffer(vb.buffer.layout)
                #         buffers[buffer_name].buffer.set_data(data)
                #         self.warn(call_id, vb0_hash, f'Created {buffer_name} buffer len {len(data)} for GUID {draw_guid}')

                self.imported_components.append(draw_guid)

                # blend_buffer.extend(vertex_count - blend_buffer.num_elements)
                # blendices_semantic = AbstractSemantic(Semantic.Blendindices, 0)
                # blend_buffer.set_values(blendices_semantic, blend_buffer_idx.get_values(blendices_semantic))
                # blendweights_semantic = AbstractSemantic(Semantic.Blendweight, 0)
                # blend_buffer.set_values(blendweights_semantic, [1, 0, 0, 0] * vertex_count)


                buffers['CALL_ID'] = call_id

                draw_data = DrawData(
                    object_id=object_id,
                    vertex_offset=vertex_offset,
                    vertex_count=vertex_count,
                    index_offset=index_offset,
                    index_count=index_count,
                    # dispatch_x=branch_call.call.parameters[CallParameters.Dispatch].ThreadGroupCountX,
                    buffers=buffers,
                    textures=textures,
                )

                cached_draw_data = self.draw_data.get(draw_guid, None)

                if cached_draw_data is None:
                    self.draw_data[draw_guid] = draw_data
                else:
                    if len(ib.buffer.get_data()) != len(cached_draw_data.index_buffer.get_data()):
                        raise ValueError(f'index data mismatch for DRAW_VS')

                    # if texcoord_buffer.num_elements != 0:
                    #     cached_draw_data.texcoord_buffer = texcoord_buffer

                    cached_draw_data.textures.extend(textures)

    def verify_shader_hash(self, call, shader_id, max_call_shaders):
        if len(call.shaders) != max_call_shaders:
            raise ValueError(f'number of associated shaders for {shader_id} call should be equal to {max_call_shaders}!')
        cached_shader_hash = self.shader_hashes.get(shader_id, None)
        call_shader_hash = next(iter(call.shaders.values())).hash
        if cached_shader_hash is None:
            self.shader_hashes[shader_id] = call_shader_hash
        # elif cached_shader_hash != call_shader_hash:
        #     raise ValueError(f'inconsistent shader hash {cached_shader_hash} for {shader_id}')
