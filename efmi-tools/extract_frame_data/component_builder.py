import logging
import copy

from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path
from collections import OrderedDict

from ..migoto_io.data_model.byte_buffer import ByteBuffer, IndexBuffer, BufferLayout, BufferSemantic, AbstractSemantic, Semantic, NumpyBuffer
from ..migoto_io.dump_parser.filename_parser import ResourceDescriptor, WrappedResource

from .data_extractor import ShapeKeyData, DrawData
from .shapekey_builder import ShapeKeys

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


@dataclass()
class MeshComponent:
    ib_hash: str
    vb0_hash: str
    sk_hash: str
    vertex_offset: int
    vertex_count: int
    index_offset: int
    index_count: int
    vg_offset: int
    vg_count: int
    buffers: Dict[str, NumpyBuffer]
    textures: Dict[str, List[ResourceDescriptor]]
    ib_source: WrappedResource
    # index_buffer: IndexBuffer
    # vertex_buffer: ByteBuffer
    # skeleton_buffer: ByteBuffer


@dataclass()
class MeshComponentData:
    draw_data: DrawData


@dataclass()
class MeshObject:
    object_id: str = None
    ib_hash: str = None
    vb0_hash: str = None
    cb4_hash: str = None
    vertex_count: int = 0
    index_count: int = 0
    shapekey_hash: str = None
    components_data: List[MeshComponentData] = field(init=False)
    shapekey_data: ShapeKeys = field(init=False)
    components: List[MeshComponent] = field(init=False)
    vg_map: Dict[int, int] = field(init=False)

    def __post_init__(self):
        self.components_data = []
        self.components = []

    def verify(self):
        object_ids = set([component.draw_data.object_id for component in self.components_data])
        if len(object_ids) > 1:
            raise ValueError(f'components object ids mismatch for object %s (ids: %s)' % (
                self.vb0_hash, ', '.join(object_ids)))
        self.object_id = list(object_ids)[0]

        # cb4_hashes = [component.draw_data.cb4_hash for component in self.components_data]
        # common_cb4_hash = max(set(cb4_hashes), key=cb4_hashes.count)
        # for component_id, component in enumerate(self.components_data):
        #     if component.draw_data.cb4_hash != common_cb4_hash:
        #         if component.draw_data.cb3_hash != common_cb4_hash:
        #             raise ValueError(f'component %d CB4 hash mismatch for object %s (common hash: %s)' % (component_id, self.vb0_hash, common_cb4_hash))
        #         component.draw_data.skeleton_data = component.draw_data.skeleton_data_cb3

        # self.cb4_hash = common_cb4_hash

    def import_component_data(self, draw_data: DrawData):
        self.components_data.append(MeshComponentData(
            draw_data=draw_data,
        ))
        
    def build_components(self, vb_layout: BufferLayout, shapekeys: Dict[str, ShapeKeys]):
        self.verify()
        self.components_data.sort(key=lambda data: data.draw_data.vertex_offset, reverse=False)
        self.import_shapekey_data(shapekeys)
        for component_data in self.components_data:
            component = self.build_component(component_data.draw_data, vb_layout)
            self.vertex_count += component.vertex_count
            self.index_count += component.index_count
            self.components.append(component)
        # vg_map = self.get_merged_vg_map()
        for component_id, component in enumerate(self.components):
            # component.vg_map = vg_map[component_id]
            component.vg_map = {}

    def import_shapekey_data(self, shapekeys: Dict[str, ShapeKeys]):
        """
        Imports shapekeys data based on hash and ensures its uniqueness
        """
        for component_data in self.components_data:
            if component_data.draw_data.shapekey_hash is not None:
                if self.shapekey_hash is None:
                    self.shapekey_hash = component_data.draw_data.shapekey_hash
                elif self.shapekey_hash != component_data.draw_data.shapekey_hash:
                    raise ValueError(f'shapekeys data hash mismatch between components of object %s (%s != %s)' % (
                        self.object_id, self.shapekey_hash, component_data.draw_data.shapekey_hash
                    ))
        
        self.shapekey_data = shapekeys.get(self.shapekey_hash, None)

    def build_component(self, draw_data: DrawData, vb_layout_old: BufferLayout):
        """
        Compiles component data from multiple sources into single export-optimized object
        """
        vb_layout = BufferLayout([])

        vb_names = [name for name in ['VB0', 'VB1', 'VB2'] if name in draw_data.buffers.keys()]
        is_static_object = 'VB2' not in vb_names

        try:              
            for vb_name in vb_names:
                if draw_data.buffers[vb_name].buffer is None:
                    # Weapon
                    continue
                vb_layout.merge(draw_data.buffers[vb_name].buffer.layout)
        except Exception as e:
            raise e

        # print(f'Building {draw_data.vb0_hash} buffer len {draw_data.vertex_count}')
        
        ib = draw_data.buffers['IB'].buffer
        if is_static_object:
            ib.data['INDEX'] -= draw_data.vertex_offset

        vb = NumpyBuffer(vb_layout, size=draw_data.vertex_count)

        for vb_name in vb_names:
            buffer = draw_data.buffers[vb_name].buffer
            for semantic in buffer.layout.semantics:
                semantic_name = semantic.get_name()
                data = buffer.get_field(semantic_name)
                if is_static_object:
                     data = data[draw_data.vertex_offset : draw_data.vertex_offset + draw_data.vertex_count]     
                try:              
                    vb.set_field(semantic_name, data)
                except Exception as e:
                    raise e
               
        textures = {}
        for texture in draw_data.textures:
            for descriptor in texture.descriptors:
                slot_hash = descriptor.get_slot_hash()
                textures[slot_hash] = descriptor

        buffers = {
            'IB': ib,
            'VB': vb,
            'SkeletonBuffer': None,
        }

        return MeshComponent(
            ib_hash=draw_data.ib_hash,
            vb0_hash=draw_data.vb0_hash,
            sk_hash=draw_data.shapekey_hash,
            vertex_offset=draw_data.vertex_offset,
            vertex_count=draw_data.vertex_count,
            index_offset=draw_data.index_offset,
            index_count=draw_data.index_count,
            vg_offset=0,
            vg_count=0,
            buffers=buffers,
            textures=textures,
            ib_source=draw_data.buffers['IB'],
        )

    def get_merged_vg_map(self):
        """
        Concatenates VGs of components and remaps duplicate VGs based on bone values from skeleton buffers
        """
        vg_offset = 0
        vg_map = {}
        unique_bones = {}

        for component_id, component in enumerate(self.components):
            vg_map[component_id] = {}
            # Fetch joined list of all VG ids of all vertices of the component (4 VG ids per vertex)
            vertex_groups = component.vertex_buffer.get_values(AbstractSemantic(Semantic.Blendindices))
            # For remapping purposes, VG count is the highest used VG id among all vertices of the component
            # It allows to efficiently construct merged skeleton buffer in-game via vg_offset & vg_count of components
            component.vg_offset = vg_offset
            component.vg_count = max(vertex_groups) + 1
            # Ensure frame dump data integrity
            if component.skeleton_buffer.num_elements < component.vg_count:
                raise ValueError('skeleton of Component_%d has only %d bones, while there are %d VGs declared' % (
                    component_id, component.skeleton_buffer.num_elements, component.vg_count))
            # Build VG map
            for vg_id in range(component.vg_count):
                # Fetch data floats of bone which VG is linked to
                buffer_element = component.skeleton_buffer.get_element(vg_id)
                bone_data = tuple(buffer_element.get_value(AbstractSemantic(Semantic.RawData)))
                # Skip zero-valued bone (garbage data)
                if all(v == 0 for v in bone_data):
                    continue
                # Get desc object of already registered unique bone data
                unique_bone_data = unique_bones.get(bone_data, None)
                # Register VG in VG map
                if unique_bone_data is None or unique_bone_data['component_id'] == component_id:
                    # Handle new VG or duplicate VG within same component
                    shifted_vg_id = vg_offset + vg_id  # Remap VG to VG of merged skeleton
                    vg_map[component_id][vg_id] = shifted_vg_id  # Remap VG to VG of merged skeleton
                    unique_bones[bone_data] = {  # Register unique bone data
                        'component_id': component_id,
                        'vg_id': shifted_vg_id
                    }
                else:
                    # Handle duplicate VG across different components
                    vg_map[component_id][vg_id] = unique_bone_data['vg_id']  # Remap VG to VG of already registered bone
                    log.info(f'Remapped duplicate VG %d of Component_%d to VG %d of Component_%s' % (
                        vg_id, component_id, vg_map[component_id][vg_id], unique_bone_data["component_id"]))
                    
            vg_offset += component.vg_count

        log.info(f'Build Merged VG Map for {vg_offset} Vertex Groups')

        return dict(sorted(vg_map.items()))

    # def merge_vertex_groups(self):
    #         # Remap VG ids based on map we've constructed
    #         merged_vertex_groups = [vg_map[vg_id] for vg_id in vertex_groups]
    #         # Write edited data back to the byte buffer
    #         component.vertex_buffer.set_values(AbstractSemantic(Semantic.Blendindices), merged_vertex_groups)


@dataclass
class ComponentBuilder:
    # Input
    output_vb_layout: BufferLayout
    shader_hashes: Dict[str, str]
    shapekeys: Dict[str, ShapeKeys]
    draw_data: Dict[tuple, DrawData]
    # Output
    mesh_objects: Dict[str, MeshObject] = field(init=False)

    def __post_init__(self):

        self.mesh_objects = {}

        sorted_draw_data = {}

        total_vertex_count = 0
        total_vertex_count_weapon = 0

        for (object_id, vb0_hash, vertex_offset, vertex_count), draw_data in self.draw_data.items():

            draw_guid = (object_id, vb0_hash, vertex_offset, vertex_count)

            if draw_data is None:
                raise ValueError(f'no draw data found for component {":".join(draw_guid)}')
            
            positions = draw_data.buffers['VB0'].buffer.get_field(AbstractSemantic(Semantic.Position))[:, 2]

            min_pos = float(positions.min())

            if min_pos > -0.02:
                total_vertex_count += vertex_count
            else:
                total_vertex_count_weapon += vertex_count

            sorted_draw_data[float(positions.max())] = (object_id, vb0_hash, vertex_offset, vertex_count, min_pos, draw_data)

        sorted_draw_data = dict(sorted(sorted_draw_data.items(), key=lambda x: x[0], reverse=True))


        for max_pos, (object_id, vb0_hash, vertex_offset, vertex_count, min_pos, draw_data) in sorted_draw_data.items():
            
            if min_pos < -0.02:
                object_id = f'Weapon {total_vertex_count_weapon}'
            else:
                object_id = f'Character {total_vertex_count}'

            draw_data.object_id = object_id

            # if draw_data.texcoord_buffer.num_elements == 0:
            #     draw_data.texcoord_buffer.extend(draw_data.position_buffer.num_elements)
                # print(f'Skipped incomplete object {draw_data.vb_hash} (no texcoord_buffer)')
                # continue

            # if draw_data.color_buffer.num_elements == 0:
            #     draw_data.color_buffer.extend(draw_data.position_buffer.num_elements)
                # print(f'Skipped incomplete object {draw_data.vb_hash} (no color_buffer)')
                # continue

            if object_id not in self.mesh_objects:
                self.mesh_objects[object_id] = MeshObject()

            self.mesh_objects[object_id].import_component_data(draw_data)
            
        for mesh_object in self.mesh_objects.values():
            mesh_object.build_components(self.output_vb_layout, self.shapekeys)

        log.info(f'Collected components for {len(self.mesh_objects)} VB hashes: {", ".join(self.mesh_objects.keys())}')



    
    









