import re
import bpy

from typing import List, Dict, Union
from dataclasses import dataclass, field
from enum import Enum
from textwrap import dedent

from ..addon.exceptions import ConfigError

from ..migoto_io.blender_interface.collections import *
from ..migoto_io.blender_interface.objects import *

from ..migoto_io.blender_tools.modifiers import apply_modifiers_for_object_with_shape_keys
from ..migoto_io.blender_tools.vertex_groups import fill_gaps_in_vertex_groups

from ..extract_frame_data.metadata_format import ExtractedObject


class SkeletonType(Enum):
    Merged = 'Merged'
    PerComponent = 'Per-Component'


@dataclass
class TempObject:
    name: str
    object: bpy.types.Object
    vertex_count: int = 0
    index_count: int = 0
    index_offset: int = 0


@dataclass
class MergedObjectComponent:
    objects: List[TempObject]
    id: int = 0
    vertex_count: int = 0
    index_count: int = 0
    blend_remap_id: int = -1
    blend_remap_vg_count: int = 0
    
    def get_object(self, object_name):
        for obj in self.objects:
            if obj.name == object_name:
                return obj


@dataclass
class MergedObjectShapeKeys:
    vertex_count: int = 0


@dataclass
class MergedObject:
    object: bpy.types.Object
    mesh: bpy.types.Mesh
    components: List[MergedObjectComponent]
    shapekeys: MergedObjectShapeKeys
    skeleton_type: SkeletonType
    vertex_count: int = 0
    index_count: int = 0
    vg_count: int = 0
    blend_remap_count: int = 0




import math
from mathutils import Vector, Matrix

def unit_vector_to_octahedron(n):
    """
    Converts a unit vector to octahedron coordinates.
    n is a mathutils.Vector
    """
    # Ensure input is a unit vector
    if n.length_squared > 1e-10:
        n.normalize()
    else:
        return Vector((0.0, 0.0))

    # Calculate L1 norm
    l1_norm = abs(n.x) + abs(n.y) + abs(n.z)
    if l1_norm < 1e-10:
        return Vector((0.0, 0.0))

    # Project to octahedron plane
    x = n.x / l1_norm
    y = n.y / l1_norm

    # Negative hemisphere mapping (only applied when z < 0)
    if n.z < 0:
        # Use precise sign function
        sign_x = math.copysign(1.0, x)
        sign_y = math.copysign(1.0, y)

        # Original mapping formula (preserves good behavior at z=0)
        new_x = (1.0 - abs(y)) * sign_x
        new_y = (1.0 - abs(x)) * sign_y

        # Apply new coordinates directly (remove transition interpolation)
        x = new_x
        y = new_y

    return Vector((x, y))

# def calc_smooth_normals(mesh):
#     """Calculate smooth normals (angle-weighted average)"""
#     vertex_normals = {}

#     # Use vertex index as key (avoid floating point precision issues)
#     for i, vert in enumerate(mesh.vertices):
#         vertex_normals[i] = Vector((0, 0, 0))

#     # Calculate normal for each face and accumulate to vertices with weighting
#     for poly in mesh.polygons:
#         verts = [mesh.vertices[i] for i in poly.vertices]
#         face_normal = poly.normal

#         for i, vert in enumerate(verts):
#             # Get adjacent edge vectors
#             v1 = verts[(i+1) % len(verts)].co - vert.co
#             v2 = verts[(i-1) % len(verts)].co - vert.co

#             # Calculate angle weight
#             v1_len = v1.length
#             v2_len = v2.length
#             if v1_len > 1e-6 and v2_len > 1e-6:
#                 v1.normalize()
#                 v2.normalize()
#                 weight = math.acos(max(-1.0, min(1.0, v1.dot(v2))))
#             else:
#                 weight = 0.0

#             # Accumulate weighted normals
#             vertex_normals[vert.index] += face_normal * weight

#     # Normalize normals
#     for idx in vertex_normals:
#         if vertex_normals[idx].length > 1e-6:
#             vertex_normals[idx].normalize()

#     return vertex_normals

def calc_smooth_normals(mesh):
    vertex_normals = {i: Vector((0,0,0)) for i in range(len(mesh.vertices))}

    for poly in mesh.polygons:
        coords = [mesh.vertices[i].co for i in poly.vertices]
        face_normal = poly.normal

        for i, vi in enumerate(poly.vertices):
            v_prev = coords[i-1] - coords[i]
            v_next = coords[(i+1) % len(coords)] - coords[i]

            # if v_prev.length < 1e-8 or v_next.length < 1e-8:
            #     continue

            # angle at vertex
            weight = v_prev.angle(v_next)
            vertex_normals[vi] += face_normal * weight

    # Normalize accumulated normals
    for idx, n in vertex_normals.items():
        vertex_normals[idx] = n.normalized()
        # if n.length > 1e-8:
        #     vertex_normals[idx] = n.normalized()
        # else:
        #     vertex_normals[idx] = Vector((0,0,1))

    return vertex_normals

def process_object(obj):

    mesh = obj.data

    # # Set active UV to first layer (index 0) before operation
    # if len(mesh.uv_layers) > 0:
    #     mesh.uv_layers.active_index = 0

    # Calculate smooth normals
    smooth_normals = calc_smooth_normals(mesh)

    # # Ensure mesh has UV layer (required for tangent calculation)
    # if len(mesh.uv_layers) == 0:
    #     mesh.uv_layers.new(name="UVMap")

    # Calculate tangent space (TBN matrix)
    mesh.calc_tangents()

    # # Create/get UV layer
    # uv_layer_name = "TEXCOORD3.xy"
    # if uv_layer_name in mesh.uv_layers:
    #     uv_layer = mesh.uv_layers[uv_layer_name]
    # else:
    #     uv_layer = mesh.uv_layers.new(name=uv_layer_name)

    color_attribute = mesh.color_attributes.new(name="COLOR1", type='FLOAT_COLOR', domain='CORNER')
        
    # Process each vertex of each face
    for poly in mesh.polygons:
        for loop_idx in range(poly.loop_start, poly.loop_start + poly.loop_total):
            loop = mesh.loops[loop_idx]
            vertex_idx = loop.vertex_index

            # Get smooth normal
            normal = smooth_normals[vertex_idx]

            # Build TBN matrix (tangent space to model space transformation)
            tbn_matrix = Matrix((
                loop.tangent,
                loop.bitangent,
                loop.normal
            )).transposed() # Transpose to convert from row vectors to column vectors

            # Check if matrix is invertible
            try:
                # Attempt to calculate inverse matrix
                tbn_inverse = tbn_matrix.inverted()

                # Transform normal from model space to tangent space
                tangent_normal = tbn_inverse @ normal
                tangent_normal.normalize()
            except ValueError:
                # Fallback for non-invertible matrix
                print(f"Warning: TBN matrix for vertex {vertex_idx} is non-invertible, using default normal")

                tangent_normal = Vector((0, 0, 1))  # Default to Z-axis as normal

            # # Get smooth normal (model space)
            # normal = smooth_normals[vertex_idx].normalized()

            # # UE4-correct TBN
            # T = loop.tangent.normalized()
            # N = loop.normal.normalized()
            # B = N.cross(T) * loop.bitangent_sign  # IMPORTANT

            # # Transform into tangent space
            # tangent_normal = Vector((
            #     normal.dot(T),
            #     normal.dot(B),
            #     normal.dot(N)
            # ))

            if tangent_normal.length > 1e-6:
                tangent_normal.normalize()
            else:
                tangent_normal = Vector((0, 0, 1))

            # Octahedral projection
            oct_coords = unit_vector_to_octahedron(tangent_normal)
            
            # # Set UV coordinates
            # u = oct_coords.x
            # v = oct_coords.y + 1.0
            # uv_layer.data[loop_idx].uv = (u, v)

            # Color (RGBA)
            color_attribute.data[loop_idx].color = (1 - (oct_coords.x * 0.5 + 0.5), (oct_coords.y * 0.5 + 0.5), 0, 0)

            # color_attribute.data[loop_idx].color = (0, 0, 0, 0)

    # Free tangent data
    mesh.free_tangents()


@dataclass
class ObjectMerger:
    # Input
    context: bpy.types.Context
    extracted_object: ExtractedObject
    ignore_nested_collections: bool
    ignore_hidden_collections: bool
    ignore_hidden_objects: bool
    ignore_muted_shape_keys: bool
    apply_modifiers: bool
    collection: str
    skeleton_type: SkeletonType
    component_id: int = -1
    mesh_scale: float = 1.0
    mesh_rotation: Tuple[float] = (0.0, 0.0, 0.0)
    add_missing_vertex_groups: bool = False
    force_object_name: str =''
    allow_empty_components: bool = False
    # Output
    merged_object: MergedObject = field(init=False)

    def __post_init__(self):
        collection_was_hidden = collection_is_hidden(self.collection)
        unhide_collection(self.collection)

        self.initialize_components()
        try:
            self.import_objects_from_collection()
            if len(self.components) == 1 and self.allow_empty_components and len(self.components[0].objects) == 0:
                self.merged_object = MergedObject(
                    object=None,
                    mesh=None,
                    components=self.components,
                    vertex_count=0,
                    index_count=0,
                    vg_count=0,
                    shapekeys=None,
                    skeleton_type=SkeletonType.PerComponent,
                )
            else:
                self.prepare_temp_objects()
                self.build_merged_object()
        except Exception as e:
            self.remove_temp_objects()
            raise e
        
        if collection_was_hidden:
            hide_collection(self.collection)

    def initialize_components(self):
        self.components = []
        for component_id, component in enumerate(self.extracted_object.components):
            if self.component_id == -1 or self.component_id == component_id:
                self.components.append(
                    MergedObjectComponent(
                        id=self.component_id if self.component_id != -1 else component_id,
                        objects=[],
                        index_count=0,
                    )
                )

    def import_objects_from_collection(self):

        num_objects = 0
        
        if self.force_object_name:
            component_pattern = re.compile(r'^{}'.format(self.force_object_name))
        elif self.component_id == -1:
            component_pattern = re.compile(r'.*component[_ -]*(\d+).*')
        else:
            component_pattern = re.compile(r'.*component[_ -]*({})(?!\d).*'.format(self.component_id))

        for obj in get_collection_objects(self.collection, 
                                          recursive = not self.ignore_nested_collections, 
                                          skip_hidden_collections = self.ignore_hidden_collections):

            if self.ignore_hidden_objects and object_is_hidden(obj):
                continue

            if obj.name.startswith('TEMP_'):
                continue
            
            if not self.force_object_name:
                match = component_pattern.findall(obj.name.lower())
                if len(match) == 0:
                    continue
                component_id = int(match[0])
            elif self.force_object_name == obj.name:
                component_id = 0
            else:
                continue

            if component_id >= len(self.extracted_object.components):
                raise ConfigError('object_source_folder', f'Metadata.json in specified folder is missing Component {component_id}!\nMost likely it contains sources for other object.')

            temp_obj = copy_object(self.context, obj, name=f'TEMP_{obj.name}', collection=self.collection)

            for component in self.components:
                if component.id != component_id:
                    continue
                component.objects.append(TempObject(
                    name=obj.name,
                    object=temp_obj,
                ))
                break

            num_objects += 1

        if num_objects == 0 and not self.allow_empty_components:
            raise ValueError(f'No eligible `Component {self.component_id}` objects found!')

    def prepare_temp_objects(self):

        index_offset = 0

        for component_id, component in enumerate(self.components):

            component.objects.sort(key=lambda x: x.name)

            for temp_object in component.objects:
                temp_obj = temp_object.object
                # Remove muted shape keys
                if self.ignore_muted_shape_keys and temp_obj.data.shape_keys:
                    muted_shape_keys = []
                    for shapekey_id in range(len(temp_obj.data.shape_keys.key_blocks)):
                        shape_key = temp_obj.data.shape_keys.key_blocks[shapekey_id]
                        if shape_key.mute:
                            muted_shape_keys.append(shape_key)
                    for shape_key in muted_shape_keys:
                        temp_obj.shape_key_remove(shape_key)
                # Modify temporary object
                with OpenObject(self.context, temp_obj, mode='OBJECT') as obj:
                    # Apply all transforms
                    bpy.ops.object.transform_apply(location = True, rotation = True, scale = True)
                    # Apply all modifiers
                    if self.apply_modifiers:
                        bpy.ops.object.convert(target='MESH')
                    # Triangulate (this step is crucial since export supports only triangles)
                    triangulate_object(self.context, temp_obj)
                    # Handle missing UV map
                    uv_name = f'TEXCOORD.xy'
                    if uv_name not in temp_obj.data.uv_layers:
                        uv_layer = temp_obj.data.uv_layers.new(name=uv_name)
                        uv_data = numpy.zeros(len(uv_layer.data) * 2, dtype=numpy.float32)
                        uv_layer.data.foreach_set('uv', uv_data)
                # Handle Vertex Groups
                vertex_groups = get_vertex_groups(temp_obj)
                # Fill gaps in Vertex Groups list based on VG names (i.e. add group '1' between '0' and '2' if it's missing)
                if self.add_missing_vertex_groups:
                    fill_gaps_in_vertex_groups(self.context, temp_obj, internal_call=True)
                # Remove ignored or unexpected vertex groups
                if self.skeleton_type == SkeletonType.Merged:
                    # Exclude VGs with 'ignore' tag or with higher VG id than total VG count from Metadata.ini
                    total_vg_count = sum([component.vg_count for component in self.extracted_object.components])
                    ignore_list = [vg for vg in vertex_groups if 'ignore' in vg.name.lower() or vg.index >= total_vg_count]
                elif self.skeleton_type == SkeletonType.PerComponent:
                    # Exclude VGs with 'ignore' tag or with higher id VG count from Metadata.ini for current component
                    extracted_component = self.extracted_object.components[component_id]
                    max_id = max(
                        int(vg.name)
                        for vg in obj.vertex_groups
                        if vg.name.isdigit()
                    ) if obj.vertex_groups else -1
                    total_vg_count = max_id + 1
                    # total_vg_count = len(extracted_component.vg_map)
                    ignore_list = [vg for vg in vertex_groups if 'ignore' in vg.name.lower() or vg.index >= total_vg_count]
                remove_vertex_groups(temp_obj, ignore_list)
                # Rename VGs to their indicies to merge ones of different components together
                for vg in get_vertex_groups(temp_obj):
                    vg.name = str(vg.index)

                # process_object(temp_obj)

                # Calculate vertex count of temporary object
                temp_object.vertex_count = len(temp_obj.data.vertices)
                # Calculate index count of temporary object, IB stores 3 indices per triangle
                temp_object.index_count = len(temp_obj.data.polygons) * 3
                # Set index offset of temporary object to global index_offset
                temp_object.index_offset = index_offset
                # Update global index_offset
                index_offset += temp_object.index_count
                # Update vertex and index count of custom component
                component.vertex_count += temp_object.vertex_count
                component.index_count += temp_object.index_count

    def remove_temp_objects(self):
        for component_id, component in enumerate(self.components):
            for temp_object in component.objects:
                remove_mesh(temp_object.object.data)

    def transform_merged_object(self, merged_object):
        change_scale = self.mesh_scale != 1.0
        change_rotation = self.mesh_rotation != (0.0, 0.0, 0.0)
        if not change_scale and not change_rotation:
            return
        # Compensate transforms we're about to set
        if change_scale:
            inverted_scale = 1 / self.mesh_scale
            merged_object.scale = inverted_scale, inverted_scale, inverted_scale
        if change_rotation:
            inverted_rotation = tuple([360 - r if r != 0 and r != 0 else 0 for r in self.mesh_rotation])
            merged_object.rotation_euler = to_radians(inverted_rotation)
        bpy.ops.object.transform_apply(location = False, rotation = True, scale = True)
        # Set merged object transforms
        if change_scale:
            merged_object.scale = self.mesh_scale, self.mesh_scale, self.mesh_scale
        if change_rotation:
            merged_object.rotation_euler = to_radians(self.mesh_rotation)

    def build_merged_object(self):

        merged_object = []
        vertex_count, index_count = 0, 0
        for component in self.components:
            for temp_object in component.objects:
                merged_object.append(temp_object.object)
            vertex_count += component.vertex_count
            index_count += component.index_count
            
        join_objects(self.context, merged_object)

        obj = merged_object[0]

        rename_object(obj, 'TEMP_EXPORT_OBJECT')

        deselect_all_objects()
        select_object(obj)
        set_active_object(bpy.context, obj)
        
        self.transform_merged_object(obj)

        mesh = obj.evaluated_get(self.context.evaluated_depsgraph_get()).to_mesh()

        self.merged_object = MergedObject(
            object=obj,
            mesh=mesh,
            components=self.components,
            vertex_count=len(obj.data.vertices),
            index_count=len(obj.data.polygons) * 3,
            vg_count=len(get_vertex_groups(obj)),
            shapekeys=MergedObjectShapeKeys(),
            skeleton_type=self.skeleton_type,
        )

        if vertex_count != self.merged_object.vertex_count:
            raise ValueError('vertex_count mismatch between merged object and its components')

        if index_count != self.merged_object.index_count:
            raise ValueError('index_count mismatch between merged object and its components')
