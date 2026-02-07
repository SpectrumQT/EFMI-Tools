"""
Engine ported from Weight Match Blender Addon by Comilarex (https://gamebanana.com/tools/15699)
Intended to work with 3dm buffers outside of Blender API
"""

import numpy as np

from .mesh_object import MeshObject


def match_vertex_groups(base_obj: MeshObject, target_obj: MeshObject):
    """
    Evaluates VG layout of two MeshObject's and returns VG map matching VGs of base object to VGs of target object
    """
    vg_map = {}
    
    # Precompute centers for all target vertex groups
    target_centers = {}
    target_influence_area = calculate_vertex_influence_area(target_obj)
    for target_group_id in range(target_obj.get_vg_count()):
        target_centers[target_group_id] = get_weighted_center(target_obj, target_group_id, target_influence_area)

    # Perform the matching and renaming process
    base_influence_area = calculate_vertex_influence_area(base_obj)
    for base_group_id in range(base_obj.get_vg_count()):
        base_center = get_weighted_center(base_obj, base_group_id, base_influence_area)
        if base_center is None:
            continue

        best_match = None
        best_distance = float('inf')

        for target_group_name, target_center in target_centers.items():
            if target_center is None:
                continue

            distance = np.linalg.norm(base_center - target_center)
            if distance < best_distance:
                best_distance = distance
                best_match = target_group_name

        if best_match is not None:
            if base_group_id != best_match:
                vg_map[base_group_id] = best_match
        else:
            vg_map[base_group_id] = -1

    return vg_map


def calculate_vertex_influence_area(obj: MeshObject):
    vertex_area = [0.0] * obj.get_vertex_count()

    for face_id in range(obj.get_face_count()):
        # Assuming the area is evenly distributed among the vertices
        vertex_ids = obj.get_face_vertex_ids(face_id)
        area_per_vertex = obj.get_triangle_area(vertex_ids) / 3
        for vertex_id in vertex_ids:
            vertex_area[vertex_id] += area_per_vertex

    return vertex_area


def get_weighted_center(obj: MeshObject, vg_id, vertex_influence_area):
    total_weight_area = 0.0
    weighted_position_sum = np.array((0.0, 0.0, 0.0))

    for vertex_id in range(obj.get_vertex_count()):
        vertex = obj.get_vertex(vertex_id)

        vertex_groups = obj.get_vertex_groups(vertex)
        weights = obj.get_weights(vertex)
        weight = 0.0
        for idx, vertex_group in enumerate(vertex_groups):
            if vertex_group == vg_id:
                weight += weights[idx]

        influence_area = vertex_influence_area[vertex_id]
        weight_area = weight * influence_area

        if weight_area > 0:
            weighted_position_sum += np.array(obj.get_vertex_position(vertex)) * weight_area
            total_weight_area += weight_area

    if total_weight_area > 0:
        return weighted_position_sum / total_weight_area
    else:
        return None
