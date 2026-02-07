import bpy
import bmesh

from operator import attrgetter, itemgetter


def assert_mesh(mesh):
    if isinstance(mesh, str):
        mesh = get_mesh(mesh)
    elif mesh not in bpy.data.meshes.values():
        raise ValueError('Not of mesh type: %s' % str(mesh))
    return mesh


def get_mesh(mesh_name):
    return bpy.data.meshes[mesh_name]


def remove_mesh(mesh):
    mesh = assert_mesh(mesh)
    bpy.data.meshes.remove(mesh, do_unlink=True)


def mesh_triangulate(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()


def get_vertex_groups_from_bmesh(bm: bmesh.types.BMesh):
    layer_deform = bm.verts.layers.deform.active
    return [sorted(vert[layer_deform].items(), key=itemgetter(1), reverse=True) for vert in bm.verts]
