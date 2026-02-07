"""
High-level representation of mesh data contained in ib & vb buffers
Intended to work with 3dm buffers outside of Blender API
"""


import numpy as np

from ..data_model.byte_buffer import ByteBuffer, IndexBuffer, AbstractSemantic, Semantic


class MeshObject:
    def __init__(self, ib_buffer: IndexBuffer, vb_buffer: ByteBuffer):
        self.ib_buffer = ib_buffer
        self.vb_buffer = vb_buffer

    def get_vg_count(self):
        return max(self.vb_buffer.get_values(AbstractSemantic(Semantic.Blendindices))) + 1

    def get_face_count(self):
        return self.ib_buffer.num_elements

    def get_vertex_count(self):
        return self.vb_buffer.num_elements

    def get_face(self, face_id):
        return self.ib_buffer.get_element(face_id)

    def get_vertex(self, vertex_id):
        return self.vb_buffer.get_element(vertex_id)

    def get_face_vertex_ids(self, face):
        if isinstance(face, int):
            face = self.get_face(face)
        return face.get_value(AbstractSemantic(Semantic.Index))

    def get_vertex_position(self, vertex):
        if isinstance(vertex, int):
            vertex = self.get_vertex(vertex)
        return vertex.get_value(AbstractSemantic(Semantic.Position))

    def get_vertex_groups(self, vertex):
        if isinstance(vertex, int):
            vertex = self.get_vertex(vertex)
        return vertex.get_value(AbstractSemantic(Semantic.Blendindices))

    def get_weights(self, vertex):
        if isinstance(vertex, int):
            vertex = self.get_vertex(vertex)
        return vertex.get_value(AbstractSemantic(Semantic.Blendweight))

    def get_triangle_area(self, vertex_ids):
        triangle = np.array([[self.get_vertex_position(vertex_id) for vertex_id in vertex_ids]])
        return float(self.calc_area(triangle))

    @staticmethod
    def calc_normal(triangles):
        return np.cross(triangles[:,1] - triangles[:,0], triangles[:,2] - triangles[:,0], axis=1)

    def calc_area(self, triangles):
        return np.linalg.norm(self.calc_normal(triangles), axis=1) / 2
