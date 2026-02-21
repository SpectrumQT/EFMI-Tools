import time
import re
import numpy
import bpy
import json


from typing import Tuple, List, Dict, Optional


from ..migoto_io.data_model.dxgi_format import DXGIFormat, DXGIType
from ..migoto_io.data_model.byte_buffer import Semantic, AbstractSemantic, BufferSemantic, BufferLayout, NumpyBuffer, MigotoFmt
from ..migoto_io.data_model.data_model import DataModel


class DataModelEFMI(DataModel):
    buffers_format: Dict[str, BufferLayout] = {
        'Index': BufferLayout([
            BufferSemantic(AbstractSemantic(Semantic.Index), DXGIFormat.R16_UINT, stride=6),
        ]),
        'Position': BufferLayout([
            BufferSemantic(AbstractSemantic(Semantic.Position, 0), DXGIFormat.R32G32B32_FLOAT),
            BufferSemantic(AbstractSemantic(Semantic.EncodedData, 0), DXGIFormat.R32_UINT),
        ]),
        'TexCoord': BufferLayout([
            BufferSemantic(AbstractSemantic(Semantic.TexCoord, 0), DXGIFormat.R32G32_FLOAT),
            BufferSemantic(AbstractSemantic(Semantic.Color, 0), DXGIFormat.R8G8B8A8_SNORM),
        ]),
        'Blend': BufferLayout([
            BufferSemantic(AbstractSemantic(Semantic.Blendweight, 0), DXGIFormat.R16_UNORM, stride=8),
            BufferSemantic(AbstractSemantic(Semantic.Blendindices, 0), DXGIFormat.R8_UINT, stride=4),
        ]),
    }

    def __init__(self):
        # self.flip_winding = True
        self.flip_bitangent_sign = True
        self.flip_texcoord_v = True
        # self.flip_normal = True
        self.semantic_converters = {
            # AbstractSemantic(Semantic.Normal, 0): [self.decode_tbn_data_10_10_10_2],
        }
        self.format_converters = {
            # AbstractSemantic(Semantic.Normal, 0): [self.decode_tbn_data_10_10_10_2],
        }
        self.semantic_encoders = {
            # Reshape flat array [[0,0,0],[0,0,0]] to [[0,0,0,1],[0,0,0,1]]
            AbstractSemantic(Semantic.Tangent, 0): [lambda data: self.converter_resize_second_dim(data, 4, fill=1)],
        }
        self.format_encoders = {
            # Normalize weights to 16-bit values, skip sanitizing since it's already done by DataExtractor
            AbstractSemantic(Semantic.Blendweight, 0): [lambda data: self.converter_normalize_weights(data, sanitize=False, dtype=numpy.uint16)],
            # Reshape flat array [0,1,2,3,4,5] to [[0,1,2],[3,4,5]]
            AbstractSemantic(Semantic.Index): [lambda data: self.converter_reshape_second_dim(data, 3)],
        }
    
    def set_data(self, 
                 obj: bpy.types.Mesh, 
                 mesh: bpy.types.Mesh, 
                 index_buffer: NumpyBuffer,
                 vertex_buffer: NumpyBuffer,
                 vg_remap: Optional[numpy.ndarray],
                 mirror_mesh: bool = False,
                 mesh_scale: float = 1.0,
                 mesh_rotation: Tuple[float] = (0.0, 0.0, 0.0),
                 import_tangent_data_to_attribute: bool = False):

        # Set import_format for NORMAL0 to prevent automatic addition of default format converter (one that would reshape array from 1 to 1,3)
        # buffer_semanic = vertex_buffer.layout.get_element(AbstractSemantic(Semantic.EncodedData))
        # buffer_semanic.import_format = DXGIFormat.R32G32B32_FLOAT
        encoded_data = vertex_buffer.get_field(AbstractSemantic(Semantic.EncodedData, semantic_index=0))

        if encoded_data is not None:
            try:
                decoded_normals = self.decode_tbn_data_10_10_10_2(encoded_data)
            except Exception as e:
                raise e
            
        # Execute real set_data from super class  
        vertex_ids = super().set_data(obj, mesh, index_buffer, vertex_buffer, vg_remap, mirror_mesh, mesh_scale, mesh_rotation)

        if encoded_data is not None:

            # Invert X coord of every vector in arrays required to mirror mesh
            if mirror_mesh:
                decoded_normals = self.converter_mirror_vector(decoded_normals)

            # Flip normals
            if self.flip_normal:
                decoded_normals = self.converter_flip_vector(decoded_normals)

            self.data_importer.import_normals(mesh, decoded_normals, vertex_ids)

            if import_tangent_data_to_attribute:
                # DEBUG: import encoded tangents as vertex attribute
                normals, encoded_tangents, bitangent_signs = self.decode_tbn_data_10_10_10_2(encoded_data, debug=True)

                negatives = numpy.where(encoded_tangents < 0, encoded_tangents, 0)
                positives = numpy.where(encoded_tangents > 0, encoded_tangents, 0)
                data = numpy.stack([negatives*-1, positives], axis=1)

                self.data_importer.import_attribute(mesh, BufferSemantic(AbstractSemantic(Semantic.Attribute), DXGIFormat.R32_FLOAT).get_name(), data)

    def get_data(self, 
                 context: bpy.types.Context, 
                 collection: bpy.types.Collection, 
                 obj: bpy.types.Object, 
                 mesh: bpy.types.Mesh, 
                 excluded_buffers: List[str],
                 buffers_format: Optional[Dict[str, BufferLayout]] = None,
                 mirror_mesh: bool = False,
                 object_index_layout: Optional[List[int]] = None) -> Tuple[Dict[str, NumpyBuffer], int, Optional[List[int]]]:

        if buffers_format is None:
            buffers_format = self.buffers_format

        build_blend_remaps = object_index_layout is not None and 'Blend' not in excluded_buffers

        # Request 16-bit VG ids for Blend Remap system
        if build_blend_remaps:
            # Number of VGs per vertex may vary based on buffers_format, we should respect it
            num_vgs = buffers_format['Blend'].get_element(AbstractSemantic(Semantic.Blendindices, 0)).get_num_values()
            buffers_format['BlendRemapVertexVG'] = BufferLayout([
                BufferSemantic(AbstractSemantic(Semantic.Blendindices, 1), DXGIFormat.R16_UINT, stride=num_vgs*2),
            ])

        # Request TBN data (tangents, bitangent signs and normals) signs for encoding
        buffers_format['TBN'] = BufferLayout([
            BufferSemantic(AbstractSemantic(Semantic.Tangent, 1), DXGIFormat.R32G32B32_FLOAT),
            BufferSemantic(AbstractSemantic(Semantic.BitangentSign, 1), DXGIFormat.R16_FLOAT),
            BufferSemantic(AbstractSemantic(Semantic.Normal, 1), DXGIFormat.R32G32B32_FLOAT),
            # BufferSemantic(AbstractSemantic(Semantic.Attribute, 0), DXGIFormat.R32G32B32A32_FLOAT),
        ])

        # Create color0 if not found
        for layout in buffers_format.values():
            color_semantic = AbstractSemantic(Semantic.Color, 0)
            if color_semantic in [x.abstract for x in layout.semantics]:
                if hasattr(mesh, 'vertex_colors') and not color_semantic.get_name() in mesh.vertex_colors:
                    # Legacy projects support
                    mesh.vertex_colors.new(name=color_semantic.get_name())
                else:
                    mesh.color_attributes.new(name=color_semantic.get_name(), type='FLOAT_COLOR', domain='CORNER')
                break

        index_data, vertex_buffer = self.export_data(context, collection, mesh, excluded_buffers, buffers_format, mirror_mesh, build_blend_remaps)

        # Remove TBN, we don't want to export it as buffer
        del buffers_format['TBN']

        vertex_ids = vertex_buffer.get_field(AbstractSemantic(Semantic.VertexId))

        tangents = vertex_buffer.get_field(AbstractSemantic(Semantic.Tangent, 0))

        if tangents is None:
                
            # Fill ENCODEDDATA0 field (encoded TBN)
            tangents = vertex_buffer.get_field(AbstractSemantic(Semantic.Tangent, 1))
            bitangent_signs = vertex_buffer.get_field(AbstractSemantic(Semantic.BitangentSign, 1))
            normals = vertex_buffer.get_field(AbstractSemantic(Semantic.Normal, 1))
            # attr0 = vertex_buffer.get_field(AbstractSemantic(Semantic.Attribute))
            if self.flip_texcoord_v:
                tangents *= -1
            encoded_data = self.encode_tbn_data_10_10_10_2(normals, tangents, bitangent_signs)
            vertex_buffer.set_field(AbstractSemantic(Semantic.EncodedData, 0), encoded_data)

        else:

            tangents[:, 3] = vertex_buffer.get_field(Semantic.BitangentSign)

        # positions = vertex_buffer.get_field(AbstractSemantic(Semantic.Position))
        # uvs = vertex_buffer.get_field(AbstractSemantic(Semantic.TexCoord))

        # # DEBUG: Tangents encoder test (write result to new vertex attribute)
        # def test_tangents_encoder(tangents, normals):
        #     data = self.encode_tangents(tangents, normals)
            
        #     negatives = numpy.where(data < 0, data, 0)
        #     positives = numpy.where(data > 0, data, 0)

        #     data = numpy.stack([negatives*-1, positives], axis=1)
        #     self._create_verterx_attribute('TANGENT_NEW_TEST', obj.name, data, vertex_ids)
        #     self._create_verterx_attribute('TANGENT_NEW_TEST', 'Component 10 122c46be.002', data, vertex_ids)
        # test_tangents_encoder(tangents, normals)

        # face_normals = numpy.array([poly.normal[:] for poly in mesh.polygons])
        # vertex_face_normals = numpy.zeros((len(mesh.vertices), 3), dtype=numpy.float32)
        # for poly, fn in zip(mesh.polygons, face_normals):
        #     for vi in poly.vertices:
        #         vertex_face_normals[vi] += fn
        # normals = vertex_face_normals[vertex_ids]

        # DEBUG: Zero-out COLOR0
        # color0 = vertex_buffer.get_field(AbstractSemantic(Semantic.Color))
        # color0 *= 0
        # vertex_buffer.set_field(AbstractSemantic(Semantic.Color), color0)

        # Assemble data into requested buffers
        buffers = self.build_buffers(index_data, vertex_buffer, excluded_buffers, buffers_format)

        if build_blend_remaps:
            blend_buffer = buffers.get('Blend', None)
            if blend_buffer is not None:
                index_buffer = buffers.get('Index', None)
                vg_buffer = buffers.get('BlendRemapVertexVG', None)
                blend_remaps = self.build_blend_remap(context, object_index_layout, index_buffer, blend_buffer, vg_buffer)
                buffers.update(blend_remaps)

        return buffers, len(vertex_ids)
    
    def decode_tbn_data_10_10_10_2(self, data: numpy.ndarray, debug: bool = False) -> numpy.ndarray:
        """
        Unpacks normals, encoded tangents and bitangent signs from R10G10B10A2_UINT
        X (10-bit) - octahedrally encoded normal
        Y (10-bit) - octahedrally encoded normal
        Z (10-bit) - encoded tangent
        W (2-bit) - encoded flags: packed data flag (bit30) and bitangent sign (bit31)
        Actually for import we need only normals, the rest is useful for debug
        """
        assert data.ndim == 1, 'Array for 10-10-10-2 decoding must be 1D'
        assert data.dtype == numpy.uint32, 'Array for 10-10-10-2 decoding must have dtype int32'

        data = self.converter_decode_10_10_10_2(data)

        # Make sure all bit 30 values are set to 1 (which signals shaders that data is packed)
        packed_flags =  data[:, 3]
        assert numpy.all(packed_flags == 1), 'NORMAL0 data is not 10-10-10-2 encoded!'

        normals = self.converter_oct_decode_vector(data[:, :2])

        if debug:
            encoded_tangents = data[:, 2]
            bitangent_signs = numpy.where(data[:, 4] == 1, 1, -1)
            return normals, encoded_tangents, bitangent_signs
        else:
            return normals

    @staticmethod
    def encode_tangents(tangents: numpy.ndarray, normals: numpy.ndarray):
        # Reference tangent
        R = numpy.stack([
            normals[:,1] - normals[:,2],
            normals[:,2] - normals[:,0],
            normals[:,0] - normals[:,1]
        ], axis=1)

        R_norm = numpy.linalg.norm(R, axis=1, keepdims=True)
        small_mask = R_norm[:,0] < 1e-6
        
        # Build perpendicular vector for degenerate cases
        if numpy.any(small_mask):
            helper = numpy.where(numpy.abs(normals[:,0:1]) < 0.9, numpy.array([1.0,0.0,0.0]), numpy.array([0.0,1.0,0.0]))
            v_perp = numpy.cross(normals, helper)
            v_perp /= numpy.linalg.norm(v_perp, axis=1, keepdims=True)

            # Select R
            R = numpy.where(small_mask[:,None], v_perp, R / R_norm)

        # Bitangent B = cross(R, N)
        B = numpy.cross(R, normals)

        # Project tangent onto the basis {R, B}
        cos_theta = numpy.sum(tangents * R, axis=1)
        sin_theta = numpy.sum(tangents * B, axis=1)

        # Clamp to avoid numerical issues
        cos_theta = numpy.clip(cos_theta, -1.0, 1.0)
        sin_theta = numpy.clip(sin_theta, -1.0, 1.0)

        # Compute parameter t in [0,1]
        denom = numpy.abs(cos_theta) + numpy.abs(sin_theta)
        u_t = cos_theta / denom
        t = 1 - (1 - u_t) / 2.0

        # Sign of sin (treat zero as positive)
        s = numpy.where(sin_theta == 0.0, 1.0, numpy.sign(sin_theta))
        t = numpy.copysign(t, s)

        return t

    @staticmethod
    def encode_tangents_debug(tangents: numpy.ndarray, normals: numpy.ndarray):
        # Reference tangent R = (Ny - Nz, Nz - Nx, Nx - Ny)
        R = numpy.stack([
            normals[:,1] - normals[:,2],
            normals[:,2] - normals[:,0],
            normals[:,0] - normals[:,1]
        ], axis=1)

        R_norm = numpy.linalg.norm(R, axis=1, keepdims=True)
        small_mask = R_norm[:, 0] < 1e-6

        # Normalise R, but handle degenerate cases where R is zero
        if numpy.any(small_mask):
            # For degenerate normals, build an arbitrary perpendicular vector
            abs_n = numpy.abs(normals)
            min_idx = numpy.argmin(abs_n, axis=1)   # index of smallest component
            v_perp = numpy.zeros_like(normals)

            mask0 = (min_idx == 0) & small_mask
            if numpy.any(mask0):
                v_perp[mask0, 0] = 0.0
                v_perp[mask0, 1] = -normals[mask0, 2]
                v_perp[mask0, 2] =  normals[mask0, 1]

            mask1 = (min_idx == 1) & small_mask
            if numpy.any(mask1):
                v_perp[mask1, 0] =  normals[mask1, 2]
                v_perp[mask1, 1] = 0.0
                v_perp[mask1, 2] = -normals[mask1, 0]

            mask2 = (min_idx == 2) & small_mask
            if numpy.any(mask2):
                v_perp[mask2, 0] = -normals[mask2, 1]
                v_perp[mask2, 1] =  normals[mask2, 0]
                v_perp[mask2, 2] = 0.0

            # Normalise the perpendicular vectors
            v_norm = numpy.linalg.norm(v_perp, axis=1, keepdims=True)
            v_perp = v_perp / v_norm
            R[small_mask] = v_perp[small_mask]

        # Now normalise R for the rest
        R[~small_mask] = R[~small_mask] / R_norm[~small_mask]

        # Bitangent B = cross(R, N)
        B = numpy.cross(R, normals, axis=1)

        # Project tangent onto the basis {R, B}
        cos_theta = numpy.sum(tangents * R, axis=1)
        sin_theta = numpy.sum(tangents * B, axis=1)

        # Clamp to avoid numerical issues
        cos_theta = numpy.clip(cos_theta, -1.0, 1.0)
        sin_theta = numpy.clip(sin_theta, -1.0, 1.0)

        # Compute parameter t in [0,1]
        denom = numpy.abs(cos_theta) + numpy.abs(sin_theta)
        u_t = cos_theta / denom
        t = (1.0 - u_t) / 2.0
        t = 1 - t
                
        # Sign of sin (treat zero as positive)
        s = numpy.where(sin_theta == 0.0, 1.0, numpy.sign(sin_theta))
        t = numpy.copysign(t, s)

        return t

    def encode_tbn_data_10_10_10_2(self, normals: numpy.ndarray, tangents: numpy.ndarray, bitangent_signs: numpy.ndarray) -> numpy.ndarray:
        """
        Packs normals, encoded tangents and bitangent_signs signs to R10G10B10A2_UINT
        X (10-bit) - octahedrally encoded normal
        Y (10-bit) - octahedrally encoded normal
        Z (10-bit) - encoded tangent
        W (2-bit) - encoded flags: packed data flag (bit30) and bitangent_signs sign (bit31)
        """
        assert normals.ndim == 2, 'Normals array for 2-10-10-10 encoding must be 2D'
        assert normals.shape[1] == 3, 'Normals array for 2-10-10-10 encoding must be with shape (N, 3)'
        assert numpy.issubdtype(normals.dtype, numpy.floating), 'Normals array for 2-10-10-10 decoding must have numpy.floating types'

        assert tangents.ndim == 2, 'Tangents array for 2-10-10-10 encoding must be 2D'
        assert tangents.shape[1] == 3, 'Tangents array for 2-10-10-10 encoding must be with shape (N, 3)'
        assert numpy.issubdtype(tangents.dtype, numpy.floating), 'Tangents array for 2-10-10-10 decoding must have numpy.floating types'

        assert bitangent_signs.ndim == 1, 'Bitangent signs array for 2-10-10-10 encoding must be 1D'
        assert numpy.issubdtype(bitangent_signs.dtype, numpy.floating), 'Bitangent signs array for 2-10-10-10 decoding must have numpy.floating types'

        # Octahedral normals encoding
        encoded_normals = self.converter_oct_encode_vector(normals)

        # Angle-based tangent encoding
        encoded_tangents = self.encode_tangents(tangents, normals)

        # Make array of `1` flags to mark data as "packed" for shaders to be aware
        packed_flags = numpy.ones(len(bitangent_signs))
        
        # Make array of `0` and `1` flags out of array of bitangent signs (consiting of -1 and 1)
        sign_flags = (bitangent_signs + 1) * 0.5

        # Encode data to R10G10B10A2_UINT
        data = numpy.stack([encoded_normals[:, 0], encoded_normals[:, 1], encoded_tangents, packed_flags, sign_flags], axis=1)
        encoded = self.converter_encode_10_10_10_2(data)

        return encoded

    def build_blend_remap(self, 
                         context: bpy.types.Context, 
                         index_layout: List[int], 
                         index_buffer: NumpyBuffer,
                         blend_buffer: NumpyBuffer,
                         vg_buffer: NumpyBuffer) -> Dict[str, NumpyBuffer]:
        
        start_time = time.time()

        remapped_vgs_counts = []

        if context.scene.efmi_tools_settings.index_data_cache:
            # Partial export is enabled and index buffer cache exists, lets load it
            index_data = numpy.array(json.loads(context.scene.efmi_tools_settings.index_data_cache)).ravel()
        else:
            if index_buffer is None:
                raise ValueError(f'Failed to build blend remap: `Index` buffer does not exist!')
            index_data = index_buffer.get_field(0).ravel()

        vg_ids = vg_buffer.get_field(vg_buffer.layout.get_element(AbstractSemantic(Semantic.Blendindices, 1)))
        vg_weights = blend_buffer.get_field(blend_buffer.layout.get_element(AbstractSemantic(Semantic.Blendweight, 0)))
        
        blend_remap_forward = numpy.empty(0, dtype=numpy.uint16)
        blend_remap_reverse = numpy.empty(0, dtype=numpy.uint16)

        index_offset = 0
        for index_count in index_layout:
            # Skip remapping the component if its custom mesh is empty
            if index_count == 0:
                remapped_vgs_counts.append(0)
                continue
    
            # Extract a segment of Index Buffer for the component (index_count number of indices starting from index_offset)
            vertex_ids = index_data[index_offset:index_offset+index_count]
            # Remove duplicate vertex ids (since multiple indices may reference the same vertex)
            vertex_ids = numpy.unique(vertex_ids)

            # Get VG ids used to weight vertices used in the component
            obj_vg_ids = vg_ids[vertex_ids].flatten()
            
            # Skip remapping the component if it references VG ids below 256 only
            if numpy.max(obj_vg_ids) < 256:
                index_offset += index_count
                remapped_vgs_counts.append(0)
                continue

            # Get weights for vertices referenced by the component
            obj_vg_weights = vg_weights[vertex_ids].flatten()
            # Get indices of non-zero weights (to skip remapping VG ids that are listed but not actually used)
            non_zero_idx = numpy.nonzero(obj_vg_weights > 0)[0]

            obj_vg_ids = obj_vg_ids[non_zero_idx]
            obj_vg_ids = numpy.unique(obj_vg_ids)

            if numpy.max(obj_vg_ids) < 256:
                index_offset += index_count
                remapped_vgs_counts.append(0)
                continue
            
            remapped_vgs_counts.append(len(obj_vg_ids))

            forward = numpy.zeros(512, dtype=numpy.uint16)
            forward[numpy.arange(len(obj_vg_ids))] = obj_vg_ids

            reverse = numpy.zeros(512, dtype=numpy.uint16)
            reverse[obj_vg_ids] = numpy.arange(len(obj_vg_ids))

            blend_remap_forward = numpy.concatenate((blend_remap_forward, forward), axis=0)
            blend_remap_reverse = numpy.concatenate((blend_remap_reverse, reverse), axis=0)

            index_offset += index_count

        buffers = {}

        buffers['BlendRemapForward'] = NumpyBuffer(BufferLayout([
            BufferSemantic(AbstractSemantic(Semantic.RawData, 0), DXGIFormat.R16_UINT),
        ]))
        buffers['BlendRemapReverse'] = NumpyBuffer(BufferLayout([
            BufferSemantic(AbstractSemantic(Semantic.RawData, 1), DXGIFormat.R16_UINT),
        ]))
        buffers['BlendRemapLayout'] = NumpyBuffer(BufferLayout([
            BufferSemantic(AbstractSemantic(Semantic.RawData, 2), DXGIFormat.R32_UINT),
        ]))

        buffers['BlendRemapForward'].set_data(blend_remap_forward)
        buffers['BlendRemapReverse'].set_data(blend_remap_reverse)
        buffers['BlendRemapLayout'].set_data(numpy.array(remapped_vgs_counts))

        print(f'Blend remap time: {time.time() - start_time :.3f}s ({int(len(blend_remap_forward) / 512)} remaps)')

        return buffers
