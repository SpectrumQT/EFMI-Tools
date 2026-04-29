from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path

from ..data_model.dxgi_format import *
from ..data_model.byte_buffer import NumpyBuffer, AbstractSemantic, Semantic
from .migoto_format import MigotoFormat
from .types import Topology


class WeightingType(Enum):
    Explicit = "EXPLICITLY_WEIGHTED"
    Implicit = "IMPLICITLY_WEIGHTED"
    NoWeights = "NOT_WEIGHTED"


@dataclass
class MigotoMesh:
    format: MigotoFormat | None = None
    index_buffer: NumpyBuffer | None = None
    vertex_buffer: NumpyBuffer | None = None
    cpu_posed: bool = False

    def get_data(self, semantic: AbstractSemantic | Semantic):
        if isinstance(semantic, Semantic):
            semantic = AbstractSemantic(semantic)
        if semantic.enum == Semantic.Index:
            if self.index_buffer is None:
                raise ValueError("Mesh has no index buffer")
            return self.index_buffer.get_field(semantic)
        else:
            return self.vertex_buffer.get_field(semantic)
        
    def get_weighting_type(self) -> WeightingType | None:
        if self.vertex_buffer is None:
            return None
        indices = self.vertex_buffer.layout.get_element(Semantic.Blendindices)
        weights = self.vertex_buffer.layout.get_element(Semantic.Blendweights)
        if indices and weights:
            return WeightingType.Explicit
        elif indices and not weights:
            return WeightingType.Implicit
        else:
            return WeightingType.NoWeights

    @classmethod
    def from_paths(
            cls,
            migoto_format: MigotoFormat | None = None,
            vb_path: Path | None = None,
            ib_path: Path | None = None,
            fmt_path: Path | None = None
    ) -> 'MigotoMesh':

        # Make migoto format from fmt file
        if migoto_format is None:
            migoto_format = MigotoFormat.from_paths(fmt_path, vb_path, ib_path)

        vb_bytes = None
        if vb_path is None:
            vb_path = cls.resolve_partner_path(ib_path or fmt_path, '.vb')
        if vb_path is not None:
            with open(vb_path, 'rb') as vb:
                vb_bytes = vb.read()

        ib_bytes = None
        if ib_path is None:
            ib_path = cls.resolve_partner_path(vb_path or fmt_path, '.ib')
        if ib_path is not None:
            with open(ib_path, 'rb') as ib:
                ib_bytes = ib.read()

        return cls.from_bytes(migoto_format, vb_bytes, ib_bytes)

    @staticmethod
    def resolve_partner_path(path: Path | None, partner_suffix: str) -> Path:
        if path is None:
            return None
        partner_path = path.with_suffix(partner_suffix)
        if not partner_path.is_file():
            return None
        return partner_path

    @classmethod
    def from_bytes(
            cls,
            migoto_format: MigotoFormat,
            vb_bytes: bytes | None = None,
            ib_bytes: bytes | None = None
    ) -> 'MigotoMesh':

        if migoto_format is None:
            raise ValueError('migoto_format is required')

        mesh = cls(format=migoto_format)

        if ib_bytes:
            try:
                mesh.index_buffer = NumpyBuffer(migoto_format.ib_layout)
                mesh.index_buffer.import_raw_data(ib_bytes)
            except Exception as e:
                raise ValueError(
                    f'failed to create index buffer from {len(ib_bytes)} bytes for layout stride {migoto_format.ib_layout.calculate_stride()}: {e}')

        if vb_bytes:
            try:
                mesh.vertex_buffer = NumpyBuffer(migoto_format.vb_layout)
                mesh.vertex_buffer.import_raw_data(vb_bytes)
            except Exception as e:
                raise ValueError(
                    f'failed to create vertex buffer from {len(vb_bytes)} bytes for layout stride {migoto_format.vb_layout.calculate_stride()}: {e}')

        return mesh

    @classmethod
    def from_numpy_buffers(
            cls,
            index_buffer: NumpyBuffer,
            vertex_buffer: NumpyBuffer,
            migoto_format: MigotoFormat | None = None,
            topology: Topology | None = None,
    ) -> 'MigotoMesh':

        mesh = cls(
            index_buffer=index_buffer,
            vertex_buffer=vertex_buffer,
            format=migoto_format,
        )

        try:
            mesh.validate()
        except Exception as e:
            raise ValueError(f"Failed to create MigotoMesh from NumpyBuffers: {e}")

        if migoto_format is None:

            mesh.format = MigotoFormat.from_layouts(
                topology=topology,
            )

            if index_buffer is not None:
                index_semantic = index_buffer.layout.get_element(AbstractSemantic(Semantic.Index, 0))
                mesh.format.ib_layout = mesh.index_buffer.layout
                mesh.format.index_count = len(mesh.index_buffer) * index_semantic.get_num_values()

            if vertex_buffer is not None:
                mesh.format.vb_layout = mesh.vertex_buffer.layout
                mesh.format.vertex_count = len(mesh.vertex_buffer)

        return mesh

    def validate(self):
        if self.index_buffer is not None:
            # Ensure IB contains only 1 semantic.
            if len(self.index_buffer.layout.semantics) != 1:
                raise ValueError(f"Index Buffer layout contains {len(self.index_buffer.layout.semantics)} semantics, while only 1 is allowed")
            # Ensure IB contains INDEX_0 semantic.
            index_semantic = self.index_buffer.layout.get_element(AbstractSemantic(Semantic.Index, 0))
            if index_semantic is None:
                raise ValueError(f"Index Buffer layout is missing INDEX_0 semantic")
        if self.vertex_buffer is not None:
            # Ensure VB contains at least 1 semantic.
            if len(self.vertex_buffer.layout.semantics) < 1:
                raise ValueError(f"Vertex Buffer layout contains {len(self.vertex_buffer.layout.semantics)} semantics, while at least 1 is required")

    def export_as_migoto_raw_buffers(self, folder_path: Path, mesh_name: str) -> None:
        output_path = folder_path / mesh_name
        if self.format:
            with open(output_path.with_suffix(".fmt"), 'w') as fmt:
                fmt.write(self.format.to_blender_addon_string())
        if self.index_buffer:
            with open(output_path.with_suffix(".ib"), 'wb') as ib:
                ib.write(self.index_buffer.get_bytes())
        if self.vertex_buffer:
            with open(output_path.with_suffix(".vb"), 'wb') as vb:
                vb.write(self.vertex_buffer.get_bytes())


class ChamferMixin:

    @staticmethod
    def calculate_linear_chamfer_distance(points_a: numpy.ndarray, points_b: numpy.ndarray) -> numpy.ndarray:
        """Calculates symmetric Chamfer distance using linear distances."""
        dist1 = VertexGroupsMatcher.calculate_min_distances(points_a, points_b)
        dist2 = VertexGroupsMatcher.calculate_min_distances(points_b, points_a)
        return numpy.mean(dist1) + numpy.mean(dist2)

    @staticmethod
    def calculate_min_distances(points_a: numpy.ndarray, points_b: numpy.ndarray, chunk_size=256):
        """Computes minimum distance for each vertex in points_a to any vertex in points_b in chunks to save memory."""
        cd_chunks = []
        for start in range(0, len(points_a), chunk_size):
            end = start + chunk_size
            diff_chunk = points_a[start:end, None, :] - points_b[None, :, :]
            dist_chunk = numpy.min(numpy.linalg.norm(diff_chunk, axis=2), axis=1)
            cd_chunks.append(dist_chunk)
        return numpy.concatenate(cd_chunks)


class VertexGroupsMatcher(ChamferMixin):
    def __init__(self, candidates_count=3):
        self.candidates_count = candidates_count

    def match_vertex_groups(self, mesh_a: MigotoMesh, mesh_b: MigotoMesh) -> dict[int, int]:
        """Maps vertex group IDs from mesh A to vertex group IDs of mesh B based on geometry.

        Supports multiple vertex groups per vertex (any 2nd dim size).
        Uses centroid pre-filtering to reduce number of Chamfer computations.

        Args:
            mesh_a: Source mesh whose vertex group IDs will be mapped.
            mesh_b: Target mesh whose vertex group IDs will be matched against.

        Returns:
            A dictionary mapping vertex group IDs from mesh A to the corresponding
            vertex group IDs of mesh B. Keys are vertex group IDs from mesh A, and
            values are the best-matching vertex group IDs from mesh B.
        """
        positions_a = mesh_a.get_data(Semantic.Position)
        indices_a = mesh_a.get_data(Semantic.Blendindices)
        weights_a = mesh_a.get_data(Semantic.Blendweights)

        positions_b = mesh_b.get_data(Semantic.Position)
        indices_b = mesh_b.get_data(Semantic.Blendindices)
        weights_b = mesh_b.get_data(Semantic.Blendweights)

        if indices_a is None:
            return {0: 0}

        # Remap zero-rows to virtual ID to simplify VG=0 handling
        # It allows to avoid branching for VG 0 with numpy.all instead of numpy.any
        indices_a, zero_id_a = self.remap_zero_rows(indices_a, weights_a)
        indices_b, zero_id_b = self.remap_zero_rows(indices_b, weights_b)

        # Unique VG IDs
        unique_indices_a = numpy.unique(indices_a[indices_a != 0])
        unique_indices_b = numpy.unique(indices_b[indices_b != 0])

        # Build point clouds for all VGs of mesh B
        points_list_b = [positions_b[numpy.any(indices_b == vg_b, axis=1)] for vg_b in unique_indices_b]
        centroids_b = numpy.array(
            [pts.mean(axis=0) if len(pts) > 0 else numpy.array([numpy.inf] * 3) for pts in points_list_b],
            dtype=numpy.float32
        )

        mapping = {}

        for vg_a in unique_indices_a:
            points_a = positions_a[numpy.any(indices_a == vg_a, axis=1)].astype(numpy.float32)
            if len(points_a) == 0:
                mapping[vg_a] = None
                continue

            # Compute mesh A centroid
            full_centroid = points_a.mean(axis=0)

            # Pre-filter VG candidates from mesh B using centroid distance
            dists = numpy.linalg.norm(centroids_b - full_centroid, axis=1)
            candidate_indices = numpy.argsort(dists)[:self.candidates_count]

            best_cd = numpy.inf
            best_vg_b = None

            for idx in candidate_indices:

                points_b = points_list_b[idx]
                if len(points_b) == 0:
                    continue

                cd = self.calculate_linear_chamfer_distance(points_a, points_b)
                if cd < best_cd:
                    best_cd = cd
                    best_vg_b = unique_indices_b[idx]

            vg_a, best_vg_b = int(vg_a), int(best_vg_b)

            vg_a = vg_a if vg_a != zero_id_a else 0
            best_vg_b = best_vg_b if best_vg_b != zero_id_b else 0

            mapping[vg_a] = best_vg_b

        return dict(sorted(mapping.items()))

    @staticmethod
    def remap_zero_rows(vg_ids: numpy.ndarray, vg_weights: numpy.ndarray) -> numpy.ndarray:
        """Returns a copy of vg_ids with VG 0 replaced with max VG ID + 1."""
        vg_ids = vg_ids.copy()
        virtual_id = int(vg_ids.max() + 1)
        # Create dummy weights array if not found (aka [[1, 0, 0, 0] * len(vg_ids)])
        if vg_weights is None:
            num_rows = vg_ids.shape[0]
            num_zeros = vg_ids.shape[1] - 1
            vg_weights = numpy.hstack([
                numpy.ones((num_rows, 1), dtype=numpy.uint8),
                numpy.zeros((num_rows, num_zeros), dtype=numpy.uint8)
            ])
        # Replace zeros where weight > 0
        mask = (vg_ids == 0) & (vg_weights > 0)
        vg_ids[mask] = virtual_id

        return vg_ids, virtual_id


class GeometryMatcherMethod(Enum):
    Voxel = "VOXEL"
    PointCloud = "POINT_CLOUD"


@dataclass
class GeometryMatcherConfig:
    method: GeometryMatcherMethod = GeometryMatcherMethod.Voxel
    sensitivity: float = 0.5
    voxel_size: float = 0.05
    samples_count: int = 5000


@dataclass
class GeometryMatcher(ChamferMixin):
    cfg: GeometryMatcherConfig

    def calculate_similarity(self, mesh_a: MigotoMesh, mesh_b: MigotoMesh) -> float:
        if self.cfg.method == GeometryMatcherMethod.Voxel:
            return self.calculate_similarity_voxel(mesh_a, mesh_b)
        if self.cfg.method == GeometryMatcherMethod.PointCloud:
            return self.calculate_similarity_point_cloud(mesh_a, mesh_b)
        raise ValueError(f'Unknown geometry matching method {self.cfg.method}!')

    def calculate_similarity_point_cloud(self, mesh_a: MigotoMesh, mesh_b: MigotoMesh) -> float:
        """Calculates similarity between Mesh A and Mesh B.

        Algo is based on average Chamfer distance between uniformly sampled triangles.
        """
        points_a = self.sample_points_on_mesh(mesh_a)
        points_b = self.sample_points_on_mesh(mesh_b)

        cd = self.calculate_linear_chamfer_distance(points_a, points_b)

        # Use bounding box diagonal as scale
        positions_a = mesh_a.get_data(Semantic.Position)
        scale = numpy.linalg.norm(positions_a.max(axis=0) - positions_a.min(axis=0))

        # Calculate similarity percentage with desired sensitivity
        similarity = max(0.0, 1 - cd / (scale * self.cfg.sensitivity)) * 100

        return similarity

    def sample_points_on_mesh(self, mesh: MigotoMesh) -> numpy.ndarray:
        """Uniformly samples points on a mesh surface using triangle areas."""
        indices = mesh.get_data(Semantic.Index)
        positions = mesh.get_data(Semantic.Position)

        v0 = positions[indices[:, 0]]
        v1 = positions[indices[:, 1]]
        v2 = positions[indices[:, 2]]

        # Triangle areas
        tri_areas = 0.5 * numpy.linalg.norm(numpy.cross(v1 - v0, v2 - v0), axis=1)
        tri_probs = tri_areas / numpy.sum(tri_areas)

        # Sample triangles proportional to area
        tri_indices = numpy.random.choice(len(indices), size=self.cfg.samples_count, p=tri_probs)

        # Barycentric coordinates
        r1 = numpy.sqrt(numpy.random.rand(self.cfg.samples_count))
        r2 = numpy.random.rand(self.cfg.samples_count)
        a = 1 - r1
        b = r1 * (1 - r2)
        c = r1 * r2

        sampled_points = a[:, None] * v0[tri_indices] + b[:, None] * v1[tri_indices] + c[:, None] * v2[tri_indices]

        return sampled_points

    def calculate_similarity_voxel(self, mesh_a, mesh_b):

        points_a = self.voxel_sample_mesh(mesh_a, voxel_size=self.cfg.voxel_size)
        points_b = self.voxel_sample_mesh(mesh_b, voxel_size=self.cfg.voxel_size)

        if len(points_a) == 0 or len(points_b) == 0:
            return 0.0

        d_ab = self.calculate_min_distances(points_a, points_b)
        d_ba = self.calculate_min_distances(points_b, points_a)

        mean_ab = d_ab.mean()
        mean_ba = d_ba.mean()

        chamfer = 0.5 * (mean_ab + mean_ba)
        asym = abs(mean_ab - mean_ba)

        coverage_tol = float(self.cfg.voxel_size)
        coverage = min(
            numpy.mean(d_ab < coverage_tol),
            numpy.mean(d_ba < coverage_tol)
        )

        raw = chamfer + 0.5 * asym

        similarity = max(0.0, 1.0 - raw / float(self.cfg.sensitivity))

        similarity *= (0.7 + 0.3 * coverage)

        return similarity * 100.0

    def voxel_sample_mesh(self, mesh, voxel_size=0.05):
        """
        Deterministic voxel-grid sampling of mesh surface.
        """
        positions = mesh.get_data(Semantic.Position)
        indices = mesh.get_data(Semantic.Index)

        tris = positions[indices]  # (T,3,3)

        # Sample triangle centers (cheap & deterministic)
        tri_centers = tris.mean(axis=1)

        # Normalize to unit bbox
        center = tri_centers.mean(axis=0)
        tri_centers = tri_centers - center

        bbox = tri_centers.max(axis=0) - tri_centers.min(axis=0)
        scale = numpy.linalg.norm(bbox)
        if scale > 0:
            tri_centers = tri_centers / scale

        # Voxelize
        vox = numpy.floor(tri_centers / voxel_size).astype(numpy.int32)

        # Unique voxels
        _, unique_idx = numpy.unique(vox, axis=0, return_index=True)
        sampled_points = tri_centers[unique_idx]

        return sampled_points
