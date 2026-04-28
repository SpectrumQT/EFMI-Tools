import json
import time

from pathlib import Path
from operator import itemgetter
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from ..migoto_io.data_model.numpy_mesh import NumpyMesh, GeometryMatcher, VertexGroupsMatcher
from ..migoto_io.migoto_model.migoto_format import MigotoFormat

from .output_builder import ObjectData


@dataclass
class LODMatcher:
    full_model_path: Path

    geo_matcher_method: str
    geo_matcher_sensivity: float
    geo_matcher_voxel_size: float
    geo_matcher_sample_size: int

    geo_matcher_prefilter_voxel_size: float
    geo_matcher_prefilter_sample_size: int
    geo_matcher_prefilter_candidates_count: int
    vg_matcher_candidates_count: int

    vg_matcher: Optional[VertexGroupsMatcher] = None
    lod_model_path: Optional[Path] = None
    lod_objects: Optional[Dict[str, ObjectData]] = None

    def __post_init__(self):
        self.geo_matcher = GeometryMatcher(method=self.geo_matcher_method, sensivity=self.geo_matcher_sensivity)
        self.vg_matcher = VertexGroupsMatcher(candidates_count=self.vg_matcher_candidates_count)

        self.full_components: Dict[str, str] = {}
        self.lod_components: Dict[str, str] = {}

        self.lod_hash_to_name: Dict[str, str] = {}

        self.full_meshes: Dict[str, NumpyMesh] = {}
        self.lod_meshes: Dict[str, NumpyMesh] = {}

        self.matched: Dict[str, str] = {}

        self.vg_maps: Dict[str, Tuple[str, Dict[int, int]]] = {}

    # -------------------------
    # Loading
    # -------------------------

    def _load_component_hashes(self, path: Optional[Path] = None, json_str: str = '', object_id: str = '') -> Dict[str, str]:
        if path is not None:
            with open(path / 'Metadata.json', 'r') as f:
                metadata = json.load(f)
            return {
                f'Component {i}': component['vb0_hash']
                for i, component in enumerate(metadata['components'])
            }
        else:
            metadata = json.loads(json_str)
            return {
                f'{object_id} - Component {i}': component['vb0_hash']
                for i, component in enumerate(metadata['components'])
            }

    def load_metadata(self):
        self.full_components = self._load_component_hashes(self.full_model_path)
        if self.lod_model_path is not None:
            self.lod_components = self._load_component_hashes(self.lod_model_path)
        if self.lod_objects is not None:
            for object_id, obj_data in self.lod_objects.items():
                self.lod_components.update(self._load_component_hashes(json_str=obj_data.metadata, object_id=object_id))
        self.lod_hash_to_name = {hash: name for name, hash in self.lod_components.items()}

    def load_meshes(self):
        t0 = time.time()

        self.full_meshes = {
            name: NumpyMesh.from_paths(vb_path=self.full_model_path / f'{name}.vb')
            for name in self.full_components
        }

        if self.lod_model_path is not None:
            self.lod_meshes = {
                name: NumpyMesh.from_paths(vb_path=self.lod_model_path / f'{name}.vb')
                for name in self.lod_components
            }

        if self.lod_objects is not None:
            for object_id, obj_data in self.lod_objects.items():
                for component_id, component in enumerate(obj_data.components):
                    try:
                        fmt = MigotoFormat.from_fmt_text(component.fmt)
                        mesh = NumpyMesh.from_bytes(migoto_format=fmt, vb_bytes=component.vb, ib_bytes=component.ib)
                        self.lod_meshes[f'{object_id} - Component {component_id}'] = mesh
                    except Exception as e:
                        print(f'Failed to load mesh `{object_id} - Component {component_id}`: {component.ib_source.data.path}')
                        self.lod_meshes[f'{object_id} - Component {component_id}'] = None

        print(f'Models data load time: {time.time() - t0:.03f}s')

    # -------------------------
    # Matching
    # -------------------------

    def match_by_hash(self):
        for full_name, full_hash in self.full_components.items():
            lod_name = self.lod_hash_to_name.pop(full_hash, None)
            if lod_name is None:
                continue

            if self.lod_meshes[lod_name] is None:
                continue

            similarity = self.geo_matcher.calculate_similarity(
                self.full_meshes[full_name],
                self.lod_meshes[lod_name],
            )

            self.matched[full_hash] = lod_name

            print(
                f'{full_name} {full_hash} = {lod_name} {full_hash} '
                f'(by hash from {len(self.lod_hash_to_name)} candidates) '
                f'similarity={similarity:.2f}'
            )

    def match_by_geometry(self):
        
        def calculate_similarities(lod_hash_to_names):

            similarities = {}

            for lod_hash, lod_name in lod_hash_to_names.items():
                # if self.lod_meshes[lod_name] is None:
                #     continue
                similarity = self.geo_matcher.calculate_similarity(
                    full_mesh, self.lod_meshes[lod_name]
                )
                similarities[lod_hash] = similarity

            similarities = dict(
                sorted(similarities.items(), key=itemgetter(1), reverse=True)
            )

            return similarities
        
        for full_name, full_hash in self.full_components.items():

            if full_hash in self.matched:
                raise ValueError(f'Duplicate component vb0 hash {full_hash} found in Metadata.json!')

            full_mesh = self.full_meshes[full_name]
            
            best_lod_hash, best_similarity = None, None

            # Try to get LoD by full model hash

            best_lod_name = self.lod_hash_to_name.pop(full_hash, None)

            if best_lod_name is not None:

                similarity = self.geo_matcher.calculate_similarity(
                    self.full_meshes[full_name],
                    self.lod_meshes[best_lod_name],
                )

                print(
                    f'{full_name} {full_hash} = {best_lod_name} {full_hash} '
                    f'(by hash from {len(self.lod_hash_to_name)} candidates) '
                    f'similarity={similarity:.2f}'
                )
                
                best_similarity = similarity
                best_lod_hash = full_hash
                t_geo = 0

            else:

                # Try to get LoD by geometry matching

                t_geo = time.time()

                self.geo_matcher.samples_count = self.geo_matcher_prefilter_sample_size
                self.geo_matcher.voxel_size = self.geo_matcher_prefilter_voxel_size

                prefiltered_similarities = calculate_similarities(self.lod_hash_to_name)

                prefiltered_lod_hashes = list(prefiltered_similarities.keys())[:min(self.geo_matcher_prefilter_candidates_count, len(prefiltered_similarities))]

                lod_hash_to_names = {hash: self.lod_hash_to_name[hash] for hash in prefiltered_lod_hashes}
                
                self.geo_matcher.samples_count = self.geo_matcher_sample_size
                self.geo_matcher.voxel_size = self.geo_matcher_voxel_size

                similarities = calculate_similarities(lod_hash_to_names)

                t_geo = time.time() - t_geo

                best_lod_hash, best_similarity = next(iter(similarities.items()))
                best_lod_name = self.lod_hash_to_name.pop(best_lod_hash)

                print(
                    f'{full_name} {full_hash} = {best_lod_name} {best_lod_hash} {len(self.lod_meshes[best_lod_name].vertex_buffer)}'
                    f'(by geo from {len(self.lod_hash_to_name)} candidates) '
                    f'similarity={best_similarity:.2f}%, '
                )

            self.matched[full_hash] = (best_lod_name, best_lod_hash, best_similarity)
                
            # Match VGs

            t_vg = time.time()

            vg_map = self.vg_matcher.match_vertex_groups(
                full_mesh,
                self.lod_meshes[best_lod_name],
            )

            t_vg = time.time() - t_vg

            # Save result

            remapped = sum(1 for k, v in vg_map.items() if k != v)

            if remapped > 0:
                self.vg_maps[full_hash] = (best_lod_hash, vg_map)
                print(f'remapped VGs={remapped}/{len(vg_map) or 1}')
            else:
                print(f'all {len(vg_map)} VGs matched (LoD uses full skeleton)')

            print(f'    LoD meshes match time: {t_geo:.03f}s')
            print(f'    Vertex Groups match time: {t_vg:.03f}s')

    # -------------------------
    # Public API
    # -------------------------

    def run(self) -> Dict[str, str]:
        t0 = time.time()

        self.load_metadata()
        self.load_meshes()

        # self.match_by_hash()
        self.match_by_geometry()

        print(f'Total processing time: {time.time() - t0:.03f}s')
        return self.matched
