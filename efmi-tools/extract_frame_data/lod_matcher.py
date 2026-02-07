import json
import time

from pathlib import Path
from operator import itemgetter
from typing import Optional, Dict, Tuple

from ..migoto_io.data_model.byte_buffer import MigotoFormat
from ..migoto_io.data_model.numpy_mesh import NumpyMesh, GeometryMatcher, VertexGroupsMatcher

from .output_builder import ObjectData


class LODMatcher:
    def __init__(
        self,
        geo_matcher: GeometryMatcher,
        vg_matcher: VertexGroupsMatcher,
        full_model_path: Path,
        lod_model_path: Optional[Path] = None,
        lod_objects: Optional[Dict[str, ObjectData]] = None,
    ):
        self.geo_matcher = geo_matcher
        self.vg_matcher = vg_matcher
        self.full_model_path = full_model_path
        self.lod_model_path = lod_model_path
        self.lod_objects: Optional[Dict[str, ObjectData]] = lod_objects

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
        for full_name, full_hash in self.full_components.items():

            if full_hash in self.matched:
                continue

            full_mesh = self.full_meshes[full_name]
            similarities = {}

            t_geo = time.time()

            for lod_hash, lod_name in self.lod_hash_to_name.items():
                if self.lod_meshes[lod_name] is None:
                    continue
                similarity = self.geo_matcher.calculate_similarity(
                    full_mesh, self.lod_meshes[lod_name]
                )
                similarities[lod_hash] = similarity

            similarities = dict(
                sorted(similarities.items(), key=itemgetter(1), reverse=True)
            )
            t_geo = time.time() - t_geo

            best_lod_hash, best_similarity = next(iter(similarities.items()))
            best_lod_name = self.lod_hash_to_name.pop(best_lod_hash)

            self.matched[full_hash] = best_lod_name

            t_vg = time.time()
            vg_map = self.vg_matcher.match_vertex_groups(
                full_mesh,
                self.lod_meshes[best_lod_name],
            )
            self.vg_maps[full_hash] = (best_lod_hash, vg_map, best_similarity)
            t_vg = time.time() - t_vg

            remapped = sum(1 for k, v in vg_map.items() if k != v)

            print(
                f'{full_name} {full_hash} = {best_lod_name} {best_lod_hash} {len(self.lod_meshes[best_lod_name].vertex_buffer)}'
                f'(by geo from {len(self.lod_hash_to_name)} candidates) '
                f'similarity={best_similarity:.2f}%, '
                f'remapped VGs={remapped}/{len(vg_map) or 1}'
            )
            print(f'    LoD meshes match time: {t_geo:.03f}s')
            print(f'    Vertex Groups match time: {t_vg:.03f}s')

    # -------------------------
    # Public API
    # -------------------------

    def run(self) -> Dict[str, str]:
        t0 = time.time()

        self.load_metadata()
        self.load_meshes()

        self.match_by_hash()
        self.match_by_geometry()

        print(f'Total processing time: {time.time() - t0:.03f}s')
        return self.matched
