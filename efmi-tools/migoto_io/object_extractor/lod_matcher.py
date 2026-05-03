import time
import re

from dataclasses import dataclass, field
from collections import defaultdict
from operator import itemgetter

from .migoto_object.migoto_object_builder import MigotoObject, MigotoComponent
from ..migoto_model.migoto_mesh import MigotoMesh, WeightingType
from ..migoto_model.migoto_mesh import GeometryMatcherConfig, GeometryMatcher, VertexGroupsMatcher


class LODMatcherError(Exception):
    pass


class ObjectLowSimilarityError(LODMatcherError):
    pass


class ComponentLowSimilarityError(LODMatcherError):
    pass


@dataclass
class SimilarityGraph:

    data: dict[MigotoComponent, dict[MigotoComponent, float]]

    def calculate_object_similarity(self) -> float:
        total_similarity = 0
        for lod_component, similarities in self.data.items():
            if not similarities:
                continue
            similarity = next(iter(similarities.values()))
            total_similarity += similarity
        weighted_similarity = total_similarity / len(self.data)
        return weighted_similarity

    def verify_endmin_similarity_graph(self):
        endmin_lod1_to_full_map = {
            "5c29f1fc": "3d9e52b8",
            "070d7b84": "5825df15",
            "2f3d2c97": "b1f947ec",
            "3fc2a3de": "bf3c08af",
            "9b189efd": "b3bf2e13",
            "7cdfa2a3": "b57bbb30",
        }

        for lod_component, similarities in self.data.items():
            lod_hash = lod_component.metadata.ib_hash
            full_hash = next(iter(similarities.keys())).metadata.ib_hash
            correct_full_hash = endmin_lod1_to_full_map.get(lod_hash, None)
            if correct_full_hash is None:
                continue
            if full_hash != correct_full_hash:
                raise ValueError(f"LOD {lod_hash} matched {full_hash}, while {correct_full_hash} was expected")
            else:
                print(f"LOD {lod_hash} matched {full_hash} as expected")


@dataclass
class LODMatcher:

    component_min_vertex_count: int
    component_hash_blacklist: str

    object_similarity_threshold: float
    component_similarity_threshold: float

    geo_matcher_main_config: GeometryMatcherConfig

    geo_matcher_prefilter_config: GeometryMatcherConfig
    geo_matcher_prefilter_candidates_count: int

    vg_matcher_candidates_count: int

    geo_matcher: GeometryMatcher = field(init=False)
    vg_matcher: VertexGroupsMatcher = field(init=False)

    def __post_init__(self):
        self.geo_matcher = GeometryMatcher(self.geo_matcher_main_config)
        self.vg_matcher = VertexGroupsMatcher(candidates_count=self.vg_matcher_candidates_count)

    def find_matching_lods(
        self,
        full_object: MigotoObject,
        lod_candidate_objects: list[MigotoObject],
    ) -> tuple[MigotoObject, dict[MigotoComponent, tuple[MigotoComponent, dict[int, int] | None]]]:
        t = time.time()

        lod_object_candidates = self.prefilter_lod_object_candidates(full_object, lod_candidate_objects)

        lod_object, hash_matched_components = self.find_lod_object_by_hash(full_object, lod_object_candidates)

        if lod_object is None:
            lod_object, object_similarity, similarity_graph = self.find_lod_object_by_similarity(full_object, lod_object_candidates)
            if object_similarity < self.object_similarity_threshold:
                raise ObjectLowSimilarityError(f"Best matching LoD for object {full_object.id} has {object_similarity:.2f}% similarity!")
        else:
            similarity_graph = self.match_components_by_similarity(full_object, lod_object, hash_matched_components)

        # similarity_graph.verify_endmin_similarity_graph()

        geo_matched_components = self.get_best_matching_components(similarity_graph)

        matched_components: dict[MigotoComponent, MigotoComponent] = (
            hash_matched_components | geo_matched_components
        )

        print(f'Meshes match time: {time.time()-t:.2f}s')

        vg_maps = self.remap_vertex_groups(matched_components)

        result: dict[MigotoComponent, tuple[MigotoComponent, dict[int, int] | None]] = {}

        for lod_component, full_component in matched_components.items():
            result[full_component] = (lod_component, vg_maps.get(lod_component))

        return lod_object, result

    def prefilter_lod_object_candidates(
        self,
        full_object: MigotoObject,
        lod_candidate_objects: list[MigotoObject],
    ) -> list[MigotoObject]:

        candidates = []

        component_hash_blacklist = set([x for x in re.split(r"[,; ]", self.component_hash_blacklist) if x])

        lod_hashes = {}
        for full_component in full_object.components:
            for lod in full_component.metadata.lods:
                lod_hashes[lod.ib_hash] = lod.lod_object_name

        for lod_object in lod_candidate_objects:
            # Skip object with 2+ times fewer components.
            if len(lod_object.components) < len(full_object.components) / 2:
                continue

            for lod_component in lod_object.components:

                # Check if lod_component hash is already imported from other lod object.
                known_lod_object = lod_hashes.get(lod_component.metadata.ib_hash, None)
                if known_lod_object is not None and known_lod_object != lod_object.id:
                    lod_component.metadata.mesh_name = f"Skipped Component ib={lod_component.metadata.ib_hash} (already imported from {known_lod_object})"
                    continue

                if lod_component.metadata.ib_hash in component_hash_blacklist:
                    lod_component.metadata.mesh_name = f"Skipped Component ib={lod_component.metadata.ib_hash} (component hash blacklisted)"
                    continue

                if lod_component.metadata.vertex_count < self.component_min_vertex_count:
                    lod_component.metadata.mesh_name = f"Skipped Component ib={lod_component.metadata.ib_hash} (vertex count below minimum)"
                    continue

            candidates.append(lod_object)
            
        return candidates

    def remap_vertex_groups(
        self,
        matched_components: dict[MigotoComponent, MigotoComponent]
    ) -> dict[MigotoComponent, dict[int, int]]:

        print(f"Remapping Vertex Groups for {len(matched_components)} components...")

        t = time.time()

        vg_maps = {}

        for lod_component, full_component in matched_components.items():
            vg_map = self.vg_matcher.match_vertex_groups(
                full_component.mesh,
                lod_component.mesh,
            )

            remapped = sum(1 for k, v in vg_map.items() if k != v)

            component_desc = f"{full_component.metadata.mesh_name} LoD (full={full_component.metadata.ib_hash}, lod={lod_component.metadata.ib_hash})"

            if remapped > 0:
                vg_maps[lod_component] = vg_map
                print(f"{component_desc}: {remapped} out of used {len(vg_map) or 1} VGs are different (LoD mesh uses simplified skeleton)")
            else:
                print(f"{component_desc}: all {len(vg_map)} VGs are identical (LoD mesh uses full skeleton)")

        print(f"Vertex Groups match time: {time.time() - t:.03f}s")

        return vg_maps

    def find_lod_object_by_hash(
        self,
        full_object: MigotoObject,
        lod_object_candidates: list[MigotoObject],
    ) -> tuple[MigotoObject | None, dict[MigotoComponent, MigotoComponent]]:

        full_by_hash = {component.metadata.ib_hash: component for component in full_object.components}

        lods: dict[MigotoObject, dict[MigotoComponent, MigotoComponent]] = {}

        for lod_object in lod_object_candidates:
            matches = {}

            for lod_component in lod_object.components:
                if lod_component.metadata.mesh_name.startswith("Skipped"):
                    continue

                full_component = full_by_hash.get(lod_component.metadata.ib_hash)

                if full_component is None:
                    continue

                matches[lod_component] = full_component

                similarity = self.geo_matcher.calculate_similarity(full_component.mesh, lod_component.mesh)

                print(f"Match by hash (mesh similarity: {similarity:.2f}%): {full_component.__repr__()} == {lod_component.__repr__()} ")

            if matches:
                lods[lod_object] = matches

        if not lods:
            return None, {}

        matched_lod_object = max(
            lods,
            key=lambda obj: len(lods[obj]),
        )

        return matched_lod_object, lods[matched_lod_object]

    def find_lod_object_by_similarity(
        self,
        full_object: MigotoObject,
        lod_object_candidates: list[MigotoObject],
    ) -> tuple[MigotoObject, float, SimilarityGraph]:

        lod_object_similarity_graphs = {}
        lod_object_similarities = {}

        for lod_object in lod_object_candidates:
            similarity_graph = self.calculate_similarity_graph(full_object.components, lod_object.components)
            lod_object_similarity_graphs[lod_object] = similarity_graph
            lod_object_similarities[lod_object] = similarity_graph.calculate_object_similarity()

        matched_lod_object = max(
            lod_object_similarity_graphs,
            key=lambda obj: lod_object_similarities[obj],
        )

        object_similarity = lod_object_similarities[matched_lod_object]
        similarity_graph = lod_object_similarity_graphs[matched_lod_object]

        return matched_lod_object, object_similarity, similarity_graph

    def calculate_component_similarities(
        self,
        component: MigotoComponent,
        candidates: list[MigotoComponent],
    ) -> dict[MigotoComponent, float]:
        mesh_similarities = {}

        for candidate_component in candidates:
            similarity = self.geo_matcher.calculate_similarity(candidate_component.mesh, component.mesh)
            mesh_similarities[candidate_component] = similarity

        mesh_similarities = dict(
            sorted(mesh_similarities.items(), key=itemgetter(1), reverse=True)
        )

        return mesh_similarities

    def calculate_similarity_graph(
        self,
        full_components: list[MigotoComponent],
        lod_components: list[MigotoComponent],
    ) -> SimilarityGraph:

        similarities = {}

        for lod_component in lod_components:
            if lod_component.metadata.mesh_name.startswith("Skipped"):
                continue

            self.geo_matcher.cfg = self.geo_matcher_prefilter_config

            valid_full_components = [
                full_component for full_component in full_components
                if full_component.metadata.vertex_count >= lod_component.metadata.vertex_count
            ]

            prefilter_similarities = self.calculate_component_similarities(lod_component, valid_full_components)

            self.geo_matcher.cfg = self.geo_matcher_main_config

            prefiltered_full_components = list(prefilter_similarities.keys())[:self.geo_matcher_prefilter_candidates_count]

            similarities[lod_component] = self.calculate_component_similarities(lod_component, prefiltered_full_components)

        return SimilarityGraph(data=similarities)

    def match_components_by_similarity(
        self,
        full_object: MigotoObject,
        lod_object: MigotoObject,
        matched_lod_to_full_components: dict[MigotoComponent, MigotoComponent],
    ) -> SimilarityGraph:

        # Exclude already matched full components from matching.
        full_components = [
            full_component for full_component in full_object.components
            if full_component not in matched_lod_to_full_components.values()
        ]

        # Exclude already matched lod components from matching.
        lod_components = [
            lod_component for lod_component in lod_object.components
            if lod_component not in matched_lod_to_full_components.keys()
        ]

        similarity_graph = self.calculate_similarity_graph(full_components, lod_components)

        return similarity_graph

    def get_best_matching_components(self, similarity_graph: SimilarityGraph) -> dict[MigotoComponent, MigotoComponent]:
        result = {}
        for lod_component, similarities in similarity_graph.data.items():
            full_component, similarity = next(iter(similarities.items()))
            if similarity < self.object_similarity_threshold:
                raise ComponentLowSimilarityError(f"Best matching LoD for {full_component.metadata.mesh_name} has {similarity:.2f}% similarity!")
            result[lod_component] = full_component
            print(f"Match by geometry (mesh similarity: {similarity:.2f}%): {full_component.__repr__()} == {lod_component.__repr__()} ")
        return result





