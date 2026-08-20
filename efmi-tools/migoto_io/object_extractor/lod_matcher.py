import time
import re
import numpy

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


class HungarianSolver:
    """Solve a dense linear assignment problem using the Hungarian algorithm.

    The solver minimizes the total cost of a rectangular cost matrix.

    Every row is assigned to a distinct column. If the input has more rows
    than columns, the matrix is transposed internally, so the returned
    assignment always contains ``min(rows, columns)`` pairs.

    Non-finite costs (NaN and +/-inf) represent forbidden assignments.
    A ValueError is raised when a complete assignment of the smaller dimension is not possible.

    Complexity:
        Time:  O(n^2 * m), where n <= m.
               O(n^3) for square matrices.
        Space: O(n + m) auxiliary space, excluding the cost matrix.
    """

    def __init__(self, cost: numpy.ndarray) -> None:
        cost = numpy.asarray(cost, dtype=numpy.float64)

        if cost.ndim != 2:
            raise ValueError("cost must be a 2D matrix")

        self._transposed = cost.shape[0] > cost.shape[1]
        self._cost = cost.T.copy() if self._transposed else cost.copy()

    @classmethod
    def maximize(cls, weights: numpy.ndarray) -> "HungarianSolver":
        """Create a solver for a maximum-weight assignment.

        Args:
            weights:
                A 2D array where ``weights[i, j]`` is the score for assigning row ``i`` to column ``j``.
                Non-finite values (NaN and +/-inf) represent forbidden assignments.

        Returns:
            A ``HungarianSolver`` configured to maximize the total weight.

        Raises:
            ValueError:
                If ``weights`` is not two-dimensional.

        Notes:
            The Hungarian algorithm is implemented as a minimization algorithm.
            Maximization is therefore converted to minimization using:

                cost = max(weight) - weight

            where ``max(weight)`` is taken over finite entries only.

            This transformation preserves the optimal assignment because
            every feasible edge is shifted by the same constant. 
            Forbidden edges remain infinite and are never considered by the algorithm.

            The returned solver is executed by calling ``solve()``:

                HungarianSolver.maximize(weights).solve()
        """
        weights = numpy.asarray(weights, dtype=numpy.float64)

        if weights.ndim != 2:
            raise ValueError("weights must be a 2D matrix")

        if weights.size == 0:
            return cls(weights)

        finite = numpy.isfinite(weights)
        max_weight = weights[finite].max()

        cost = numpy.full_like(weights, numpy.inf)
        cost[finite] = max_weight - weights[finite]

        return cls(cost)

    def solve(self) -> list[tuple[int, int]]:
        """Return a minimum-cost assignment.

        Returns:
            A list of ``(row, column)`` pairs using the indices of the original input matrix.

        Raises:
            ValueError:
                If no complete assignment exists.

        Notes:
            For an ``r x c`` matrix, the returned assignment contains
            ``min(r, c)`` pairs. If ``r > c``, exactly ``c`` rows are assigned.
        """
        if self._cost.size == 0:
            return []

        n_rows, n_columns = self._cost.shape

        # The implementation assumes n_rows <= n_columns. This invariant is
        # established by transposing the matrix in __init__ when necessary.
        assert n_rows <= n_columns

        row_potential = numpy.zeros(n_rows + 1)
        column_potential = numpy.zeros(n_columns + 1)

        # column_to_row[j] is the row currently matched to column j.
        # Index 0 is a sentinel used by the augmenting-path algorithm.
        column_to_row = numpy.zeros(n_columns + 1, dtype=numpy.int32)

        # predecessor[j] stores the previous column on the augmenting path ending at column j.
        predecessor = numpy.zeros(n_columns + 1, dtype=numpy.int32)

        for row in range(1, n_rows + 1):
            self._augment(
                row=row,
                row_potential=row_potential,
                column_potential=column_potential,
                column_to_row=column_to_row,
                predecessor=predecessor,
            )

        return self._restore_assignment(column_to_row)

    def _augment(
        self,
        *,
        row: int,
        row_potential: numpy.ndarray,
        column_potential: numpy.ndarray,
        column_to_row: numpy.ndarray,
        predecessor: numpy.ndarray,
    ) -> None:
        """Augment the current matching by one row.

        This is the shortest augmenting-path formulation of the Hungarian algorithm.
        The dual potentials maintain non-negative reduced costs.
        Each iteration expands the alternating tree until it reaches an unmatched column.
        """
        n_columns = self._cost.shape[1]

        column_to_row[0] = row

        # min_reduced_cost[j] is the smallest reduced cost currently known
        # for reaching column j from the alternating tree.
        min_reduced_cost = numpy.full(n_columns + 1, numpy.inf)

        # Columns already included in the current alternating tree.
        used = numpy.zeros(n_columns + 1, dtype=bool)

        current_column = 0

        while True:
            used[current_column] = True
            current_row = column_to_row[current_column]

            delta = numpy.inf
            next_column = 0

            for column in range(1, n_columns + 1):
                if used[column]:
                    continue

                reduced_cost = self._reduced_cost(
                    row=current_row,
                    column=column,
                    row_potential=row_potential,
                    column_potential=column_potential,
                )

                if reduced_cost < min_reduced_cost[column]:
                    min_reduced_cost[column] = reduced_cost
                    predecessor[column] = current_column

                if min_reduced_cost[column] < delta:
                    delta = min_reduced_cost[column]
                    next_column = column

            # No finite reduced-cost edge remains. Therefore the current
            # partial matching cannot be augmented to a complete matching.
            if not numpy.isfinite(delta):
                raise ValueError(
                    "no complete feasible assignment exists"
                )

            # Update the dual variables for the alternating tree. This keeps
            # reduced costs non-negative and makes the selected next edge tight, allowing the tree to grow.
            for column in range(n_columns + 1):
                if used[column]:
                    row_potential[column_to_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    min_reduced_cost[column] -= delta

            current_column = next_column

            # An unmatched column terminates the augmenting path.
            if column_to_row[current_column] == 0:
                break

        # Flip the matching along the discovered alternating path.
        while True:
            previous_column = predecessor[current_column]

            column_to_row[current_column] = (
                column_to_row[previous_column]
            )

            current_column = previous_column

            if current_column == 0:
                break

    def _reduced_cost(
        self,
        *,
        row: int,
        column: int,
        row_potential: numpy.ndarray,
        column_potential: numpy.ndarray,
    ) -> float:
        """Return the reduced cost of an edge.

        Non-finite input costs are treated as forbidden edges and therefore have infinite reduced cost.
        """
        cost = self._cost[row - 1, column - 1]

        if not numpy.isfinite(cost):
            return numpy.inf

        return (
            cost
            - row_potential[row]
            - column_potential[column]
        )

    def _restore_assignment(
        self,
        column_to_row: numpy.ndarray,
    ) -> list[tuple[int, int]]:
        """Convert the internal matching back to original coordinates."""
        assignments: list[tuple[int, int]] = []

        for internal_column in range(1, len(column_to_row)):
            internal_row = column_to_row[internal_column]

            if internal_row == 0:
                continue

            row = internal_row - 1
            column = internal_column - 1

            if self._transposed:
                # The internal matrix is cost.T, so swap the coordinates
                # when converting back to the caller's coordinate system.
                assignments.append((column, row))
            else:
                assignments.append((row, column))

        # The algorithm naturally produces assignments in column order.
        # Return them in row order for a deterministic public API.
        assignments.sort()

        return assignments


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

    def find_optimal_matching(
        self,
        min_similarity: float = 0.0,
    ) -> "SimilarityGraph":
        """
        Find the globally optimal one-to-one component matching by maximizing total similarity.

        Components with no feasible match, or whose similarity is below `min_similarity`, remain unmatched.
        """
        rows = list(self.data)
        columns = list({
            candidate
            for similarities in self.data.values()
            for candidate in similarities
        })

        if not rows or not columns:
            return SimilarityGraph({})

        column_indices = {component: i for i, component in enumerate(columns)}

        # Real edges below the threshold are forbidden.
        weights = numpy.full(
            (len(rows), len(columns) + len(rows)),
            min_similarity,
            dtype=numpy.float64,
        )

        weights[:, :len(columns)] = -numpy.inf

        for row_index, component in enumerate(rows):
            for candidate, similarity in self.data[component].items():
                if similarity >= min_similarity:
                    weights[row_index, column_indices[candidate]] = similarity

        assignments = HungarianSolver.maximize(weights).solve()

        matched_data = {
            rows[row_index]: {
                columns[column_index]: float(weights[row_index, column_index])
            }
            for row_index, column_index in assignments
            if column_index < len(columns)
        }

        return SimilarityGraph(matched_data)

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
    skip_components_below_similarity_threshold: bool

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

        print(f"Searching for matching LoD object among {len(lod_candidate_objects)} candidates...")
        
        t = time.time()

        lod_object_candidates = self.prefilter_lod_object_candidates(full_object, lod_candidate_objects)

        lod_object, hash_matched_components = self.find_lod_object_by_hash(full_object, lod_object_candidates)

        if lod_object is not None:
            print(f"Found matching LoD object by shared component hashes: {lod_object.id}")
        else:
            lod_object, object_similarity, similarity_graph = self.find_lod_object_by_similarity(full_object, lod_object_candidates)
            if object_similarity < self.object_similarity_threshold:
                raise ObjectLowSimilarityError(f"Best matching LoD for object {full_object.id} has {object_similarity:.2f}% similarity!")
            print(f"Found matching LoD object by geometrical similarity: {lod_object.id} ({object_similarity:.2f}%)")

        # similarity_graph.verify_endmin_similarity_graph()

        print(
            f"Matching {len(lod_object.components)} LoD object components "
            f"agaisnt {len(full_object.components)} full object components..."
        )
        
        if lod_object is not None:
            similarity_graph = self.match_components_by_similarity(full_object, lod_object, hash_matched_components)

        geo_matched_components = self.get_best_matching_components(similarity_graph)

        matched_components: dict[MigotoComponent, MigotoComponent] = (
            hash_matched_components | geo_matched_components
        )
            
        for lod_component in lod_object.components:
            if lod_component.metadata.mesh_name.startswith("Skipped"):
                continue
            if lod_component not in matched_components:
                lod_component.metadata.mesh_name = f"Skipped Component ib={lod_component.metadata.ib_hash} (no matching full component found)"

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
                if lod.ib_hash == full_component.metadata.ib_hash:
                    continue
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

                lod_component.metadata.mesh_name = self.make_matched_mesh_name(full_component, lod_component, "hash")

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
    
    def make_matched_mesh_name(self, full_component: MigotoComponent, lod_component: MigotoComponent, similarity: float | str) -> str:
        match_type = f"{similarity:.2f}%" if isinstance(similarity, float) else similarity
        mesh_name = f"{full_component.metadata.mesh_name} full={full_component.metadata.ib_hash} lod={lod_component.metadata.ib_hash} match={match_type}"
        if lod_component.metadata.ib_hash == full_component.metadata.ib_hash:
            if lod_component.metadata.vg_map:
                mesh_name += f" (full mesh, full skeleton)"
            else:
                mesh_name += f" (full mesh, simplified skeleton)"
        else:
            mesh_name += f" (simplified mesh and skeleton)"
        return mesh_name

    def get_best_matching_components(
        self,
        similarity_graph: SimilarityGraph,
    ) -> dict[MigotoComponent, MigotoComponent]:
        
        # Find the globally optimal one-to-one assignment.
        matched_graph = similarity_graph.find_optimal_matching(min_similarity=0.0
            # min_similarity=self.object_similarity_threshold
            # if self.skip_components_below_similarity_threshold
            # else 0.0
        )

        result = {}

        for lod, similarities in matched_graph.data.items():

            if not similarities:
                continue

            # maximum_weight_matching guarantees at most one match per LoD.
            full, similarity = next(iter(similarities.items()))

            if similarity < self.object_similarity_threshold:
                if self.skip_components_below_similarity_threshold:
                    print(
                        f"Skipped match by geometry below {self.object_similarity_threshold:.2f}% threshold "
                        f"(mesh similarity: {similarity:.2f}%): {full.__repr__()} == {lod.__repr__()}"
                    )
                    lod.metadata.mesh_name = (
                        f"Skipped Component ib={lod.metadata.ib_hash} (mesh similarity {similarity:.2f}% "
                        f"is below configured {self.object_similarity_threshold:.2f}% threshold)"
                    )
                    continue

                raise ComponentLowSimilarityError(
                    f"Best matching LoD for {full.metadata.mesh_name} has {similarity:.2f}% similarity!"
                )

            lod.metadata.mesh_name = self.make_matched_mesh_name(full, lod, similarity)
            result[lod] = full

            print(f"Match by geometry (mesh similarity: {similarity:.2f}%): {full.__repr__()} == {lod.__repr__()}")

        return result
