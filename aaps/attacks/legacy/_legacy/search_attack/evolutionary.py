"""Island-based MAP-Elites controller (OpenEvolve-inspired).

Implements the controller component from Appendix D of Nasr *et al.*,
"The Attacker Moves Second" (arXiv:2510.09023, bibkey
``nasr2025attacker``):
maintains separate islands of candidates with MAP-Elites grids,
enabling diverse exploration of the adversarial trigger space.

NOTE: This module is a standalone experimental utility. It is not imported
by the main search attack pipeline (``attack.py``), which implements its own
inline MAP-Elites archive and island migration logic. This class provides an
alternative implementation with different feature computation (lexical diversity
instead of strategy category for dimension 2) and configurable migration size.
"""

from typing import List, Dict, Any, Tuple, Optional
import random

from aaps.attacks.legacy._legacy.search_attack.fitness import FitnessFunction


class IslandMAPElites:
    """Island-based MAP-Elites for diverse adversarial trigger search."""

    def __init__(
        self,
        fitness_fn: FitnessFunction,
        num_islands: int = 2,
        feature_dims: Tuple[int, int] = (5, 6),
        migration_interval: int = 5,
        migration_size: int = 2,
    ):
        self.fitness_fn = fitness_fn
        self.num_islands = num_islands
        self.feature_dims = feature_dims
        self.migration_interval = migration_interval
        self.migration_size = migration_size

        self.islands: List[Dict[Tuple[int, int], Tuple[str, float, str]]] = [
            {} for _ in range(num_islands)
        ]
        self.generation = 0

    def compute_features(self, text: str) -> Tuple[int, int]:
        words = text.split()
        length_bin = min(len(words) // 10, self.feature_dims[0] - 1)
        unique = len(set(w.lower() for w in words)) / max(len(words), 1)
        diversity_bin = min(
            int(unique * self.feature_dims[1]), self.feature_dims[1] - 1
        )
        return (length_bin, diversity_bin)

    def add_solution(
        self,
        solution: str,
        fitness: float,
        feedback: str = "",
        island_idx: Optional[int] = None,
    ) -> bool:
        if island_idx is None:
            island_idx = hash(solution) % self.num_islands
        features = self.compute_features(solution)
        island = self.islands[island_idx]
        if features not in island or fitness > island[features][1]:
            island[features] = (solution, fitness, feedback)
            return True
        return False

    def select_parents(self, island_idx: int, k: int = 4) -> List[Tuple[str, float, str]]:
        island = self.islands[island_idx]
        if not island:
            return []
        elites = sorted(island.values(), key=lambda x: x[1], reverse=True)
        top = elites[: min(k, len(elites))]
        if len(elites) > k:
            extra = random.sample(elites[k:], min(2, len(elites) - k))
            top.extend(extra)
        return top

    def migrate(self):
        """Migrate top solutions between islands."""
        for src_idx in range(self.num_islands):
            dst_idx = (src_idx + 1) % self.num_islands
            src = self.islands[src_idx]
            if not src:
                continue
            top = sorted(src.values(), key=lambda x: x[1], reverse=True)
            for sol, fit, fb in top[: self.migration_size]:
                self.add_solution(sol, fit, fb, island_idx=dst_idx)

    def step(self):
        self.generation += 1
        if self.generation % self.migration_interval == 0:
            self.migrate()

    def get_global_best(self) -> Optional[Tuple[str, float, str]]:
        best = None
        for island in self.islands:
            for sol, fit, fb in island.values():
                if best is None or fit > best[1]:
                    best = (sol, fit, fb)
        return best

    def get_stats(self) -> Dict[str, Any]:
        total_cells = self.feature_dims[0] * self.feature_dims[1]
        island_sizes = [len(isl) for isl in self.islands]
        return {
            "generation": self.generation,
            "num_islands": self.num_islands,
            "island_sizes": island_sizes,
            "total_elites": sum(island_sizes),
            "coverage": sum(island_sizes) / (total_cells * self.num_islands),
        }
