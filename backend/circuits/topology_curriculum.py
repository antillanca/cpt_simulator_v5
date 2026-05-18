"""Topology Curriculum scheduler for CPT v2.9E.

Provides deterministic difficulty levels for curriculum training based on
circuit graph topology (nodes, edges, cycles).
"""

from enum import IntEnum
from typing import List

from backend.circuits.graph_dataset import CircuitGraph


class CurriculumLevel(IntEnum):
    """Deterministic difficulty levels based on topology."""
    LEVEL_0_TRIVIAL = 0  # Chains, Stars (<= 4 nodes, 0 cycles)
    LEVEL_1_SIMPLE = 1   # Trees, Simple Meshes (<= 6 nodes, <= 1 cycle)
    LEVEL_2_MEDIUM = 2   # Medium Meshes, Bridges (<= 10 nodes, <= 3 cycles)
    LEVEL_3_DENSE = 3    # Dense Meshes, Random Graphs (everything else)


def determine_level(graph: CircuitGraph) -> CurriculumLevel:
    """Classify a circuit graph into a curriculum level.
    
    classification rules:
      - Nivel 0: Chains, Stars (<= 4 nodos, 0 cycles)
      - Nivel 1: Trees, Simple Meshes (<= 6 nodos, <= 1 cycle)
      - Nivel 2: Medium Meshes, Bridges (<= 10 nodos, <= 3 cycles)
      - Nivel 3: Dense Meshes, Random Graphs (everything else)
    """
    num_nodes = graph.node_features.size(0)
    num_cycles = graph.cycle_matrix.size(0) if graph.cycle_matrix.numel() > 0 else 0

    if num_nodes <= 4 and num_cycles == 0:
        return CurriculumLevel.LEVEL_0_TRIVIAL
    elif num_nodes <= 6 and num_cycles <= 1:
        return CurriculumLevel.LEVEL_1_SIMPLE
    elif num_nodes <= 10 and num_cycles <= 3:
        return CurriculumLevel.LEVEL_2_MEDIUM
    else:
        return CurriculumLevel.LEVEL_3_DENSE


class TopologyCurriculum:
    """Filters training datasets progressively by topological difficulty."""

    @staticmethod
    def filter_dataset(dataset: List[CircuitGraph], max_level: CurriculumLevel) -> List[CircuitGraph]:
        """Return only graphs whose difficulty level is <= max_level."""
        filtered = []
        for g in dataset:
            if determine_level(g) <= max_level:
                filtered.append(g)
        return filtered
