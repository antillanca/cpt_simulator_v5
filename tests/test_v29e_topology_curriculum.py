import pytest
import torch
from backend.circuits.graph_dataset import CircuitGraph
from backend.circuits.topology_curriculum import CurriculumLevel, determine_level, TopologyCurriculum

def make_dummy_graph(num_nodes: int, num_cycles: int) -> CircuitGraph:
    return CircuitGraph(
        node_features=torch.zeros(num_nodes, 13),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_features=torch.zeros(0, 7),
        target_voltages=torch.zeros(num_nodes),
        node_names=tuple([str(i) for i in range(num_nodes)]),
        fingerprint="dummy",
        cycle_matrix=torch.zeros(num_cycles, 1)
    )

def test_determine_level():
    # Level 0: <= 4 nodes, 0 cycles
    g0 = make_dummy_graph(3, 0)
    assert determine_level(g0) == CurriculumLevel.LEVEL_0_TRIVIAL
    g0_bound = make_dummy_graph(4, 0)
    assert determine_level(g0_bound) == CurriculumLevel.LEVEL_0_TRIVIAL

    # Level 1: <= 6 nodes, <= 1 cycle
    g1 = make_dummy_graph(5, 1)
    assert determine_level(g1) == CurriculumLevel.LEVEL_1_SIMPLE
    g1_nodes = make_dummy_graph(6, 0)
    assert determine_level(g1_nodes) == CurriculumLevel.LEVEL_1_SIMPLE

    # Level 2: <= 10 nodes, <= 3 cycles
    g2 = make_dummy_graph(8, 2)
    assert determine_level(g2) == CurriculumLevel.LEVEL_2_MEDIUM
    g2_cycles = make_dummy_graph(10, 3)
    assert determine_level(g2_cycles) == CurriculumLevel.LEVEL_2_MEDIUM

    # Level 3: Dense
    g3 = make_dummy_graph(11, 4)
    assert determine_level(g3) == CurriculumLevel.LEVEL_3_DENSE
    g3_cycles = make_dummy_graph(5, 4)
    assert determine_level(g3_cycles) == CurriculumLevel.LEVEL_3_DENSE

def test_topology_curriculum_filter():
    dataset = [
        make_dummy_graph(4, 0),   # 0
        make_dummy_graph(6, 1),   # 1
        make_dummy_graph(10, 3),  # 2
        make_dummy_graph(20, 5),  # 3
    ]
    
    assert len(TopologyCurriculum.filter_dataset(dataset, CurriculumLevel.LEVEL_0_TRIVIAL)) == 1
    assert len(TopologyCurriculum.filter_dataset(dataset, CurriculumLevel.LEVEL_1_SIMPLE)) == 2
    assert len(TopologyCurriculum.filter_dataset(dataset, CurriculumLevel.LEVEL_2_MEDIUM)) == 3
    assert len(TopologyCurriculum.filter_dataset(dataset, CurriculumLevel.LEVEL_3_DENSE)) == 4
