import pytest
from backend.circuits.ood_stress_suite import (
    generate_dense_grid,
    generate_ladder_network,
    generate_cycle_dominant_loops,
)
from backend.circuits.failure_analysis import _is_connected

def test_dense_grid_generator():
    grid = generate_dense_grid(3, 4)
    assert grid.name == "stress_grid_3x4"
    assert len(grid.voltage_sources) == 1
    assert grid.ground_node == "0"
    # A 3x4 grid has 12 nodes. Ground is "0" (which was 3_4).
    # Remaining 11 active nodes.
    assert len(grid.all_nodes) == 11
    assert _is_connected(grid)

def test_ladder_network_generator():
    ladder = generate_ladder_network(8)
    assert ladder.name == "stress_ladder_8_stages"
    assert len(ladder.voltage_sources) == 1
    assert ladder.ground_node == "0"
    assert len(ladder.all_nodes) == 8
    assert _is_connected(ladder)

def test_cycle_dominant_loops_generator():
    loops = generate_cycle_dominant_loops(4, 4)
    assert loops.name == "stress_loops_4x4"
    assert len(loops.voltage_sources) == 1
    assert loops.ground_node == "0"
    # Interconnected cycle loop checks
    assert len(loops.all_nodes) > 0
    assert _is_connected(loops)
