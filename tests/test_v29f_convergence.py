import pytest
import torch
import numpy as np
from backend.circuits.models import Circuit, Resistor, VoltageSource
from backend.circuits.graph_dataset import circuit_to_graph
from backend.circuits.physics_projection import PhysicsProjection, ProjectionConfig
from backend.circuits.dc_solver import solve_dc_circuit

def _make_long_radial_chain(num_nodes: int, resistance: float = 100.0, voltage: float = 10.0) -> Circuit:
    resistors = []
    for i in range(1, num_nodes):
        a = str(i)
        b = str(i + 1) if i + 1 < num_nodes else "0"
        resistors.append(Resistor(f"R{i}", a, b, resistance))
    vs = VoltageSource("V1", positive="1", negative="0", voltage=voltage)
    return Circuit(name="long_radial_chain", ground_node="0", resistors=tuple(resistors), voltage_sources=(vs,))

def test_residual_decreases_monotonically():
    circuit = _make_long_radial_chain(15)
    solver = solve_dc_circuit(circuit)
    graph = circuit_to_graph(circuit, solver)
    
    init_v = torch.zeros_like(graph.target_voltages)
    
    cfg = ProjectionConfig(steps=8, virtual_node_enabled=True, virtual_conductance=0.1, blend_factor=0.5)
    proj = PhysicsProjection(cfg)
    
    metrics = proj.project_step_metrics(graph, circuit, init_v)
    residuals = [m["kcl_max_residual"] for m in metrics]
    
    for earlier, later in zip(residuals, residuals[1:]):
        assert later <= earlier + 1e-6, "Residual did not decrease monotonically"

def test_projection_stable_for_ood_graphs():
    # OOD generator typically generates random parameters
    # Just to simulate an OOD graph, we use a radial chain with extreme resistance
    circuit = _make_long_radial_chain(10, resistance=1e6, voltage=1000.0)
    solver = solve_dc_circuit(circuit)
    graph = circuit_to_graph(circuit, solver)
    
    # Random OOD-like init
    init_v = torch.randn_like(graph.target_voltages) * 100.0
    
    cfg = ProjectionConfig(steps=10, virtual_node_enabled=True)
    proj = PhysicsProjection(cfg)
    
    out_v = proj.project(graph, circuit, init_v)
    
    assert not torch.isnan(out_v).any()
    assert not torch.isinf(out_v).any()
    
    # Ensure it didn't explode
    assert torch.max(torch.abs(out_v)) < 1e6
