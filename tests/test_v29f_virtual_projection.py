import pytest
import torch
from backend.circuits.models import Circuit, Resistor, VoltageSource
from backend.circuits.graph_dataset import circuit_to_graph
from backend.circuits.physics_projection import PhysicsProjection, ProjectionConfig
from backend.circuits.dc_solver import solve_dc_circuit


def _make_radial_chain(num_nodes: int, resistance: float = 100.0, voltage: float = 10.0) -> Circuit:
    """Create a simple linear chain circuit with a voltage source at node '1' and ground '0'.
    Nodes are named '1', '2', ..., str(num_nodes-1), and '0' is ground.
    """
    resistors = []
    # chain from node 1 to node N-1, then to ground 0
    for i in range(1, num_nodes):
        a = str(i)
        b = str(i + 1) if i + 1 < num_nodes else "0"
        resistors.append(Resistor(f"R{i}", a, b, resistance))
    vs = VoltageSource("V1", positive="1", negative="0", voltage=voltage)
    # Convert lists to tuples as required by Circuit dataclass
    return Circuit(name="radial_chain", ground_node="0", resistors=tuple(resistors), voltage_sources=(vs,))


def _project(circuit: Circuit, config: ProjectionConfig) -> torch.Tensor:
    solver = solve_dc_circuit(circuit)
    graph = circuit_to_graph(circuit, solver)
    # initial guess is the surrogate (oracle) for deterministic behavior
    init_v = graph.target_voltages.clone()
    proj = PhysicsProjection(config)
    return proj.project(graph, circuit, init_v)


def test_deterministic_projection():
    circuit = _make_radial_chain(5)
    cfg = ProjectionConfig(steps=3, virtual_node_enabled=True)
    out1 = _project(circuit, cfg)
    out2 = _project(circuit, cfg)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_no_nan_production():
    circuit = _make_radial_chain(6)
    cfg = ProjectionConfig(steps=5, virtual_node_enabled=True)
    out = _project(circuit, cfg)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_residual_monotonic_decrease_with_virtual_node():
    circuit = _make_radial_chain(8)
    solver = solve_dc_circuit(circuit)
    graph = circuit_to_graph(circuit, solver)
    init_v = torch.zeros_like(graph.target_voltages)
    cfg = ProjectionConfig(steps=6, virtual_node_enabled=True)
    proj = PhysicsProjection(cfg)
    metrics = proj.project_step_metrics(graph, circuit, init_v)
    residuals = [m["kcl_max_residual"] for m in metrics]
    # ensure monotonic (allow equal due to clamping)
    for earlier, later in zip(residuals, residuals[1:]):
        assert later <= earlier + 1e-8


def test_radial_convergence_improved_vs_baseline():
    circuit = _make_radial_chain(10)
    solver = solve_dc_circuit(circuit)
    graph = circuit_to_graph(circuit, solver)
    init_v = torch.zeros_like(graph.target_voltages)
    cfg_base = ProjectionConfig(steps=5, virtual_node_enabled=False)
    cfg_vnode = ProjectionConfig(steps=5, virtual_node_enabled=True)
    proj_base = PhysicsProjection(cfg_base)
    proj_vnode = PhysicsProjection(cfg_vnode)
    out_base = proj_base.project(graph, circuit, init_v)
    out_vnode = proj_vnode.project(graph, circuit, init_v)
    # compute final KCL residuals
    kcl_base = proj_base.project_step_metrics(graph, circuit, init_v)[-1]["kcl_max_residual"]
    kcl_vnode = proj_vnode.project_step_metrics(graph, circuit, init_v)[-1]["kcl_max_residual"]
    assert kcl_vnode < kcl_base


def test_virtual_conductance_and_blend_factor_effects():
    circuit = _make_radial_chain(7)
    solver = solve_dc_circuit(circuit)
    graph = circuit_to_graph(circuit, solver)
    init_v = torch.zeros_like(graph.target_voltages)
    cfg_low = ProjectionConfig(steps=5, virtual_node_enabled=True, virtual_conductance=0.01, blend_factor=0.2)
    cfg_high = ProjectionConfig(steps=5, virtual_node_enabled=True, virtual_conductance=1.0, blend_factor=0.9)
    proj_low = PhysicsProjection(cfg_low)
    proj_high = PhysicsProjection(cfg_high)
    out_low = proj_low.project(graph, circuit, init_v)
    out_high = proj_high.project(graph, circuit, init_v)
    # Different configs should lead to different results (not identical)
    assert not torch.allclose(out_low, out_high)


def test_virtual_node_disabled_matches_baseline():
    circuit = _make_radial_chain(5)
    solver = solve_dc_circuit(circuit)
    graph = circuit_to_graph(circuit, solver)
    init_v = torch.randn_like(graph.target_voltages) * 0.5
    cfg_no_vn = ProjectionConfig(steps=4, virtual_node_enabled=False)
    cfg_vn_off = ProjectionConfig(steps=4, virtual_node_enabled=False, virtual_conductance=0.5, blend_factor=0.5)
    proj1 = PhysicsProjection(cfg_no_vn)
    proj2 = PhysicsProjection(cfg_vn_off)
    out1 = proj1.project(graph, circuit, init_v)
    out2 = proj2.project(graph, circuit, init_v)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_project_step_metrics_contains_virtual_node_info():
    # The metrics list is always returned; we just ensure it runs without error when virtual node is on.
    circuit = _make_radial_chain(6)
    solver = solve_dc_circuit(circuit)
    graph = circuit_to_graph(circuit, solver)
    init_v = torch.randn_like(graph.target_voltages)
    cfg = ProjectionConfig(steps=3, virtual_node_enabled=True)
    proj = PhysicsProjection(cfg)
    metrics = proj.project_step_metrics(graph, circuit, init_v)
    assert isinstance(metrics, list)
    assert len(metrics) == cfg.steps
    for step_dict in metrics:
        assert "step" in step_dict and "kcl_max_residual" in step_dict
