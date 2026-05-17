from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import torch

from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.graph_dataset import CircuitGraph, collate_graphs, circuit_to_graph
from backend.circuits.physics_loss import PhysicsInformedLoss
from backend.circuits.parser import parse_netlist
from backend.governance.artifact_registry import ArtifactRegistry
from backend.neural.models.circuit_gnn import EdgeAwareCircuitGNN
from scripts.generate_v29d_report import build_report, render_markdown
from scripts.run_circuit_arena import run_arena


def _current_source_circuit():
    return parse_netlist("I1 N1 0 0.002\nR1 N1 0 1000\n", name="current_source")


def _cycle_circuit():
    return parse_netlist("V1 N1 0 5\nR1 N1 N2 1000\nR2 N2 0 2000\nR3 N1 0 3000\n", name="cycle")


def _simple_graphs():
    circuits = [_current_source_circuit(), _cycle_circuit()]
    graphs = [circuit_to_graph(circuit, solve_dc_circuit(circuit)) for circuit in circuits]
    return graphs, circuits


def _write_dataset(path: Path) -> Path:
    rows = [
        {"circuit_name": "c1", "netlist": "I1 N1 0 0.002\nR1 N1 0 1000\n"},
        {"circuit_name": "c2", "netlist": "V1 N1 0 5\nR1 N1 N2 1000\nR2 N2 0 2000\nR3 N1 0 3000\n"},
        {"circuit_name": "c3", "netlist": "V1 N1 0 5\nR1 N1 0 1000\nR2 N1 0 2000\n"},
    ]
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cycle_matrix_is_deterministic():
    circuit = _cycle_circuit()
    graph_a = circuit_to_graph(circuit, solve_dc_circuit(circuit))
    graph_b = circuit_to_graph(circuit, solve_dc_circuit(circuit))
    torch.testing.assert_close(graph_a.cycle_matrix, graph_b.cycle_matrix)
    torch.testing.assert_close(graph_a.component_edge_index, graph_b.component_edge_index)


def test_cycle_matrix_shape_matches_component_edges():
    circuit = _cycle_circuit()
    graph = circuit_to_graph(circuit, solve_dc_circuit(circuit))
    assert graph.cycle_matrix.dim() == 2
    assert graph.cycle_matrix.size(1) == graph.component_edge_index.size(1)


def test_physics_loss_returns_scalar():
    graph, circuit = _simple_graphs()[0][0], _simple_graphs()[1][0]
    pred = graph.target_voltages.clone().requires_grad_(True)
    loss_fn = PhysicsInformedLoss()
    loss = loss_fn(pred, graph.target_voltages, graph, circuit=circuit)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_physics_loss_backward_has_gradients():
    graphs, circuits = _simple_graphs()
    graph = graphs[0]
    circuit = circuits[0]
    pred = graph.target_voltages.clone().requires_grad_(True)
    loss_fn = PhysicsInformedLoss()
    loss = loss_fn(pred, graph.target_voltages, graph, circuit=circuit)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_kcl_loss_zero_when_solution_is_consistent():
    circuit = _current_source_circuit()
    graph = circuit_to_graph(circuit, solve_dc_circuit(circuit))
    pred = graph.target_voltages.clone()
    loss_fn = PhysicsInformedLoss()
    loss = loss_fn.compute_kcl_loss(pred, graph, circuit=circuit)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-9


def test_kvl_loss_zero_when_solution_is_consistent():
    circuit = _cycle_circuit()
    graph = circuit_to_graph(circuit, solve_dc_circuit(circuit))
    pred = graph.target_voltages.clone()
    loss_fn = PhysicsInformedLoss()
    loss = loss_fn.compute_kvl_loss(pred, graph)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-9


def test_power_loss_zero_when_solution_is_balanced():
    circuit = _current_source_circuit()
    graph = circuit_to_graph(circuit, solve_dc_circuit(circuit))
    pred = graph.target_voltages.clone()
    loss_fn = PhysicsInformedLoss()
    loss = loss_fn.compute_power_loss(pred, graph, circuit=circuit)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-9


def test_empty_cycle_graph_is_stable():
    graph = CircuitGraph(
        node_features=torch.zeros(1, 8, dtype=torch.float32),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_features=torch.zeros(0, 4, dtype=torch.float32),
        target_voltages=torch.zeros(1, dtype=torch.float32),
        node_names=("N1",),
        fingerprint="empty",
        component_edge_index=torch.zeros(2, 0, dtype=torch.long),
        cycle_matrix=torch.zeros(0, 0, dtype=torch.float32),
    )
    loss_fn = PhysicsInformedLoss()
    pred = graph.target_voltages.clone()
    loss = loss_fn.compute_kvl_loss(pred, graph)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_graph_batch_padding_includes_cycle_matrix():
    graphs, _ = _simple_graphs()
    batch = collate_graphs(graphs)
    assert batch.cycle_matrix.dim() == 3
    assert batch.component_edge_index.dim() == 3
    assert batch.num_component_edges == tuple(g.component_edge_index.size(1) for g in graphs)
    assert batch.num_cycles == tuple(g.cycle_matrix.size(0) for g in graphs)


def test_deterministic_training_checkpoint_twice(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset.jsonl")
    out_a = tmp_path / "run_a.pt"
    out_b = tmp_path / "run_b.pt"
    snapshot = Path("workspace/training_snapshots/training_snapshot.json")
    snap_a = tmp_path / "snap_a.json"
    snap_b = tmp_path / "snap_b.json"

    cmd = [
        "python3",
        "-u",
        "scripts/train_circuit_gnn.py",
        "--config",
        "configs/training/kaggle_v29b.yaml",
        "--dataset",
        str(dataset),
        "--epochs",
        "1",
        "--output",
        str(out_a),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    shutil.copy2(snapshot, snap_a)

    cmd[-1] = str(out_b)
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    shutil.copy2(snapshot, snap_b)

    ckpt_a = torch.load(out_a, map_location="cpu", weights_only=False)
    ckpt_b = torch.load(out_b, map_location="cpu", weights_only=False)
    assert ckpt_a["artifact_fingerprint"] == ckpt_b["artifact_fingerprint"]
    assert ckpt_a["weights_hash"] == ckpt_b["weights_hash"]
    assert ckpt_a["optimizer_state_hash"] == ckpt_b["optimizer_state_hash"]
    assert ckpt_a["eval_fingerprint"] == ckpt_b["eval_fingerprint"]
    metrics_a = json.loads(out_a.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    metrics_b = json.loads(out_b.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    for m in [metrics_a, metrics_b]:
        if "history" in m:
            for entry in m["history"]:
                entry.pop("train_time_sec", None)
                entry.pop("eval_time_sec", None)
    assert metrics_a == metrics_b
    assert _sha256(snap_a) == _sha256(snap_b)


def test_arena_rerun_is_deterministic():
    torch.manual_seed(42)
    graphs, circuits = _simple_graphs()
    model = EdgeAwareCircuitGNN(node_dim=8, edge_dim=4, hidden_dim=16)
    results_a = run_arena(
        model,
        graphs,
        circuits,
        graphs,
        circuits,
        graphs[:1],
        circuits[:1],
        use_edge=True,
    )
    results_b = run_arena(
        model,
        graphs,
        circuits,
        graphs,
        circuits,
        graphs[:1],
        circuits[:1],
        use_edge=True,
    )
    assert results_a["gnn"]["in_distribution"] == results_b["gnn"]["in_distribution"]
    assert results_a["gnn"]["ood"] == results_b["gnn"]["ood"]
    assert results_a["mean_baseline"] == results_b["mean_baseline"]
    assert results_a["linear_baseline"] == results_b["linear_baseline"]
    assert results_a["random_baseline"] == results_b["random_baseline"]
    assert results_a["deterministic_rerun_validation"] == results_b["deterministic_rerun_validation"]
    assert results_a["gnn"]["replay_consistency_metrics"] == results_b["gnn"]["replay_consistency_metrics"]
    assert results_a["gnn"]["speed_in_distribution"]["speedup"] >= 0.0
    assert results_b["gnn"]["speed_in_distribution"]["speedup"] >= 0.0


def test_report_generation_builds_markdown():
    report = build_report()
    markdown = render_markdown(report)
    assert report["report_fingerprint"]
    assert "# CPT v2.9D Physics-Informed Comparison" in markdown
    assert "Power violation (64-case slice)" in markdown
    assert "Honest Assessment" in markdown


def test_artifact_registry_roundtrip_for_v29d(tmp_path):
    registry_path = tmp_path / "artifact_registry.json"
    registry = ArtifactRegistry(path=registry_path)
    report = build_report()
    registry.register(
        artifact_type="evaluation_report",
        schema_version="2.9d",
        fingerprint=report["report_fingerprint"],
        parent_fingerprints=[
            report.get("metrics_v29d", {}).get("checkpoint_fingerprint", ""),
            report.get("arena_v29d", {}).get("summary", {}).get("checkpoint_artifact_fingerprint", ""),
        ],
        metadata={"output": "docs/V29D_PHYSICS_INFORMED_COMPARISON.md"},
    )
    registry.save(registry_path)
    loaded = ArtifactRegistry.from_file(registry_path)
    assert len(loaded.records()) == 1
    assert loaded.records()[0].artifact_id


def test_physics_penalties_reduce_kcl():
    circuit = _current_source_circuit()
    graph = circuit_to_graph(circuit, solve_dc_circuit(circuit))
    pred_perfect = graph.target_voltages.clone()
    pred_bad = graph.target_voltages.clone() + 10.0
    loss_fn = PhysicsInformedLoss()
    loss_perfect = loss_fn.compute_kcl_loss(pred_perfect, graph, circuit=circuit).item()
    loss_bad = loss_fn.compute_kcl_loss(pred_bad, graph, circuit=circuit).item()
    assert loss_perfect < loss_bad


def test_physics_penalties_reduce_kvl():
    circuit = _cycle_circuit()
    graph = circuit_to_graph(circuit, solve_dc_circuit(circuit))
    pred_perfect = graph.target_voltages.clone()
    pred_bad = graph.target_voltages.clone()
    pred_bad[0] += 10.0
    loss_fn = PhysicsInformedLoss()
    loss_perfect = loss_fn.compute_kvl_loss(pred_perfect, graph).item()
    loss_bad = loss_fn.compute_kvl_loss(pred_bad, graph).item()
    assert loss_perfect <= loss_bad


def test_failure_taxonomy_deterministic():
    from backend.circuits.failure_analysis import classify_failure, compute_invariant_violations
    circuit = _current_source_circuit()
    graph = circuit_to_graph(circuit, solve_dc_circuit(circuit))
    pred = graph.target_voltages.clone()
    oracle_tensor = graph.target_voltages.clone()
    invariants = compute_invariant_violations(circuit, graph.node_names, pred)
    c1 = classify_failure(circuit, graph, pred, oracle_tensor, invariant_metrics=invariants, ood=False)
    c2 = classify_failure(circuit, graph, pred, oracle_tensor, invariant_metrics=invariants, ood=False)
    assert c1 == c2

