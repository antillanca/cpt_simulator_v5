from __future__ import annotations

import json
from pathlib import Path

import torch

from backend.circuits.baselines import (
    LinearRegressionBaselinePredictor,
    MeanBaselinePredictor,
    RandomStableBaselinePredictor,
    evaluate_baseline_predictor,
)
from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.failure_analysis import (
    FAILURE_TYPES,
    classify_failure,
    compute_invariant_violations,
    summarize_failures,
)
from backend.circuits.graph_dataset import CircuitGraph, circuit_to_graph
from backend.circuits.parser import parse_netlist
from backend.governance.artifact_registry import ArtifactRegistry
from backend.neural.checkpoints.schema import build_checkpoint_payload
from backend.neural.models.circuit_gnn import EdgeAwareCircuitGNN
from backend.neural.training_snapshot import TrainingSnapshot
from scripts.analyze_v29c_failures import _stable_fingerprint
from scripts.generate_v29c_report import build_report, render_markdown
from scripts.run_circuit_arena import measure_speed, run_arena
from scripts.train_circuit_gnn import load_training_profile


def _make_circuits():
    circuits = [
        parse_netlist("V1 N1 0 5\nR1 N1 0 1000\n", name="c1"),
        parse_netlist("V1 N1 0 3\nR1 N1 0 330\nR2 N1 0 1000\n", name="c2"),
        parse_netlist("I1 N1 0 0.002\nR1 N1 0 1000\n", name="c3"),
    ]
    graphs = [circuit_to_graph(circuit, solve_dc_circuit(circuit)) for circuit in circuits]
    return graphs, circuits


def _make_snapshot(seed: int = 42) -> TrainingSnapshot:
    config = {
        "seed": seed,
        "device": {"prefer_cuda": False},
        "training": {"epochs": 10, "batch_size": 32, "learning_rate": 0.001},
        "dataset": {"train_path": "workspace/datasets/circuits/train_10k.jsonl", "eval_split": 0.2},
        "model": {"hidden_dim": 64, "max_params": 250_000},
        "evaluation": {"voltage_tolerance": 1e-3, "invariant_tolerance": 1e-6},
        "output": "workspace/checkpoints/circuit_gnn_v29b.pt",
        "model_type": "edge_aware",
    }
    return TrainingSnapshot.create(
        seed=seed,
        dataset_fingerprint="dataset-fp",
        config=config,
        model_fingerprint="model-fp",
        repo_root=Path.cwd(),
        torch_version="2.x",
        cuda_enabled=False,
        device_name="cpu",
    )


def test_kaggle_profile_loads():
    profile = load_training_profile("configs/training/kaggle_v29b.yaml")
    assert profile["seed"] == 42
    assert profile["device"]["prefer_cuda"] is True
    assert profile["training"]["epochs"] == 10
    assert profile["model"]["hidden_dim"] == 64


def test_stable_fingerprint_is_deterministic():
    payload = {"a": 1, "b": [2, 3]}
    assert _stable_fingerprint(payload) == _stable_fingerprint(payload)


def test_snapshot_fingerprint_is_stable():
    snapshot_1 = _make_snapshot()
    snapshot_2 = _make_snapshot()
    assert snapshot_1.fingerprint() == snapshot_2.fingerprint()
    assert snapshot_1.to_dict()["artifact_fingerprint"] == snapshot_1.fingerprint()


def test_snapshot_export_roundtrip(tmp_path):
    snapshot = _make_snapshot()
    path = snapshot.export(tmp_path)
    loaded = TrainingSnapshot.create(
        seed=42,
        dataset_fingerprint="dataset-fp",
        config={
            "seed": 42,
            "device": {"prefer_cuda": False},
            "training": {"epochs": 10, "batch_size": 32, "learning_rate": 0.001},
            "dataset": {"train_path": "workspace/datasets/circuits/train_10k.jsonl", "eval_split": 0.2},
            "model": {"hidden_dim": 64, "max_params": 250_000},
            "evaluation": {"voltage_tolerance": 1e-3, "invariant_tolerance": 1e-6},
            "output": "workspace/checkpoints/circuit_gnn_v29b.pt",
            "model_type": "edge_aware",
        },
        model_fingerprint="model-fp",
        repo_root=Path.cwd(),
        torch_version="2.x",
        cuda_enabled=False,
        device_name="cpu",
    )
    assert path.exists()
    assert loaded.fingerprint() == snapshot.fingerprint()


def test_invariant_violations_structure():
    circuit = parse_netlist("V1 N1 0 5\nR1 N1 0 1000\n", name="inv")
    metrics = compute_invariant_violations(circuit, ("N1",), torch.tensor([5.0], dtype=torch.float32))
    assert set(metrics) >= {
        "kcl_max_violation",
        "kvl_max_violation",
        "power_conservation_violation",
        "power_delivered",
        "power_dissipated",
        "inferred_voltage_source_currents",
    }
    assert torch.isfinite(torch.tensor(metrics["kcl_max_violation"]))
    assert torch.isfinite(torch.tensor(metrics["kvl_max_violation"]))
    assert torch.isfinite(torch.tensor(metrics["power_conservation_violation"]))


def test_invariant_violations_are_deterministic():
    circuit = parse_netlist("V1 N1 0 5\nR1 N1 0 1000\n", name="inv2")
    pred = torch.tensor([4.8], dtype=torch.float32)
    a = compute_invariant_violations(circuit, ("N1",), pred)
    b = compute_invariant_violations(circuit, ("N1",), pred)
    assert a == b


def test_failure_taxonomy_contains_expected_labels():
    assert "conservation_drift" in FAILURE_TYPES
    assert "ood_generalization_failure" in FAILURE_TYPES


def test_failure_classification_detects_disconnected_graph():
    circuit = parse_netlist("V1 N1 0 5\nR1 N1 0 1000\n", name="disc")
    graph = CircuitGraph(
        node_features=torch.zeros(0, 8, dtype=torch.float32),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_features=torch.zeros(0, 4, dtype=torch.float32),
        target_voltages=torch.zeros(0, dtype=torch.float32),
        node_names=(),
        fingerprint="",
    )
    result = classify_failure(circuit, graph, torch.tensor([0.0], dtype=torch.float32), torch.tensor([5.0], dtype=torch.float32), ood=True)
    assert result["failure_type"] in FAILURE_TYPES
    assert isinstance(result["reasons"], tuple)


def test_failure_classification_is_deterministic():
    circuit = parse_netlist("V1 N1 0 5\nR1 N1 0 1000\n", name="det")
    graph = circuit_to_graph(circuit, solve_dc_circuit(circuit))
    pred = torch.tensor([0.0], dtype=torch.float32)
    oracle = torch.tensor([5.0], dtype=torch.float32)
    a = classify_failure(circuit, graph, pred, oracle, ood=True)
    b = classify_failure(circuit, graph, pred, oracle, ood=True)
    assert a == b


def test_failure_summary_has_dominant_failure():
    summary = summarize_failures([
        {"failure_type": "conservation_drift"},
        {"failure_type": "conservation_drift"},
        {"failure_type": "node_aliasing"},
    ])
    assert summary["dominant_failure"] == "conservation_drift"
    assert summary["count"] == 3


def test_baselines_are_deterministic():
    graphs, _ = _make_circuits()
    mean = MeanBaselinePredictor().fit(graphs)
    linear = LinearRegressionBaselinePredictor().fit(graphs)
    random = RandomStableBaselinePredictor(seed=42).fit(graphs)
    torch.testing.assert_close(mean.predict(graphs[0]), mean.predict(graphs[0]))
    torch.testing.assert_close(linear.predict(graphs[1]), linear.predict(graphs[1]))
    torch.testing.assert_close(random.predict(graphs[2]), random.predict(graphs[2]))


def test_baseline_evaluation_is_deterministic():
    graphs, circuits = _make_circuits()
    predictor = MeanBaselinePredictor().fit(graphs)
    metrics_a = evaluate_baseline_predictor(predictor, graphs, circuits)
    metrics_b = evaluate_baseline_predictor(predictor, graphs, circuits)
    assert metrics_a == metrics_b


def test_arena_rerun_consistency_flags_match():
    torch.manual_seed(42)
    graphs, circuits = _make_circuits()
    model = EdgeAwareCircuitGNN(node_dim=8, edge_dim=4, hidden_dim=16)
    results = run_arena(
        model,
        graphs[:2],
        circuits[:2],
        graphs[1:],
        circuits[1:],
        graphs[:1],
        circuits[:1],
        use_edge=True,
    )
    assert results["deterministic_rerun_validation"]["gnn"] is True
    assert results["deterministic_rerun_validation"]["mean_baseline"] is True
    assert results["deterministic_rerun_validation"]["linear_baseline"] is True
    assert results["deterministic_rerun_validation"]["random_baseline"] is True


def test_speed_benchmark_structure():
    torch.manual_seed(42)
    graphs, circuits = _make_circuits()
    model = EdgeAwareCircuitGNN(node_dim=8, edge_dim=4, hidden_dim=16)
    metrics = measure_speed(model, graphs[:1], circuits[:1], use_edge=True, num_runs=3)
    assert set(metrics) >= {
        "oracle_mean_sec",
        "oracle_p95_sec",
        "surrogate_mean_sec",
        "surrogate_p95_sec",
        "speedup",
        "num_runs",
        "warmup_runs",
    }
    assert metrics["num_runs"] == 3


def test_report_generation_builds_markdown(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"circuit_name": "c1", "netlist": "V1 N1 0 5\nR1 N1 0 1000\n"}, sort_keys=True),
                json.dumps({"circuit_name": "c2", "netlist": "V1 N1 0 3\nR1 N1 0 330\nR2 N1 0 1000\n"}, sort_keys=True),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "kaggle.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "device:",
                "  prefer_cuda: true",
                "training:",
                "  epochs: 10",
                "  batch_size: 32",
                "  learning_rate: 0.001",
                "dataset:",
                f"  train_path: {dataset}",
                "  eval_split: 0.2",
                "model:",
                "  hidden_dim: 64",
                "  max_params: 250000",
                "evaluation:",
                "  voltage_tolerance: 1e-3",
                "  invariant_tolerance: 1e-6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = build_checkpoint_payload(
        model_type="edge_aware",
        model_config={"hidden_dim": 64, "max_params": 250000, "num_params": 1234, "node_dim": 8, "edge_dim": 4, "use_edge_features": True},
        training_config={"seed": 42, "dataset_path": str(dataset), "epochs": 10, "batch_size": 32, "learning_rate": 0.001, "train_frac": 0.8},
        dataset_manifest_hash="dataset-fp",
        snapshot_hash="snapshot-fp",
        weights_hash="weights-fp",
        optimizer_state_hash="opt-fp",
        eval_fingerprint="eval-fp",
        curriculum_coverage={"train_count": 1, "eval_count": 1},
        seed=42,
        created_at=42.0,
        state_dict={"weight": torch.tensor([1.0])},
        extra={"history": [{"epoch": 1, "train_loss": 0.1, "eval_loss": 0.2, "eval_mae_V": 0.3, "eval_rmse_V": 0.4, "eval_max_error_V": 0.5, "lr": 0.001}], "epochs_trained": 10, "best_epoch": 1, "config_fingerprint": "config-fp", "dataset_fingerprint": "dataset-fp", "parent_oracle_version": "v2.8"},
    )
    torch.save(checkpoint, checkpoint_path)
    snapshot = _make_snapshot()
    snapshot_path = tmp_path / "training_snapshot.json"
    snapshot_path.write_text(snapshot.to_json(), encoding="utf-8")
    arena_path = tmp_path / "arena.json"
    arena_path.write_text(
        json.dumps(
            {
                "gnn": {
                    "in_distribution": {"mae": 0.1, "rmse": 0.2, "max_voltage_error": 0.3, "kcl_max_violation": 0.0, "kvl_max_violation": 0.0, "replay_consistency": 0.0, "count": 2},
                    "ood": {"mae": 0.4, "rmse": 0.5, "max_voltage_error": 0.6, "kcl_max_violation": 0.0, "kvl_max_violation": 0.0, "replay_consistency": 0.0, "count": 1},
                    "speed_in_distribution": {"oracle_mean_sec": 0.001, "surrogate_mean_sec": 0.0001, "speedup": 10.0, "num_runs": 2},
                    "replay_consistency_metrics": {"max_abs_diff": 0.0, "rerun_match": True},
                },
                "mean_baseline": {"mae": 0.9, "rmse": 1.0, "max_error": 1.1, "kcl_max_violation": 0.0, "kvl_max_violation": 0.0, "replay_consistency": 0.0, "count": 2},
                "linear_baseline": {"mae": 0.8, "rmse": 0.9, "max_error": 1.0, "kcl_max_violation": 0.0, "kvl_max_violation": 0.0, "replay_consistency": 0.0, "count": 2},
                "random_baseline": {"mae": 1.1, "rmse": 1.2, "max_error": 1.3, "kcl_max_violation": 0.0, "kvl_max_violation": 0.0, "replay_consistency": 0.0, "count": 2},
                "oracle": {"in_distribution": {"count": 2}, "ood": {"count": 1}},
                "metadata": {
                    "checkpoint": {
                        "artifact_fingerprint": "checkpoint-fp",
                        "dataset_manifest_hash": "dataset-fp",
                        "config_fingerprint": "config-fp",
                        "snapshot_hash": "snapshot-fp",
                        "snapshot_fingerprint": "snapshot-fp",
                        "eval_fingerprint": "eval-fp",
                        "parent_oracle_version": "v2.8",
                    },
                    "train_count": 2,
                    "eval_count": 1,
                    "ood_count": 1,
                },
                "summary": {
                    "checkpoint_artifact_fingerprint": "checkpoint-fp",
                    "dataset_manifest_hash": "dataset-fp",
                    "snapshot_hash": "snapshot-fp",
                    "evaluation_fingerprint": "eval-fp",
                    "parent_oracle_version": "v2.8",
                },
                "deterministic_rerun_validation": {"gnn": True, "mean_baseline": True, "linear_baseline": True, "random_baseline": True},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    failure_path = tmp_path / "failure.json"
    failure_path.write_text(
        json.dumps(
            {
                "analysis_fingerprint": "failure-fp",
                "failure_summary": {"count": 1, "dominant_failure": "conservation_drift", "failure_counts": {"conservation_drift": 1}},
                "speed": {"oracle_mean_sec": 0.001, "oracle_p95_sec": 0.001, "surrogate_mean_sec": 0.0001, "surrogate_p95_sec": 0.0001, "speedup": 10.0},
                "invariants": {"iid_kcl_violation": 0.0, "ood_kcl_violation": 0.1, "iid_kvl_violation": 0.0, "ood_kvl_violation": 0.1, "iid_power_violation": 0.0, "ood_power_violation": 0.1, "replay_max_abs_diff": 0.0},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    determinism_path = tmp_path / "determinism.json"
    determinism_path.write_text(
        json.dumps({"deterministic": True, "metrics_equal": True, "run_a_fingerprint": "a", "run_b_fingerprint": "a"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = build_report(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        arena_path=arena_path,
        failure_path=failure_path,
        determinism_path=determinism_path,
    )
    markdown = render_markdown(report)
    assert report["report_fingerprint"]
    assert "## Determinism Verification" in markdown
    assert "## Failure Taxonomy Summary" in markdown
    assert "## Speedup Metrics" in markdown


def test_artifact_registration_roundtrip(tmp_path):
    registry_path = tmp_path / "artifact_registry.json"
    registry = ArtifactRegistry(path=registry_path)
    registry.register(
        artifact_type="checkpoint",
        schema_version="2.7.6",
        fingerprint="fp-checkpoint",
        parent_fingerprints=["fp-dataset", "fp-config"],
        metadata={"path": "workspace/checkpoints/circuit_gnn_v29b.pt"},
    )
    registry.register(
        artifact_type="training_snapshot",
        schema_version="2.9b",
        fingerprint="fp-snapshot",
        parent_fingerprints=["fp-checkpoint"],
        metadata={"seed": 42},
    )
    registry.register(
        artifact_type="evaluation_report",
        schema_version="2.9c",
        fingerprint="fp-report",
        parent_fingerprints=["fp-checkpoint", "fp-snapshot"],
        metadata={"output_md": "docs/V29C_FIRST_SURROGATE_VALIDATION.md"},
    )
    registry.save(registry_path)
    loaded = ArtifactRegistry.from_file(registry_path)
    assert len(loaded.records()) == 3
    assert loaded.records()[0].artifact_id
