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
from backend.circuits.graph_dataset import circuit_to_graph
from backend.circuits.losses import invariant_aware_loss, kcl_penalty, kvl_penalty, voltage_loss
from backend.circuits.parser import parse_netlist
from backend.governance.artifact_registry import ArtifactRegistry
from backend.neural.checkpoints.schema import build_checkpoint_payload
from backend.neural.training_snapshot import TrainingSnapshot
from backend.neural.models.circuit_gnn import EdgeAwareCircuitGNN
from scripts.generate_v29b_report import build_v29b_report, render_markdown
from scripts.kaggle_prepare_v29b import load_kaggle_profile, prepare_kaggle_export, resolve_dataset_path
from scripts.run_circuit_arena import run_arena


def _make_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "circuits.jsonl"
    rows = [
        {"circuit_name": "c1", "netlist": "V1 N1 0 5\nR1 N1 0 1000\n"},
        {"circuit_name": "c2", "netlist": "V1 N1 0 3\nR1 N1 0 330\nR2 N1 0 1000\n"},
        {"circuit_name": "c3", "netlist": "I1 N1 0 0.002\nR1 N1 0 1000\n"},
    ]
    dataset.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return dataset


def _make_graphs():
    circuits = [
        parse_netlist("V1 N1 0 5\nR1 N1 0 1000\n", name="c1"),
        parse_netlist("V1 N1 0 3\nR1 N1 0 330\nR2 N1 0 1000\n", name="c2"),
        parse_netlist("I1 N1 0 0.002\nR1 N1 0 1000\n", name="c3"),
    ]
    graphs = [circuit_to_graph(circuit, solve_dc_circuit(circuit)) for circuit in circuits]
    return graphs, circuits


def _make_snapshot(seed: int = 42) -> TrainingSnapshot:
    return TrainingSnapshot.create(
        seed=seed,
        dataset_fingerprint="dataset-fp",
        config={
            "seed": seed,
            "device": {"prefer_cuda": False},
            "training": {"epochs": 2, "batch_size": 2, "learning_rate": 1e-3},
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


def test_kaggle_profile_loads_required_fields():
    profile = load_kaggle_profile("configs/training/kaggle_v29b.yaml")
    assert profile["seed"] == 42
    assert profile["device"]["prefer_cuda"] is True
    assert profile["training"]["epochs"] == 10
    assert profile["model"]["hidden_dim"] == 64


def test_kaggle_dataset_path_resolution():
    resolved = resolve_dataset_path("workspace/datasets/circuits/train_10k.jsonl")
    assert resolved.name == "circuits.jsonl"
    assert "train_10k" in str(resolved)


def test_snapshot_fingerprint_is_deterministic():
    s1 = _make_snapshot()
    s2 = _make_snapshot()
    assert s1.fingerprint() == s2.fingerprint()


def test_snapshot_to_dict_is_stable():
    snapshot = _make_snapshot()
    payload = snapshot.to_dict()
    assert payload["seed"] == 42
    assert payload["cuda_enabled"] is False
    assert payload["artifact_fingerprint"] == snapshot.fingerprint()


def test_snapshot_export_roundtrip(tmp_path):
    snapshot = _make_snapshot()
    path = snapshot.export(tmp_path)
    loaded = TrainingSnapshot.create(
        seed=42,
        dataset_fingerprint="dataset-fp",
        config={
            "seed": 42,
            "device": {"prefer_cuda": False},
            "training": {"epochs": 2, "batch_size": 2, "learning_rate": 1e-3},
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


def test_voltage_loss_is_finite():
    pred = torch.tensor([1.0, 2.0], dtype=torch.float32)
    target = torch.tensor([1.5, 1.5], dtype=torch.float32)
    assert torch.isfinite(voltage_loss(pred, target))


def test_kcl_penalty_is_finite():
    circuit = parse_netlist("V1 N1 0 5\nR1 N1 0 1000\n", name="kcl")
    pred = torch.tensor([4.9], dtype=torch.float32)
    penalty = kcl_penalty(circuit, ("N1",), pred)
    assert torch.isfinite(penalty)


def test_kvl_penalty_is_finite():
    circuit = parse_netlist("V1 N1 0 5\nR1 N1 0 1000\n", name="kvl")
    pred = torch.tensor([4.9], dtype=torch.float32)
    penalty = kvl_penalty(circuit, ("N1",), pred)
    assert torch.isfinite(penalty)


def test_invariant_loss_matches_formula():
    circuit = parse_netlist("V1 N1 0 5\nR1 N1 0 1000\n", name="loss")
    pred = torch.tensor([4.9], dtype=torch.float32)
    target = torch.tensor([5.0], dtype=torch.float32)
    expected = voltage_loss(pred, target) + 0.1 * kcl_penalty(circuit, ("N1",), pred) + 0.1 * kvl_penalty(circuit, ("N1",), pred)
    actual = invariant_aware_loss(pred, target, circuit, ("N1",))
    torch.testing.assert_close(actual, expected)


def test_mean_baseline_is_deterministic():
    graphs, _ = _make_graphs()
    predictor = MeanBaselinePredictor().fit(graphs)
    pred1 = predictor.predict(graphs[0])
    pred2 = predictor.predict(graphs[0])
    torch.testing.assert_close(pred1, pred2)


def test_linear_baseline_is_deterministic():
    graphs, _ = _make_graphs()
    predictor = LinearRegressionBaselinePredictor().fit(graphs)
    pred1 = predictor.predict(graphs[1])
    pred2 = predictor.predict(graphs[1])
    torch.testing.assert_close(pred1, pred2)


def test_random_stable_baseline_is_deterministic():
    graphs, _ = _make_graphs()
    predictor = RandomStableBaselinePredictor(seed=42).fit(graphs)
    pred1 = predictor.predict(graphs[2])
    pred2 = predictor.predict(graphs[2])
    torch.testing.assert_close(pred1, pred2)


def test_baseline_evaluation_is_deterministic():
    graphs, circuits = _make_graphs()
    predictor = MeanBaselinePredictor().fit(graphs)
    metrics_1 = evaluate_baseline_predictor(predictor, graphs, circuits)
    metrics_2 = evaluate_baseline_predictor(predictor, graphs, circuits)
    assert metrics_1 == metrics_2


def test_arena_rerun_validation_reports_consistency():
    torch.manual_seed(42)
    graphs, circuits = _make_graphs()
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
    registry.save(registry_path)
    loaded = ArtifactRegistry.from_file(registry_path)
    assert len(loaded.records()) == 2
    assert loaded.records()[0].artifact_id


def test_kaggle_prepare_manifest_generation(tmp_path):
    dataset = _make_dataset(tmp_path)
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
    output_dir = tmp_path / "kaggle_export"
    manifest = prepare_kaggle_export(config_path, output_dir)
    assert (output_dir / "run_kaggle_v29b.sh").exists()
    assert (output_dir / "kaggle_v29b_manifest.json").exists()
    assert manifest["profile"] == "kaggle_v29b"


def test_report_generation_builds_markdown(tmp_path):
    dataset = _make_dataset(tmp_path)
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
                "  eval_split: 0.5",
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
        training_config={"seed": 42, "dataset_path": str(dataset), "epochs": 10, "batch_size": 32, "learning_rate": 0.001, "train_frac": 0.5},
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
    arena_payload = {
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
    }
    arena_path.write_text(json.dumps(arena_payload, indent=2, sort_keys=True), encoding="utf-8")

    kaggle_manifest_path = tmp_path / "kaggle_v29b_manifest.json"
    kaggle_manifest_path.write_text(
        json.dumps(
            {
                "profile": "kaggle_v29b",
                "profile_fingerprint": "profile-fp",
                "dataset_fingerprint": "dataset-fp",
                "git_commit": "git-fp",
                "run_command": "bash run_kaggle_v29b.sh",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = build_v29b_report(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        arena_path=arena_path,
        snapshot_path=snapshot_path,
        kaggle_manifest_path=kaggle_manifest_path,
    )
    markdown = render_markdown(report)
    assert report["report_fingerprint"]
    assert "# CPT v2.9B Reproducible Surrogate" in markdown
    assert "## Baselines" in markdown
    assert "## Kaggle Metadata" in markdown
