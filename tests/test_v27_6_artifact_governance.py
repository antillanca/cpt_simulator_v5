import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from backend.datasets.manifest import DatasetManifest
from backend.governance.artifact_registry import ArtifactRegistry
from backend.neural.checkpoints import CHECKPOINT_SCHEMA_VERSION, validate_checkpoint_payload
from backend.neural.checkpoints.migrate import migrate_checkpoint
from backend.neural.tiny_experiments import TrainConfig, load_checkpoint, train_model
from backend.reporting.eval_diff import diff_eval_reports
from backend.reporting.report_builder import build_evaluation_report
from scripts.model_eval_runner import evaluate_dataset


def _make_dataset(tmp_path: Path, n: int = 4) -> Path:
    rows = []
    for i in range(n):
        rows.append(
            {
                "sample_id": f"s{i}",
                "question": f"Q{i}",
                "structured_state": {"initial_state": {"x": i}, "parameters": {}, "module": "m", "module_version": "v"},
                "reasoning_trace": [],
                "equations_used": [],
                "invariants_checked": ["logic_basic"],
                "final_answer": {"x": i},
                "verification_status": {"passed": True, "violations": []},
                "module_source": "m::layer_00",
                "curriculum_layer": 0,
                "seed": 7,
                "timestamp": float(i),
                "dataset_version": "2.7.0",
                "snapshot_hash": "snap",
                "module_hash": "mod",
            }
        )
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    manifest = DatasetManifest(
        generation_seed=7,
        record_count=n,
        snapshot_hash="snap",
        module_hash="mod",
        curriculum_coverage=[0],
        modules_used=["m"],
        shard_list=[],
    )
    manifest.save(dataset.with_suffix(".manifest.json"))
    return dataset


def _train_checkpoint(tmp_path: Path, dataset: Path) -> Path:
    cfg = TrainConfig(
        model_type="transformer",
        seed=5,
        data_path=dataset,
        output_dir=tmp_path / "train_out",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        max_steps=1,
        device="cpu",
        eval_every=1,
        save_every=1,
    )
    result = train_model(cfg)
    return Path(result["checkpoint_path"])


def test_checkpoint_schema_validation_and_load_roundtrip(tmp_path):
    dataset = _make_dataset(tmp_path)
    checkpoint = _train_checkpoint(tmp_path, dataset)

    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert validate_checkpoint_payload(payload, allow_legacy=False) == []

    model, tokenizer, loaded_payload = load_checkpoint(checkpoint)
    assert loaded_payload["artifact_fingerprint"] == payload["artifact_fingerprint"]
    assert tokenizer.pad_id >= 0
    assert model.model_type == "transformer"


def test_checkpoint_migration_from_legacy_payload(tmp_path):
    dataset = _make_dataset(tmp_path)
    checkpoint = _train_checkpoint(tmp_path, dataset)
    current = torch.load(checkpoint, map_location="cpu")
    legacy = dict(current)
    legacy.pop("schema_version", None)
    legacy.pop("model_config", None)
    legacy.pop("training_config", None)
    legacy.pop("dataset_manifest_hash", None)
    legacy.pop("snapshot_hash", None)
    legacy.pop("weights_hash", None)
    legacy.pop("optimizer_state_hash", None)
    legacy.pop("eval_fingerprint", None)
    legacy.pop("curriculum_coverage", None)
    legacy.pop("created_at", None)
    legacy.pop("artifact_fingerprint", None)
    legacy["config"] = dict(current["config"])
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy, legacy_path)

    migration = migrate_checkpoint(legacy_path, CHECKPOINT_SCHEMA_VERSION, dry_run=True)
    assert migration.source_version == "2.7.5"
    assert migration.target_version == CHECKPOINT_SCHEMA_VERSION
    assert "schema_version" in migration.fields_added
    assert migration.migration_fingerprint

    model, tokenizer, migrated = load_checkpoint(legacy_path)
    assert migrated["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert model.model_type == "transformer"
    assert tokenizer.pad_id >= 0


def test_artifact_registry_roundtrip(tmp_path):
    registry = ArtifactRegistry(path=tmp_path / "registry.json")
    a1 = registry.register(
        artifact_type="checkpoint",
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        fingerprint="abc123",
        metadata={"model": "tiny_transformer"},
        created_at=1.0,
    )
    a2 = registry.register(
        artifact_type="evaluation_report",
        schema_version="2.7.6",
        fingerprint="def456",
        parent_fingerprints=["abc123"],
        metadata={"dataset": "demo"},
        created_at=2.0,
    )
    path = registry.save()
    loaded = ArtifactRegistry.from_file(path)
    assert [record.artifact_id for record in loaded.records()] == sorted([a1.artifact_id, a2.artifact_id])
    assert loaded.to_dict()["registry_fingerprint"] == registry.to_dict()["registry_fingerprint"]


def test_report_generation_json_and_markdown(tmp_path):
    dataset = _make_dataset(tmp_path)
    checkpoint = _train_checkpoint(tmp_path, dataset)
    eval_run = evaluate_dataset(dataset, checkpoint=checkpoint, model_type="transformer", seed=5, output_path=tmp_path / "eval.json")

    report = build_evaluation_report(eval_run, checkpoint_path=checkpoint, seed=123)
    payload1 = report.to_dict()
    payload2 = report.to_dict()
    assert payload1["report_fingerprint"] == payload2["report_fingerprint"]
    assert "summary" in payload1
    assert "failure_summary" in payload1

    json_path = report.save(tmp_path / "report.json")
    md_path = report.save(tmp_path / "report.md", markdown=True)
    assert json_path.exists()
    assert md_path.exists()
    assert "# CPT Evaluation Report" in md_path.read_text(encoding="utf-8")


def test_evaluation_diff_stability(tmp_path):
    dataset = _make_dataset(tmp_path)
    checkpoint = _train_checkpoint(tmp_path, dataset)
    eval_run = evaluate_dataset(dataset, checkpoint=checkpoint, model_type="transformer", seed=5, output_path=tmp_path / "eval.json")
    report_a = build_evaluation_report(eval_run, checkpoint_path=checkpoint, seed=1)
    candidate_run = json.loads(json.dumps(eval_run))
    candidate_run["evaluation"]["metrics"]["exact_match_rate"] = float(candidate_run["evaluation"]["metrics"].get("exact_match_rate", 0.0)) + 0.1
    report_b = build_evaluation_report(candidate_run, checkpoint_path=checkpoint, seed=1)

    baseline_path = report_a.save(tmp_path / "baseline.json")
    candidate_path = report_b.save(tmp_path / "candidate.json")
    diff1 = diff_eval_reports(baseline_path, candidate_path)
    diff2 = diff_eval_reports(baseline_path, candidate_path)
    assert diff1.to_dict() == diff2.to_dict()
    assert diff1.metric_deltas["exact_match_rate"] == 0.1
    assert not diff1.same_fingerprint


def test_generate_eval_report_cli_reproducible(tmp_path):
    dataset = _make_dataset(tmp_path)
    checkpoint = _train_checkpoint(tmp_path, dataset)
    eval_run = evaluate_dataset(dataset, checkpoint=checkpoint, model_type="transformer", seed=5, output_path=tmp_path / "eval.json")
    eval_input = tmp_path / "eval_run.json"
    eval_input.write_text(json.dumps(eval_run, indent=2, sort_keys=True), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    out1 = tmp_path / "report1.json"
    out2 = tmp_path / "report2.json"
    subprocess.run([sys.executable, "scripts/generate_eval_report.py", "--input", str(eval_input), "--output", str(out1), "--json"], cwd=Path(__file__).resolve().parents[1], env=env, check=True, capture_output=True, text=True)
    subprocess.run([sys.executable, "scripts/generate_eval_report.py", "--input", str(eval_input), "--output", str(out2), "--json"], cwd=Path(__file__).resolve().parents[1], env=env, check=True, capture_output=True, text=True)
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
