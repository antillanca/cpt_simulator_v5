import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.governance.artifact_policy import (
    ArtifactCompatibilityError,
    ArtifactPolicyError,
    MissingRequiredArtifactFieldError,
    artifact_policy_fingerprint,
    load_artifact_policy,
)
from backend.neural.checkpoints import CHECKPOINT_SCHEMA_VERSION
from backend.neural.checkpoints.migrate import migrate_checkpoint
from backend.neural.checkpoints.validator import enforce_checkpoint_policy, validate_checkpoint_payload_with_policy
from backend.neural.tiny_experiments import TrainConfig, train_model
from backend.reporting.report_builder import build_evaluation_report, validate_evaluation_report
from scripts.model_eval_runner import evaluate_dataset


def _make_dataset(tmp_path: Path) -> Path:
    rows = []
    for i in range(6):
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


def _legacy_checkpoint_from_current(path: Path, output: Path) -> Path:
    import torch

    payload = torch.load(path, map_location="cpu")
    legacy = dict(payload)
    for key in [
        "schema_version",
        "model_config",
        "training_config",
        "dataset_manifest_hash",
        "snapshot_hash",
        "weights_hash",
        "optimizer_state_hash",
        "eval_fingerprint",
        "curriculum_coverage",
        "created_at",
        "artifact_fingerprint",
    ]:
        legacy.pop(key, None)
    legacy["config"] = dict(payload["config"])
    torch.save(legacy, output)
    return output


def test_policy_loading_and_fingerprint_is_stable():
    policy = load_artifact_policy(Path("configs/artifact_policy.yaml"))
    assert policy.schema_version == "1.0"
    assert artifact_policy_fingerprint(policy) == artifact_policy_fingerprint(policy)


def test_policy_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad_policy.yaml"
    path.write_text(
        "\n".join(
            [
                'schema_version: "1.0"',
                "defaults: {}",
                "artifacts: {}",
                "compatibility: {}",
                "enforcement: {}",
                "unknown_key: true",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactPolicyError):
        load_artifact_policy(path)


def test_checkpoint_validation_uses_policy_and_accepts_legacy(tmp_path):
    dataset = _make_dataset(tmp_path)
    checkpoint = _train_checkpoint(tmp_path, dataset)
    import torch

    payload = torch.load(checkpoint, map_location="cpu")
    policy = load_artifact_policy(Path("configs/artifact_policy.yaml"))
    assert validate_checkpoint_payload_with_policy(payload, allow_legacy=False, policy=policy, strict_policy=True) == []

    legacy_path = _legacy_checkpoint_from_current(checkpoint, tmp_path / "legacy.pt")
    legacy_payload = torch.load(legacy_path, map_location="cpu")
    assert validate_checkpoint_payload_with_policy(legacy_payload, allow_legacy=True, policy=policy, strict_policy=True) == []


def test_missing_required_field_rejected(tmp_path):
    policy = load_artifact_policy(Path("configs/artifact_policy.yaml"))
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": "transformer",
        "model_config": {},
        "training_config": {},
        "dataset_manifest_hash": "snap",
        "snapshot_hash": "snap",
        "weights_hash": "weights",
        "artifact_fingerprint": "",
    }
    with pytest.raises(MissingRequiredArtifactFieldError):
        # The checkpoint payload is missing the non-empty fingerprint required by policy enforcement.
        enforce_checkpoint_policy(payload, policy, strict_policy=True)
    with pytest.raises(ArtifactPolicyError):
        validate_evaluation_report(
            {
                "artifact_type": "evaluation_report",
                "schema_version": "2.7.7",
                "compatibility_status": "compatible",
                "summary": {},
                "per_layer": {},
                "per_module": {},
                "failure_summary": {},
                "ood": {},
                "dataset_manifest": {},
                "checkpoint": {},
                "evaluation": {},
                "artifacts": {},
            },
            policy=policy,
            strict_policy=True,
        )


def test_report_policy_metadata_and_cli_wiring(tmp_path):
    dataset = _make_dataset(tmp_path)
    checkpoint = _train_checkpoint(tmp_path, dataset)
    eval_run = evaluate_dataset(dataset, checkpoint=checkpoint, model_type="transformer", seed=5, output_path=tmp_path / "eval.json")
    policy = load_artifact_policy(Path("configs/artifact_policy.yaml"))

    report = build_evaluation_report(eval_run, checkpoint_path=checkpoint, seed=7, policy=policy, strict_policy=True)
    payload = report.to_dict()
    assert payload["policy"]["schema_version"] == "1.0"
    assert payload["summary"]["policy_schema_version"] == "1.0"
    assert payload["summary"]["policy_enforcement_mode"] == "strict"

    eval_input = tmp_path / "eval_run.json"
    eval_input.write_text(json.dumps(eval_run, indent=2, sort_keys=True), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    report_path = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_eval_report.py",
            "--input",
            str(eval_input),
            "--output",
            str(report_path),
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    generated = json.loads(report_path.read_text(encoding="utf-8"))
    assert generated["policy"]["schema_version"] == "1.0"
    assert generated["summary"]["policy_schema_version"] == "1.0"


def test_migration_policy_enforcement(tmp_path):
    dataset = _make_dataset(tmp_path)
    checkpoint = _train_checkpoint(tmp_path, dataset)
    legacy_path = _legacy_checkpoint_from_current(checkpoint, tmp_path / "legacy.pt")

    policy_path = tmp_path / "restricted_policy.yaml"
    policy_path.write_text(
        "\n".join(
            [
                'schema_version: "1.0"',
                "defaults:",
                "  allow_legacy_read: true",
                "  allow_legacy_write: false",
                "  require_fingerprint: true",
                "  require_manifest: true",
                "  require_snapshot_hash: true",
                "artifacts:",
                "  checkpoint:",
                "    required_fields:",
                "      - schema_version",
                "      - model_type",
                "      - model_config",
                "      - training_config",
                "      - dataset_manifest_hash",
                "      - snapshot_hash",
                "      - weights_hash",
                "      - artifact_fingerprint",
                "    migration:",
                "      allow_dry_run: false",
                "      require_explicit_target_version: true",
                "  dataset:",
                "    required_fields:",
                "      - dataset_version",
                "      - schema_version",
                "      - snapshot_hash",
                "      - module_hash",
                "      - generation_seed",
                "      - record_count",
                "      - fingerprint",
                "  evaluation_report:",
                "    required_fields:",
                "      - artifact_type",
                "      - schema_version",
                "      - compatibility_status",
                "      - policy",
                "      - summary",
                "      - per_layer",
                "      - per_module",
                "      - failure_summary",
                "      - ood",
                "      - dataset_manifest",
                "      - checkpoint",
                "      - evaluation",
                "      - artifacts",
                "compatibility:",
                "  v26:",
                "    read: true",
                "    write: false",
                "  v27:",
                "    read: true",
                "    write: true",
                "  v275:",
                "    read: true",
                "    write: true",
                "  v276:",
                "    read: true",
                "    write: false",
                "enforcement:",
                "  strict_mode: true",
                "  fail_on_unknown_artifact: true",
                "  fail_on_missing_fingerprint: true",
            ]
        ),
        encoding="utf-8",
    )

    policy = load_artifact_policy(policy_path)
    with pytest.raises(ArtifactCompatibilityError):
        migrate_checkpoint(legacy_path, CHECKPOINT_SCHEMA_VERSION, dry_run=True, policy=policy, strict_policy=True)


def test_backward_compatibility_v276_artifacts(tmp_path):
    dataset = _make_dataset(tmp_path)
    checkpoint = _train_checkpoint(tmp_path, dataset)
    import torch

    payload = torch.load(checkpoint, map_location="cpu")
    policy = load_artifact_policy(Path("configs/artifact_policy.yaml"))
    assert validate_checkpoint_payload_with_policy(payload, allow_legacy=False, policy=policy, strict_policy=True) == []
