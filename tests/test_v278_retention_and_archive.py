import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.governance.artifact_policy import load_artifact_policy
from backend.governance.archive_manifest import ArchiveManifestError, build_archive_manifest, validate_archive_manifest
from backend.governance.archive_tooling import create_artifact_bundle
from backend.governance.retention_sweeper import build_retention_plan, execute_retention_plan, scan_retention_candidates
from backend.reporting.retention_report import build_retention_report


def _write_policy(tmp_path: Path) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(
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
                "    retention:",
                "      keep_latest: 1",
                "      keep_by_fingerprint: false",
                "      keep_pinned: true",
                "  dataset:",
                "    required_fields:",
                "      - dataset_version",
                "      - schema_version",
                "      - snapshot_hash",
                "      - module_hash",
                "      - generation_seed",
                "      - record_count",
                "      - fingerprint",
                "    retention:",
                "      keep_latest: 1",
                "      keep_by_fingerprint: false",
                "      keep_pinned: true",
                "  manifest:",
                "    required_fields:",
                "      - schema_version",
                "      - fingerprint",
                "    retention:",
                "      keep_latest: 1",
                "      keep_by_fingerprint: false",
                "      keep_pinned: true",
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
                "    retention:",
                "      keep_latest: 1",
                "      keep_by_fingerprint: false",
                "  benchmark_snapshot:",
                "    required_fields:",
                "      - schema_version",
                "      - fingerprint",
                "    retention:",
                "      keep_latest: 1",
                "      keep_by_fingerprint: false",
                "  artifact_registry:",
                "    required_fields:",
                "      - schema_version",
                "      - registry_fingerprint",
                "    retention:",
                "      keep_latest: 1",
                "      keep_by_fingerprint: false",
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
                "    write: true",
                "  v278:",
                "    read: true",
                "    write: true",
                "enforcement:",
                "  strict_mode: true",
                "  fail_on_unknown_artifact: true",
                "  fail_on_missing_fingerprint: true",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    root.mkdir()
    _write_json(
        root / "checkpoint_old.pt",
        {
            "schema_version": "2.7.8",
            "model_type": "transformer",
            "model_config": {"d_model": 16},
            "training_config": {"seed": 1},
            "dataset_manifest_hash": "dm-1",
            "snapshot_hash": "snap-1",
            "weights_hash": "weights-1",
            "artifact_fingerprint": "ckpt-old",
            "created_at": 10.0,
            "pinned": False,
        },
    )
    _write_json(
        root / "checkpoint_new.pt",
        {
            "schema_version": "2.7.8",
            "model_type": "transformer",
            "model_config": {"d_model": 16},
            "training_config": {"seed": 1},
            "dataset_manifest_hash": "dm-2",
            "snapshot_hash": "snap-2",
            "weights_hash": "weights-2",
            "artifact_fingerprint": "ckpt-new",
            "created_at": 20.0,
            "pinned": True,
        },
    )
    root.joinpath("dataset.manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "2.7.0",
                "schema_version": "2.7.0",
                "snapshot_hash": "ds-snap",
                "module_hash": "mod-1",
                "generation_seed": 1,
                "record_count": 2,
                "fingerprint": "dataset-manifest",
                "created_at": 15.0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    root.joinpath("eval_report.json").write_text(
        json.dumps(
            {
                "artifact_type": "evaluation_report",
                "schema_version": "2.7.7",
                "compatibility_status": "compatible",
                "policy": {"schema_version": "1.0"},
                "summary": {"model_type": "transformer"},
                "per_layer": {},
                "per_module": {},
                "failure_summary": {"failure_counts": {}, "dominant_failure": "none"},
                "ood": {},
                "dataset_manifest": {"fingerprint": "dataset-manifest"},
                "checkpoint": {"schema_version": "2.7.8"},
                "evaluation": {"fingerprint": "eval-1"},
                "artifacts": {},
                "report_fingerprint": "report-1",
                "created_at": 25.0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    root.joinpath("benchmark.snapshot.json").write_text(
        json.dumps({"schema_version": "2.7.8", "fingerprint": "snap-1", "created_at": 30.0}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    root.joinpath("artifact_registry.json").write_text(
        json.dumps({"schema_version": "2.7.8", "registry_fingerprint": "registry-1", "records": []}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    root.joinpath("dataset_legacy.jsonl").write_text("{\"row\":1}\n{\"row\":2}\n", encoding="utf-8")
    return root


def test_retention_scan_and_plan_determinism(tmp_path):
    policy = load_artifact_policy(_write_policy(tmp_path))
    root = _make_root(tmp_path)
    candidates_a = scan_retention_candidates(root, policy)
    candidates_b = scan_retention_candidates(root, policy)
    assert [c.to_dict() for c in candidates_a] == [c.to_dict() for c in candidates_b]

    plan_a = build_retention_plan(candidates_a, policy)
    plan_b = build_retention_plan(candidates_b, policy)
    assert [c.to_dict() for c in plan_a] == [c.to_dict() for c in plan_b]
    assert any(candidate.pinned and candidate.retention_reason and candidate.retention_reason.startswith("retained") for candidate in plan_a)
    assert any(candidate.retention_reason and candidate.retention_reason.startswith("eligible_for_deletion") for candidate in plan_a)


def test_dry_run_does_not_delete(tmp_path):
    policy = load_artifact_policy(_write_policy(tmp_path))
    root = _make_root(tmp_path)
    plan = build_retention_plan(scan_retention_candidates(root, policy), policy)
    result = execute_retention_plan(plan, dry_run=True)
    assert result.deleted == 0
    assert (root / "checkpoint_old.pt").exists()


def test_archive_bundle_generation_and_manifest_validation(tmp_path):
    policy = load_artifact_policy(_write_policy(tmp_path))
    root = _make_root(tmp_path)
    bundle_path, manifest = create_artifact_bundle(
        [root / "checkpoint_new.pt", root / "dataset.manifest.json", root / "eval_report.json"],
        tmp_path / "bundle.tar.gz",
        policy=policy,
        source_snapshot_hash="snapshot-1",
        created_at=0.0,
    )
    assert bundle_path.exists()
    loaded = build_archive_manifest(bundle_path)
    validate_archive_manifest(loaded)
    assert loaded["bundle_fingerprint"] == manifest["bundle_fingerprint"]
    assert loaded["artifact_count"] == 3


def test_archive_manifest_validation_rejects_bad_payload():
    with pytest.raises(ArchiveManifestError):
        validate_archive_manifest({"bundle_version": "1.0"})


def test_retention_report_and_reclaimable_storage(tmp_path):
    policy = load_artifact_policy(_write_policy(tmp_path))
    root = _make_root(tmp_path)
    report = build_retention_report(root, policy)
    payload = report.to_dict()
    assert payload["summary"]["total_artifacts"] >= 5
    assert payload["summary"]["reclaimable_storage_bytes"] >= 0
    assert "# CPT Retention Report" in report.to_markdown()


def test_policy_enforcement_during_cleanup_requires_confirmation(tmp_path):
    policy_path = _write_policy(tmp_path)
    root = _make_root(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/retention_sweeper.py",
            "--root",
            str(root),
            "--policy",
            str(policy_path),
            "--execute",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "without --yes" in proc.stderr or "without --yes" in proc.stdout


def test_backward_compatibility_legacy_artifacts_are_scannable(tmp_path):
    policy = load_artifact_policy(_write_policy(tmp_path))
    root = tmp_path / "legacy_root"
    root.mkdir()
    root.joinpath("legacy_checkpoint.pt").write_text(
        json.dumps(
            {
                "state_dict": {"w": [1, 2]},
                "tokenizer": {"vocab": ["a", "b"]},
                "config": {"seed": 1},
                "artifact_fingerprint": "legacy",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    candidates = scan_retention_candidates(root, policy)
    assert candidates
    assert candidates[0].artifact_type == "checkpoint"
