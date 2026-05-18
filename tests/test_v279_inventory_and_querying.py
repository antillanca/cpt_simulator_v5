import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from backend.datasets.manifest import DatasetManifest
from backend.governance.artifact_inventory import build_inventory_index, load_inventory_index, save_inventory_index
from backend.governance.artifact_policy import artifact_policy_fingerprint, load_artifact_policy
from backend.governance.archive_tooling import create_artifact_bundle
from backend.governance.drift_detection import detect_inventory_drift
from backend.governance.lineage_graph import build_lineage_graph
from backend.governance.query_engine import query_inventory
from backend.reporting.workspace_summary import build_workspace_summary


def _write_workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()

    manifest = DatasetManifest(
        dataset_version="2.7.0",
        schema_version="2.7.0",
        snapshot_hash="snap-dataset",
        module_hash="module-1",
        generation_seed=11,
        record_count=2,
        shard_list=["shard_000000.jsonl"],
        modules_used=["module_a"],
        timestamp="2026-05-16T00:00:00",
        timestamp_unix=1.0,
        benchmark_fingerprint="bench-1",
    )
    manifest.save(root / "dataset.manifest.json")

    checkpoint_payload = {
        "schema_version": "2.7.8",
        "model_type": "transformer",
        "model_config": {"d_model": 16},
        "training_config": {"seed": 11},
        "dataset_manifest_hash": manifest.fingerprint,
        "snapshot_hash": "snap-checkpoint",
        "weights_hash": "weights-1",
        "optimizer_state_hash": None,
        "eval_fingerprint": "eval-1",
        "curriculum_coverage": {"0": 2},
        "seed": 11,
        "created_at": 2.0,
        "artifact_fingerprint": "ckpt-1",
        "pinned": True,
        "retention_status": "pinned",
    }
    torch.save(checkpoint_payload, root / "checkpoint.pt")

    report_payload = {
        "artifact_type": "evaluation_report",
        "schema_version": "2.7.7",
        "compatibility_status": "compatible",
        "policy": {"schema_version": "1.0"},
        "summary": {
            "checkpoint_artifact_fingerprint": "ckpt-1",
            "dataset_manifest_hash": manifest.fingerprint,
            "model_type": "transformer",
        },
        "per_layer": {},
        "per_module": {},
        "failure_summary": {"failure_counts": {}, "dominant_failure": "none"},
        "ood": {},
        "dataset_manifest": {"fingerprint": manifest.fingerprint},
        "checkpoint": {"schema_version": "2.7.8"},
        "evaluation": {"fingerprint": "eval-1"},
        "artifacts": {},
        "report_fingerprint": "report-1",
        "created_at": 3.0,
    }
    (root / "eval_report.json").write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

    registry_payload = {
        "schema_version": "2.7.8",
        "registry_fingerprint": "registry-1",
        "records": [
            {
                "artifact_id": "registry-entry",
                "artifact_type": "checkpoint",
                "schema_version": "2.7.8",
                "fingerprint": "ckpt-1",
                "parent_fingerprints": [manifest.fingerprint],
                "created_at": 2.0,
                "last_accessed_at": 2.5,
                "archived_at": None,
                "pinned": True,
                "retention_status": "pinned",
                "metadata": {},
                "compatibility_status": "compatible",
            }
        ],
    }
    (root / "artifact_registry.json").write_text(json.dumps(registry_payload, indent=2, sort_keys=True), encoding="utf-8")

    snapshot_payload = {
        "schema_version": "2.7.8",
        "fingerprint": "snap-archive",
        "created_at": 4.0,
    }
    (root / "benchmark.snapshot.json").write_text(json.dumps(snapshot_payload, indent=2, sort_keys=True), encoding="utf-8")

    (root / "shard_000000.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "s1", "value": 1}, sort_keys=True),
                json.dumps({"sample_id": "s2", "value": 2}, sort_keys=True),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle_path, _manifest = create_artifact_bundle(
        [root / "checkpoint.pt", root / "dataset.manifest.json", root / "eval_report.json"],
        root / "bundle.tar.gz",
        policy=load_artifact_policy(Path("configs/artifact_policy.yaml")),
        source_snapshot_hash="snapshot-1",
        created_at=5.0,
    )
    assert bundle_path.exists()
    return root


def test_inventory_index_determinism_and_incremental_refresh(tmp_path):
    workspace = _write_workspace(tmp_path)
    policy = load_artifact_policy(Path("configs/artifact_policy.yaml"))

    index_a = build_inventory_index(workspace, policy=policy)
    index_b = build_inventory_index(workspace, policy=policy, previous_index=index_a)
    assert index_a.to_dict() == index_b.to_dict()
    assert index_a.policy_fingerprint == artifact_policy_fingerprint(policy)
    assert index_a.entry_count >= 5

    save_path = tmp_path / "inventory_index.json"
    save_inventory_index(index_a, save_path)
    loaded = load_inventory_index(save_path)
    assert loaded.to_dict() == index_a.to_dict()

    (workspace / "extra_report.json").write_text(
        json.dumps(
            {
                "artifact_type": "evaluation_report",
                "schema_version": "2.7.7",
                "compatibility_status": "compatible",
                "policy": {"schema_version": "1.0"},
                "summary": {"checkpoint_artifact_fingerprint": "ckpt-1"},
                "per_layer": {},
                "per_module": {},
                "failure_summary": {"failure_counts": {}, "dominant_failure": "none"},
                "ood": {},
                "dataset_manifest": {"fingerprint": "extra"},
                "checkpoint": {"schema_version": "2.7.8"},
                "evaluation": {"fingerprint": "eval-2"},
                "artifacts": {},
                "report_fingerprint": "report-2",
                "created_at": 6.0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    index_c = build_inventory_index(workspace, policy=policy, previous_index=index_b)
    assert index_c.entry_count == index_a.entry_count + 1
    assert index_c.inventory_fingerprint != index_a.inventory_fingerprint


def test_query_determinism_and_lineage_graph_stability(tmp_path):
    workspace = _write_workspace(tmp_path)
    index = build_inventory_index(workspace, policy=load_artifact_policy(Path("configs/artifact_policy.yaml")))
    by_type = query_inventory(index, artifact_type="checkpoint")
    by_type_repeat = query_inventory(index, artifact_type="checkpoint")
    assert [entry.to_dict() for entry in by_type] == [entry.to_dict() for entry in by_type_repeat]
    assert by_type[0].artifact_type == "checkpoint"

    pinned = query_inventory(index, tag="pinned")
    assert pinned
    assert all("pinned" in entry.tags for entry in pinned)

    report = query_inventory(index, artifact_type="evaluation_report")[0]
    checkpoint = query_inventory(index, artifact_type="checkpoint")[0]
    assert checkpoint.artifact_id in report.lineage_parents

    graph_a = build_lineage_graph(index)
    graph_b = build_lineage_graph(index)
    assert graph_a.to_dict() == graph_b.to_dict()
    assert graph_a.graph_fingerprint
    assert any(edge.relationship == "evaluated_by" for edge in graph_a.edges)


def test_workspace_summary_and_drift_detection(tmp_path):
    workspace = _write_workspace(tmp_path)
    policy = load_artifact_policy(Path("configs/artifact_policy.yaml"))
    index = build_inventory_index(workspace, policy=policy)
    summary = build_workspace_summary(workspace, policy=policy, index=index)
    payload = summary.to_dict()
    assert payload["summary"]["total_artifacts"] >= 5
    assert payload["summary"]["pinned_artifacts"] >= 1
    assert payload["summary"]["archive_coverage"] >= 0.0
    assert "# CPT Workspace Summary" in summary.to_markdown()

    drift_before = detect_inventory_drift(index, workspace)
    assert drift_before == []

    (workspace / "eval_report.json").write_text(
        json.dumps(
            {
                "artifact_type": "evaluation_report",
                "schema_version": "2.7.7",
                "compatibility_status": "compatible",
                "policy": {"schema_version": "1.0"},
                "summary": {"checkpoint_artifact_fingerprint": "ckpt-1"},
                "per_layer": {},
                "per_module": {},
                "failure_summary": {"failure_counts": {}, "dominant_failure": "none"},
                "ood": {},
                "dataset_manifest": {"fingerprint": "mismatch"},
                "checkpoint": {"schema_version": "2.7.8"},
                "evaluation": {"fingerprint": "eval-1"},
                "artifacts": {},
                "report_fingerprint": "report-2",
                "created_at": 3.0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    drift_after = detect_inventory_drift(index, workspace)
    assert any(item.startswith("changed_fingerprint:eval_report.json") for item in drift_after)
    assert any(item == "stale_inventory_index" for item in drift_after)


def test_legacy_artifacts_remain_queryable_and_cli_reproducible(tmp_path):
    workspace = tmp_path / "legacy_workspace"
    workspace.mkdir()
    torch.save(
        {
            "state_dict": {"w": [1, 2]},
            "tokenizer": {"vocab": ["a", "b"]},
            "config": {"seed": 1},
            "artifact_fingerprint": "legacy-ckpt",
            "dataset_manifest_hash": "legacy-manifest",
            "snapshot_hash": "legacy-snap",
        },
        workspace / "legacy_checkpoint.pt",
    )
    (workspace / "dataset.manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "2.7.0",
                "schema_version": "2.7.0",
                "snapshot_hash": "legacy-snap",
                "module_hash": "legacy-module",
                "generation_seed": 1,
                "record_count": 1,
                "fingerprint": "legacy-manifest",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    index = build_inventory_index(workspace, policy=load_artifact_policy(Path("configs/artifact_policy.yaml")))
    legacy = query_inventory(index, artifact_type="checkpoint", fingerprint="legacy-ckpt")
    assert legacy and legacy[0].schema_version == "2.7.5"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    out1 = tmp_path / "inventory_1.json"
    out2 = tmp_path / "inventory_2.json"
    subprocess.run(
        [sys.executable, "scripts/build_inventory.py", "--workspace", str(workspace), "--index", str(out1)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, "scripts/build_inventory.py", "--workspace", str(workspace), "--index", str(out2)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
