#!/usr/bin/env python3
"""Generate the reproducible CPT v2.9B surrogate report."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any

import torch

from backend.governance.artifact_registry import ArtifactRegistry
from backend.neural.training_snapshot import load_training_snapshot, stable_fingerprint
from scripts.train_circuit_gnn import deterministic_split, load_training_data, load_training_profile, resolve_dataset_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training" / "kaggle_v29b.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "workspace" / "checkpoints" / "circuit_gnn_v29b.pt"
DEFAULT_ARENA = PROJECT_ROOT / "workspace" / "arena_results" / "circuit_arena_metrics.json"
DEFAULT_SNAPSHOT = PROJECT_ROOT / "workspace" / "training_snapshots" / "training_snapshot.json"
DEFAULT_KAGGLE_MANIFEST = PROJECT_ROOT / "workspace" / "kaggle_exports" / "v29b" / "kaggle_v29b_manifest.json"
DEFAULT_DOC = PROJECT_ROOT / "docs" / "V29B_REPRODUCIBLE_SURROGATE.md"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "workspace" / "arena_reports" / "v29b_report.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "workspace" / "arena_reports" / "v29b_report.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _graph_stats(graphs) -> dict[str, Any]:
    node_counts = [int(g.graph.node_features.size(0)) for g in graphs]
    edge_counts = [int(g.graph.edge_index.size(1)) for g in graphs]
    all_targets = torch.cat([g.graph.target_voltages for g in graphs]) if graphs else torch.tensor([])
    target_std = float(all_targets.std(unbiased=False).item()) if all_targets.numel() else 0.0
    return {
        "count": len(graphs),
        "node_count_mean": round(float(sum(node_counts) / max(len(node_counts), 1)), 4),
        "edge_count_mean": round(float(sum(edge_counts) / max(len(edge_counts), 1)), 4),
        "target_voltage_mean": round(float(all_targets.mean().item()) if all_targets.numel() else 0.0, 6),
        "target_voltage_std": round(target_std, 6),
        "target_voltage_min": round(float(all_targets.min().item()) if all_targets.numel() else 0.0, 6),
        "target_voltage_max": round(float(all_targets.max().item()) if all_targets.numel() else 0.0, 6),
    }


def _render_history(history: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for entry in history:
        lines.append(
            f"| {entry.get('epoch', 0)} | {entry.get('train_loss', 0):.6f} | {entry.get('eval_loss', 0):.6f} | "
            f"{entry.get('eval_mae_V', 0):.6f} | {entry.get('eval_rmse_V', 0):.6f} | {entry.get('eval_max_error_V', 0):.6f} | {entry.get('lr', 0):.2e} |"
        )
    return lines


def build_v29b_report(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    arena_path: str | Path = DEFAULT_ARENA,
    snapshot_path: str | Path = DEFAULT_SNAPSHOT,
    kaggle_manifest_path: str | Path = DEFAULT_KAGGLE_MANIFEST,
) -> dict[str, Any]:
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    arena_path = Path(arena_path)
    snapshot_path = Path(snapshot_path)
    kaggle_manifest_path = Path(kaggle_manifest_path)

    profile = load_training_profile(config_path)
    dataset_path = resolve_dataset_path(profile["dataset"]["train_path"])
    all_graphs = load_training_data(dataset_path)
    train_graphs, eval_graphs = deterministic_split(all_graphs, 1.0 - float(profile["dataset"]["eval_split"]))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    snapshot = load_training_snapshot(snapshot_path)
    arena = _load_json(arena_path)
    kaggle = _load_json(kaggle_manifest_path)

    history = checkpoint.get("extra", {}).get("history", [])
    report = {
        "schema_version": "2.9b",
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "snapshot_path": str(snapshot_path),
        "arena_path": str(arena_path),
        "kaggle_manifest_path": str(kaggle_manifest_path),
        "training_profile": profile,
        "dataset": {
            "raw": _graph_stats(all_graphs),
            "train": _graph_stats(train_graphs),
            "eval": _graph_stats(eval_graphs),
            "fingerprint": stable_fingerprint([g.graph.fingerprint for g in all_graphs]),
        },
        "checkpoint": {
            "artifact_fingerprint": checkpoint.get("artifact_fingerprint", ""),
            "model_type": checkpoint.get("model_type", ""),
            "model_config": checkpoint.get("model_config", {}),
            "training_config": checkpoint.get("training_config", {}),
            "dataset_manifest_hash": checkpoint.get("dataset_manifest_hash", ""),
            "snapshot_hash": checkpoint.get("snapshot_hash", ""),
            "eval_fingerprint": checkpoint.get("eval_fingerprint", ""),
            "curriculum_coverage": checkpoint.get("curriculum_coverage", {}),
            "extra": checkpoint.get("extra", {}),
        },
        "summary": {
            "checkpoint_artifact_fingerprint": checkpoint.get("artifact_fingerprint", ""),
            "dataset_manifest_hash": checkpoint.get("dataset_manifest_hash", ""),
            "snapshot_hash": checkpoint.get("snapshot_hash", ""),
            "evaluation_fingerprint": checkpoint.get("eval_fingerprint", ""),
            "parent_oracle_version": checkpoint.get("extra", {}).get("parent_oracle_version", "v2.8"),
        },
        "snapshot": snapshot.to_dict(),
        "arena": arena,
        "kaggle": kaggle,
        "training_history": history,
    }
    report["report_fingerprint"] = stable_fingerprint(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    checkpoint = report.get("checkpoint", {})
    dataset = report.get("dataset", {})
    arena = report.get("arena", {})
    gnn = arena.get("gnn", {})
    gnn_ind = gnn.get("in_distribution", {})
    gnn_ood = gnn.get("ood", {})
    mean_base = arena.get("mean_baseline", {})
    linear_base = arena.get("linear_baseline", {})
    random_base = arena.get("random_baseline", {})
    snapshot = report.get("snapshot", {})
    kaggle = report.get("kaggle", {})
    lines = [
        "# CPT v2.9B Reproducible Surrogate",
        "",
        "## Architecture",
        "",
        "- Oracle-generated circuit graphs feed an edge-aware GNN surrogate.",
        "- Training uses deterministic per-circuit voltage normalization and invariant-aware losses.",
        "- Evaluation compares the GNN against mean, linear, and random-stable baselines.",
        "",
        "## Dataset Stats",
        "",
        f"- Raw graphs: {dataset.get('raw', {}).get('count', 0)}",
        f"- Train graphs: {dataset.get('train', {}).get('count', 0)}",
        f"- Eval graphs: {dataset.get('eval', {}).get('count', 0)}",
        f"- Mean node count: {dataset.get('raw', {}).get('node_count_mean', 0):.4f}",
        f"- Mean edge count: {dataset.get('raw', {}).get('edge_count_mean', 0):.4f}",
        f"- Target voltage mean: {dataset.get('raw', {}).get('target_voltage_mean', 0):.6f}",
        f"- Target voltage std: {dataset.get('raw', {}).get('target_voltage_std', 0):.6f}",
        "",
        "## Model",
        "",
        f"- Type: {checkpoint.get('model_type', '')}",
        f"- Hidden dim: {checkpoint.get('model_config', {}).get('hidden_dim', 0)}",
        f"- Parameters: {checkpoint.get('model_config', {}).get('num_params', 0)}",
        f"- Checkpoint fingerprint: {checkpoint.get('artifact_fingerprint', '')}",
        "",
        "## Training Curves",
        "",
        "| Epoch | Train Loss | Eval Loss | MAE V | RMSE V | Max Error V | LR |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *(_render_history(report.get("training_history", [])) or ["| 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.00e+00 |"]),
        "",
        "## Invariant Metrics",
        "",
        f"- GNN KCL max violation: {gnn_ind.get('kcl_max_violation', 0):.2e}",
        f"- GNN KVL max violation: {gnn_ind.get('kvl_max_violation', 0):.2e}",
        f"- GNN replay consistency: {gnn_ind.get('replay_consistency', 0):.2e}",
        f"- OOD KCL max violation: {gnn_ood.get('kcl_max_violation', 0):.2e}",
        f"- OOD KVL max violation: {gnn_ood.get('kvl_max_violation', 0):.2e}",
        "",
        "## Baselines",
        "",
        f"- Mean baseline MAE: {mean_base.get('mae', 0):.6f} V",
        f"- Linear baseline MAE: {linear_base.get('mae', 0):.6f} V",
        f"- Random stable baseline MAE: {random_base.get('mae', 0):.6f} V",
        f"- GNN beats mean: {gnn_ind.get('mae', 0) < mean_base.get('mae', float('inf'))}",
        f"- GNN beats linear: {gnn_ind.get('mae', 0) < linear_base.get('mae', float('inf'))}",
        "",
        "## OOD Behavior",
        "",
        f"- OOD circuits: {gnn_ood.get('count', 0)}",
        f"- OOD MAE: {gnn_ood.get('mae', 0):.6f} V",
        f"- OOD RMSE: {gnn_ood.get('rmse', 0):.6f} V",
        f"- OOD max error: {gnn_ood.get('max_voltage_error', 0):.6f} V",
        "",
        "## Speedup",
        "",
        f"- Oracle solve mean: {gnn.get('speed_in_distribution', {}).get('oracle_mean_sec', 0)*1000:.3f} ms",
        f"- Surrogate inference mean: {gnn.get('speed_in_distribution', {}).get('surrogate_mean_sec', 0)*1000:.3f} ms",
        f"- Speedup: {gnn.get('speed_in_distribution', {}).get('speedup', 0):.2f}x",
        "",
        "## Reproducibility Guarantees",
        "",
        f"- Dataset fingerprint: {checkpoint.get('dataset_manifest_hash', '')}",
        f"- Config fingerprint: {checkpoint.get('extra', {}).get('config_fingerprint', '')}",
        f"- Snapshot fingerprint: {snapshot.get('artifact_fingerprint', '')}",
        f"- Evaluation fingerprint: {checkpoint.get('eval_fingerprint', '')}",
        f"- Git commit: {snapshot.get('git_commit', '')}",
        f"- Torch version: {snapshot.get('torch_version', '')}",
        f"- CUDA enabled: {snapshot.get('cuda_enabled', False)}",
        f"- Device name: {snapshot.get('device_name', '')}",
        "",
        "## Kaggle Metadata",
        "",
        f"- Kaggle profile: {kaggle.get('profile', '')}",
        f"- Kaggle fingerprint: {kaggle.get('profile_fingerprint', '')}",
        f"- Kaggle dataset fingerprint: {kaggle.get('dataset_fingerprint', '')}",
        f"- Kaggle git commit: {kaggle.get('git_commit', '')}",
        f"- Kaggle run command: {kaggle.get('run_command', '')}",
        "",
        "## Artifact Lineage",
        "",
        f"- Parent oracle version: {checkpoint.get('extra', {}).get('parent_oracle_version', 'v2.8')}",
        f"- Report fingerprint: {report.get('report_fingerprint', '')}",
        f"- Arena fingerprint: {arena.get('metadata', {}).get('checkpoint', {}).get('artifact_fingerprint', '')}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the v2.9B reproducible surrogate report.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--arena", default=str(DEFAULT_ARENA))
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--kaggle-manifest", default=str(DEFAULT_KAGGLE_MANIFEST))
    parser.add_argument("--output-md", default=str(DEFAULT_DOC))
    parser.add_argument("--output-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--workspace-md", default=str(DEFAULT_REPORT_MD))
    args = parser.parse_args()

    report = build_v29b_report(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        arena_path=args.arena,
        snapshot_path=args.snapshot,
        kaggle_manifest_path=args.kaggle_manifest,
    )
    markdown = render_markdown(report)

    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    workspace_md = Path(args.workspace_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    workspace_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    workspace_md.write_text(markdown, encoding="utf-8")

    registry_path = PROJECT_ROOT / "artifacts" / "artifact_registry.json"
    registry = ArtifactRegistry.from_file(registry_path) if registry_path.exists() else ArtifactRegistry(path=registry_path)
    registry.register(
        artifact_type="evaluation_report",
        schema_version="2.9b",
        fingerprint=report["report_fingerprint"],
        parent_fingerprints=[
            report.get("checkpoint", {}).get("artifact_fingerprint", ""),
            report.get("snapshot", {}).get("artifact_fingerprint", ""),
            report.get("arena", {}).get("metadata", {}).get("checkpoint", {}).get("artifact_fingerprint", ""),
        ],
        metadata={
            "output_md": str(output_md),
            "output_json": str(output_json),
            "workspace_md": str(workspace_md),
        },
    )
    registry.save(registry_path)

    print(json.dumps({"written": str(output_md), "report_fingerprint": report["report_fingerprint"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
