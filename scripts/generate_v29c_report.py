#!/usr/bin/env python3
"""Generate the CPT v2.9C first surrogate validation report."""

from __future__ import annotations

import argparse
import json
import hashlib
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.governance.artifact_registry import ArtifactRegistry
from backend.neural.training_snapshot import stable_fingerprint
from scripts.train_circuit_gnn import load_training_profile
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training" / "kaggle_v29b.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "workspace" / "checkpoints" / "circuit_gnn_v29b.pt"
DEFAULT_ARENA = PROJECT_ROOT / "workspace" / "arena_reports" / "circuit_arena_report.json"
DEFAULT_FAILURE = PROJECT_ROOT / "workspace" / "failure_analysis" / "v29c_failure_analysis.json"
DEFAULT_DETERMINISM = PROJECT_ROOT / "workspace" / "determinism_checks" / "v29c_determinism.json"
DEFAULT_DOC = PROJECT_ROOT / "docs" / "V29C_FIRST_SURROGATE_VALIDATION.md"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "workspace" / "arena_reports" / "v29c_report.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _line_count(items: list[Any]) -> int:
    return len(items) if isinstance(items, list) else 0


def build_report(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    arena_path: str | Path = DEFAULT_ARENA,
    failure_path: str | Path = DEFAULT_FAILURE,
    determinism_path: str | Path = DEFAULT_DETERMINISM,
) -> dict[str, Any]:
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    arena_path = Path(arena_path)
    failure_path = Path(failure_path)
    determinism_path = Path(determinism_path)

    profile = load_training_profile(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arena = _load_json(arena_path)
    failure = _load_json(failure_path)
    determinism = _load_json(determinism_path)

    report = {
        "schema_version": "2.9c",
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "arena_path": str(arena_path),
        "failure_path": str(failure_path),
        "determinism_path": str(determinism_path),
        "training_profile": profile,
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
        "arena": arena,
        "failure": failure,
        "determinism": determinism,
        "summary": {
            "gnn_mae": arena.get("gnn", {}).get("in_distribution", {}).get("mae", 0.0),
            "mean_baseline_mae": arena.get("mean_baseline", {}).get("mae", 0.0),
            "linear_baseline_mae": arena.get("linear_baseline", {}).get("mae", 0.0),
            "ood_cases": failure.get("failure_summary", {}).get("count", 0),
            "dominant_failure": failure.get("failure_summary", {}).get("dominant_failure", "none"),
            "deterministic": bool(determinism.get("deterministic", False)),
        },
    }
    report["report_fingerprint"] = stable_fingerprint(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    checkpoint = report.get("checkpoint", {})
    arena = report.get("arena", {})
    failure = report.get("failure", {})
    determinism = report.get("determinism", {})
    gnn = arena.get("gnn", {})
    gnn_ind = gnn.get("in_distribution", {})
    gnn_ood = gnn.get("ood", {})
    mean_base = arena.get("mean_baseline", {})
    linear_base = arena.get("linear_baseline", {})
    random_base = arena.get("random_baseline", {})
    speed = failure.get("speed", {})
    invariants = failure.get("invariants", {})
    summary = report.get("summary", {})
    history = checkpoint.get("extra", {}).get("history", [])
    lines = [
        "# CPT v2.9C First Surrogate Validation",
        "",
        "## Experiment Goals",
        "",
        "- Execute the first end-to-end surrogate validation run.",
        "- Check determinism across repeated training and evaluation passes.",
        "- Measure IID performance, OOD behavior, invariants, and latency.",
        "",
        "## Dataset Details",
        "",
        f"- Dataset fingerprint: {checkpoint.get('dataset_manifest_hash', '')}",
        f"- Train count: {checkpoint.get('curriculum_coverage', {}).get('train_count', 0)}",
        f"- Eval count: {checkpoint.get('curriculum_coverage', {}).get('eval_count', 0)}",
        f"- OOD cases: {summary.get('ood_cases', 0)}",
        "",
        "## Model",
        "",
        f"- Type: {checkpoint.get('model_type', '')}",
        f"- Hidden dim: {checkpoint.get('model_config', {}).get('hidden_dim', 0)}",
        f"- Parameters: {checkpoint.get('model_config', {}).get('num_params', 0)}",
        f"- Artifact fingerprint: {checkpoint.get('artifact_fingerprint', '')}",
        "",
        "## Training Metrics",
        "",
        "| Epoch | Train Loss | Eval Loss | MAE V | RMSE V | Max Error V | LR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in history:
        lines.append(
            f"| {entry.get('epoch', 0)} | {entry.get('train_loss', 0):.6f} | {entry.get('eval_loss', 0):.6f} | "
            f"{entry.get('eval_mae_V', 0):.6f} | {entry.get('eval_rmse_V', 0):.6f} | {entry.get('eval_max_error_V', 0):.6f} | {entry.get('lr', 0):.2e} |"
        )
    if not history:
        lines.append("| 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.00e+00 |")
    lines.extend(
        [
            "",
            "## Determinism Verification",
            f"- Deterministic: {determinism.get('deterministic', False)}",
            f"- Metrics equal: {determinism.get('metrics_equal', False)}",
            f"- Run A fingerprint: {determinism.get('run_a_fingerprint', '')}",
            f"- Run B fingerprint: {determinism.get('run_b_fingerprint', '')}",
            "",
            "## Baseline Comparisons",
            f"- GNN MAE: {gnn_ind.get('mae', 0):.6f} V",
            f"- Mean baseline MAE: {mean_base.get('mae', 0):.6f} V",
            f"- Linear baseline MAE: {linear_base.get('mae', 0):.6f} V",
            f"- Random stable baseline MAE: {random_base.get('mae', 0):.6f} V",
            f"- GNN beats mean: {gnn_ind.get('mae', 0) < mean_base.get('mae', float('inf'))}",
            f"- GNN beats linear: {gnn_ind.get('mae', 0) < linear_base.get('mae', float('inf'))}",
            "",
            "## IID Performance",
            f"- IID MAE: {gnn_ind.get('mae', 0):.6f} V",
            f"- IID RMSE: {gnn_ind.get('rmse', 0):.6f} V",
            f"- IID max error: {gnn_ind.get('max_voltage_error', 0):.6f} V",
            f"- IID KCL max violation: {gnn_ind.get('kcl_max_violation', 0):.2e}",
            f"- IID KVL max violation: {gnn_ind.get('kvl_max_violation', 0):.2e}",
            "",
            "## OOD Performance",
            f"- OOD MAE: {gnn_ood.get('mae', 0):.6f} V",
            f"- OOD RMSE: {gnn_ood.get('rmse', 0):.6f} V",
            f"- OOD max error: {gnn_ood.get('max_voltage_error', 0):.6f} V",
            f"- OOD KCL max violation: {gnn_ood.get('kcl_max_violation', 0):.2e}",
            f"- OOD KVL max violation: {gnn_ood.get('kvl_max_violation', 0):.2e}",
            "",
            "## Failure Taxonomy Summary",
            f"- Dominant failure: {summary.get('dominant_failure', 'none')}",
            f"- OOD cases classified: {summary.get('ood_cases', 0)}",
            f"- Failure counts: {json.dumps(failure.get('failure_summary', {}).get('failure_counts', {}), sort_keys=True)}",
            "",
            "## Invariant Preservation",
            f"- IID KCL violation: {invariants.get('iid_kcl_violation', 0):.2e}",
            f"- OOD KCL violation: {invariants.get('ood_kcl_violation', 0):.2e}",
            f"- IID KVL violation: {invariants.get('iid_kvl_violation', 0):.2e}",
            f"- OOD KVL violation: {invariants.get('ood_kvl_violation', 0):.2e}",
            f"- IID power violation: {invariants.get('iid_power_violation', 0):.2e}",
            f"- OOD power violation: {invariants.get('ood_power_violation', 0):.2e}",
            f"- Replay max abs diff: {invariants.get('replay_max_abs_diff', 0):.2e}",
            "",
            "## Speedup Metrics",
            f"- Oracle mean latency: {speed.get('oracle_mean_sec', 0):.6f} s",
            f"- Oracle p95 latency: {speed.get('oracle_p95_sec', 0):.6f} s",
            f"- GNN mean latency: {speed.get('surrogate_mean_sec', 0):.6f} s",
            f"- GNN p95 latency: {speed.get('surrogate_p95_sec', 0):.6f} s",
            f"- Speedup: {speed.get('speedup', 0):.2f}x",
            "",
            "## Known Weaknesses",
            f"- Dominant observed failure mode: {summary.get('dominant_failure', 'none')}.",
            "- Baseline gaps should be interpreted conservatively on this dataset if the circuit topology is simple or low-variance.",
            "- OOD behavior is only as strong as the sampled OOD generator; it does not prove general circuit validity.",
            "",
            "## Recommended Next Steps",
            "- Extend validation coverage to harder OOD topologies before claiming broad generalization.",
            "- Compare the surrogate against more structured baselines if topology complexity increases.",
            "- Keep retraining and rerun validation under identical fingerprints to guard against drift.",
            "",
            "## Reproducibility",
            f"- Report fingerprint: {report.get('report_fingerprint', '')}",
            f"- Checkpoint fingerprint: {checkpoint.get('artifact_fingerprint', '')}",
            f"- Deterministic: {determinism.get('deterministic', False)}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the v2.9C validation report.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--arena", default=str(DEFAULT_ARENA))
    parser.add_argument("--failure", default=str(DEFAULT_FAILURE))
    parser.add_argument("--determinism", default=str(DEFAULT_DETERMINISM))
    parser.add_argument("--output-md", default=str(DEFAULT_DOC))
    parser.add_argument("--output-json", default=str(DEFAULT_REPORT_JSON))
    args = parser.parse_args()

    report = build_report(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        arena_path=args.arena,
        failure_path=args.failure,
        determinism_path=args.determinism,
    )
    markdown = render_markdown(report)

    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    registry_path = PROJECT_ROOT / "artifacts" / "artifact_registry.json"
    registry = ArtifactRegistry.from_file(registry_path) if registry_path.exists() else ArtifactRegistry(path=registry_path)
    registry.register(
        artifact_type="evaluation_report",
        schema_version="2.9c",
        fingerprint=report["report_fingerprint"],
        parent_fingerprints=[
            report.get("checkpoint", {}).get("artifact_fingerprint", ""),
            report.get("arena", {}).get("summary", {}).get("checkpoint_artifact_fingerprint", ""),
            report.get("failure", {}).get("analysis_fingerprint", ""),
        ],
        metadata={
            "output_md": str(output_md),
            "output_json": str(output_json),
        },
    )
    registry.save(registry_path)

    print(json.dumps({"written": str(output_md), "report_fingerprint": report["report_fingerprint"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
