#!/usr/bin/env python3
"""Generate the CPT v2.9D physics-informed comparison report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.governance.artifact_registry import ArtifactRegistry

DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "V29D_PHYSICS_INFORMED_COMPARISON.md"
DEFAULT_JSON = PROJECT_ROOT / "workspace" / "arena_reports" / "v29d_comparison.json"
DEFAULT_ARC_C = PROJECT_ROOT / "workspace" / "arena_results" / "circuit_arena_metrics.json"
DEFAULT_ARC_D = PROJECT_ROOT / "workspace" / "arena_results" / "circuit_arena_metrics_v29d.json"
DEFAULT_INV_C = PROJECT_ROOT / "workspace" / "invariant_validation" / "v29c_invariants.json"
DEFAULT_INV_D = PROJECT_ROOT / "workspace" / "invariant_validation" / "v29d_invariants.json"
DEFAULT_DET_C = PROJECT_ROOT / "workspace" / "determinism_checks" / "v29c_determinism.json"
DEFAULT_DET_D = PROJECT_ROOT / "workspace" / "determinism_checks" / "v29d_determinism.json"
DEFAULT_FAIL_C = PROJECT_ROOT / "workspace" / "failure_analysis" / "v29c_failure_analysis.json"
DEFAULT_FAIL_D = PROJECT_ROOT / "workspace" / "failure_analysis" / "v29d_failure_analysis.json"
DEFAULT_METRICS_D = PROJECT_ROOT / "workspace" / "checkpoints" / "circuit_gnn_v29d.metrics.json"


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_report(
    *,
    arena_c_path: str | Path = DEFAULT_ARC_C,
    arena_d_path: str | Path = DEFAULT_ARC_D,
    inv_c_path: str | Path = DEFAULT_INV_C,
    inv_d_path: str | Path = DEFAULT_INV_D,
    det_c_path: str | Path = DEFAULT_DET_C,
    det_d_path: str | Path = DEFAULT_DET_D,
    fail_c_path: str | Path = DEFAULT_FAIL_C,
    fail_d_path: str | Path = DEFAULT_FAIL_D,
    metrics_d_path: str | Path = DEFAULT_METRICS_D,
) -> dict[str, Any]:
    arena_c = _load_json(arena_c_path)
    arena_d = _load_json(arena_d_path)
    inv_c = _load_json(inv_c_path)
    inv_d = _load_json(inv_d_path)
    det_c = _load_json(det_c_path)
    det_d = _load_json(det_d_path)
    fail_c = _load_json(fail_c_path)
    fail_d = _load_json(fail_d_path)
    metrics_d = _load_json(metrics_d_path)

    report = {
        "schema_version": "2.9d",
        "arena_v29c": arena_c,
        "arena_v29d": arena_d,
        "invariants_v29c": inv_c,
        "invariants_v29d": inv_d,
        "determinism_v29c": det_c,
        "determinism_v29d": det_d,
        "failure_v29c": fail_c,
        "failure_v29d": fail_d,
        "metrics_v29d": metrics_d,
    }

    table = [
        ("IID MAE (V)", arena_c.get("gnn", {}).get("in_distribution", {}).get("mae", 0.0), arena_d.get("gnn", {}).get("in_distribution", {}).get("mae", 0.0), "< 5.0"),
        ("OOD MAE (V)", arena_c.get("gnn", {}).get("ood", {}).get("mae", 0.0), arena_d.get("gnn", {}).get("ood", {}).get("mae", 0.0), "< 50"),
        ("KCL violation", arena_c.get("gnn", {}).get("in_distribution", {}).get("kcl_max_violation", 0.0), arena_d.get("gnn", {}).get("in_distribution", {}).get("kcl_max_violation", 0.0), "< 1e-3"),
        ("KVL violation", arena_c.get("gnn", {}).get("in_distribution", {}).get("kvl_max_violation", 0.0), arena_d.get("gnn", {}).get("in_distribution", {}).get("kvl_max_violation", 0.0), "< 1e-3"),
        ("Power violation (64-case slice)", inv_c.get("ood_power_violation", 0.0), inv_d.get("ood_power_violation", 0.0), "< 1e-3"),
        ("Speedup", arena_c.get("gnn", {}).get("speed_in_distribution", {}).get("speedup", 0.0), arena_d.get("gnn", {}).get("speed_in_distribution", {}).get("speedup", 0.0), "> 1x"),
    ]

    report["table"] = [
        {
            "metric": metric,
            "v29c": v29c,
            "v29d": v29d,
            "target": target,
        }
        for metric, v29c, v29d, target in table
    ]
    report["report_fingerprint"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return report


def render_markdown(report: dict[str, Any]) -> str:
    arena_d = report.get("arena_v29d", {})
    metrics_d = report.get("metrics_v29d", {})
    det_c = report.get("determinism_v29c", {})
    det_d = report.get("determinism_v29d", {})
    fail_c = report.get("failure_v29c", {})
    fail_d = report.get("failure_v29d", {})
    inv_c = report.get("invariants_v29c", {})
    inv_d = report.get("invariants_v29d", {})

    lines = [
        "# CPT v2.9D Physics-Informed Comparison",
        "",
        "## Hypothesis",
        "",
        "Adding physics-based invariant penalties should reduce KCL/KVL/power violations while preserving deterministic training and not degrading voltage accuracy too much.",
        "",
        "## Setup",
        "",
        "- Same circuit domain as v2.9C.",
        "- Same deterministic subset and split.",
        "- Same evaluation flow and same seed.",
        "- Training objective changed to voltage + KCL + KVL + power penalties.",
        "",
        "## Comparison Table",
        "",
        "| Metric | v2.9C | v2.9D | Target |",
        "|---|---:|---:|---:|",
    ]
    for row in report.get("table", []):
        lines.append(f"| {row['metric']} | {row['v29c']:.6f} | {row['v29d']:.6f} | {row['target']} |")
    lines.extend(
        [
            "",
            "## Training",
            f"- v2.9D seed: {metrics_d.get('seed', 0)}",
            f"- v2.9D best epoch: {metrics_d.get('best_epoch', 0)}",
            f"- v2.9D checkpoint fingerprint: {metrics_d.get('checkpoint_fingerprint', '')}",
            f"- v2.9D physics weights: {json.dumps(metrics_d.get('physics', {}), sort_keys=True)}",
            "",
            "### Curve",
            "| Epoch | Train Loss | Eval Loss | Eval MAE | Eval KCL | Eval KVL | Eval Power | Train Time (s) | Eval Time (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in metrics_d.get("history", []):
        lines.append(
            f"| {entry.get('epoch', 0)} | {entry.get('train_loss', 0):.6f} | {entry.get('eval_loss', 0):.6f} | "
            f"{entry.get('eval_mae_V', 0):.6f} | {entry.get('eval_kcl_penalty', 0):.2e} | {entry.get('eval_kvl_penalty', 0):.2e} | {entry.get('eval_power_penalty', 0):.2e} | "
            f"{entry.get('train_time_sec', 0):.2f} | {entry.get('eval_time_sec', 0):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Determinism",
            f"- v2.9C deterministic: {det_c.get('deterministic', False)}",
            f"- v2.9D deterministic: {det_d.get('deterministic', False)}",
            f"- v2.9C fingerprint: {det_c.get('run_a_fingerprint', '')}",
            f"- v2.9D fingerprint: {det_d.get('run_a_fingerprint', '')}",
            "",
            "## Failure Taxonomy",
            f"- v2.9C dominant failure: {fail_c.get('failure_summary', {}).get('dominant_failure', 'none')}",
            f"- v2.9D dominant failure: {fail_d.get('failure_summary', {}).get('dominant_failure', 'none')}",
            f"- v2.9C failure counts: {json.dumps(fail_c.get('failure_summary', {}).get('failure_counts', {}), sort_keys=True)}",
            f"- v2.9D failure counts: {json.dumps(fail_d.get('failure_summary', {}).get('failure_counts', {}), sort_keys=True)}",
            "",
            "## Invariant Notes",
            f"- v2.9C OOD power violation (64-case slice): {inv_c.get('ood_power_violation', 0.0):.2e}",
            f"- v2.9D OOD power violation (64-case slice): {inv_d.get('ood_power_violation', 0.0):.2e}",
            f"- v2.9C OOD KVL violation (64-case slice): {inv_c.get('ood_kvl_violation', 0.0):.2e}",
            f"- v2.9D OOD KVL violation (64-case slice): {inv_d.get('ood_kvl_violation', 0.0):.2e}",
            "",
            "## Honest Assessment",
            "- The physics-informed objective preserved determinism.",
            "- IID MAE stayed within the allowed band, but it did not improve materially.",
            "- Arena MAE and replay consistency did not improve versus v2.9C.",
            "- OOD invariant drift improved on the fixed 64-case analysis slice, especially power and KVL, but remained far above the target thresholds.",
            "- The hypothesis is only partially supported; the limiting factor appears to be that the surrogate is still too weak to satisfy hard invariants on this circuit family without a stronger representation or longer optimization schedule.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the v2.9D comparison report.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    args = parser.parse_args()

    report = build_report()
    markdown = render_markdown(report)

    output = Path(args.output)
    output_json = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    registry_path = PROJECT_ROOT / "artifacts" / "artifact_registry.json"
    registry = ArtifactRegistry.from_file(registry_path) if registry_path.exists() else ArtifactRegistry(path=registry_path)
    registry.register(
        artifact_type="evaluation_report",
        schema_version="2.9d",
        fingerprint=report["report_fingerprint"],
        parent_fingerprints=[
            report.get("metrics_v29d", {}).get("checkpoint_fingerprint", ""),
            report.get("arena_v29d", {}).get("summary", {}).get("checkpoint_artifact_fingerprint", ""),
            report.get("failure_v29d", {}).get("analysis_fingerprint", ""),
        ],
        metadata={
            "output": str(output),
            "output_json": str(output_json),
        },
    )
    registry.save(registry_path)

    print(json.dumps({"written": str(output), "output_json": str(output_json)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
