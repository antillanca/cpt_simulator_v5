"""Compact deterministic evaluation report generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.datasets.manifest import DatasetManifest
from backend.governance.artifact_policy import (
    ArtifactPolicy,
    ArtifactPolicyError,
    artifact_policy_fingerprint,
    get_artifact_policy,
)
from backend.governance.artifact_registry import ArtifactRegistry
from backend.neural.checkpoints import checkpoint_summary, infer_checkpoint_version, validate_checkpoint_payload
from backend.reporting.failure_summary import FailureSummary, summarize_failures


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_checkpoint_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    import torch

    payload = torch.load(path, map_location="cpu")
    validate_checkpoint_payload(payload, allow_legacy=True)
    if infer_checkpoint_version(payload) != "2.7.6":
        from backend.neural.checkpoints.migrations.v2_7_5_to_v2_7_6 import migrate_payload_v275_to_v276

        payload = migrate_payload_v275_to_v276(payload).payload
    return checkpoint_summary(payload)


def _policy_context(policy: ArtifactPolicy | None, *, strict_policy: bool = False) -> dict[str, Any]:
    if policy is None:
        return {}
    return {
        "schema_version": policy.schema_version,
        "fingerprint": artifact_policy_fingerprint(policy),
        "enforcement_mode": "strict" if strict_policy or bool(policy.enforcement.get("strict_mode", False)) else "permissive",
        "legacy_read": bool(policy.defaults.get("allow_legacy_read", False)),
        "legacy_write": bool(policy.defaults.get("allow_legacy_write", False)),
        "required_fingerprint": bool(policy.defaults.get("require_fingerprint", False)),
        "compatibility": dict(sorted(policy.compatibility.items())),
    }


def _load_dataset_manifest(report: dict[str, Any], dataset_manifest_path: Path | None = None) -> dict[str, Any]:
    candidates: list[Path] = []
    if dataset_manifest_path is not None:
        candidates.append(dataset_manifest_path)
    dataset_path = report.get("dataset_path")
    if dataset_path:
        dataset_path_obj = Path(dataset_path)
        if dataset_path_obj.is_dir():
            candidates.append(dataset_path_obj / "dataset.manifest.json")
        else:
            candidates.append(dataset_path_obj.with_suffix(".manifest.json"))
    for candidate in candidates:
        if candidate.exists():
            return DatasetManifest.from_file(candidate).to_dict()
    return {}


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _rate(items: list[dict[str, Any]], predicate) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if predicate(item)) / float(len(items))


def _validate_report_policy(payload: dict[str, Any], policy: ArtifactPolicy | None, *, strict_policy: bool = False) -> None:
    if policy is None:
        return
    artifact_policy = get_artifact_policy("evaluation_report", policy)
    if payload.get("artifact_type") != "evaluation_report":
        raise ArtifactPolicyError("artifact_type must be evaluation_report")
    if payload.get("schema_version") != "2.7.7":
        raise ArtifactPolicyError("schema_version must be 2.7.7 for evaluation reports")
    if strict_policy or bool(policy.enforcement.get("strict_mode", False)):
        missing = [field for field in artifact_policy.required_fields if field not in payload]
        if missing:
            raise ArtifactPolicyError("Missing required report field(s): " + ", ".join(missing))


def validate_evaluation_report(
    payload: dict[str, Any],
    *,
    policy: ArtifactPolicy | None = None,
    strict_policy: bool = False,
) -> None:
    _validate_report_policy(payload, policy, strict_policy=strict_policy)


@dataclass
class EvaluationReport:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        normalized = _normalize(self.payload)
        normalized["report_fingerprint"] = _stable_hash(normalized)
        return normalized

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        report = self.to_dict()
        summary = report["summary"]
        policy = report.get("policy", {})
        lines = [
            "# CPT Evaluation Report",
            "",
            "## Summary",
            f"- Model: {summary.get('model_type', 'unknown')}",
            f"- Dataset: {summary.get('dataset_name', 'unknown')}",
            f"- Exact Match: {summary.get('exact_match', 0.0):.1%}",
            f"- Structural Match: {summary.get('structural_match', 0.0):.1%}",
            f"- Invariant Retention: {summary.get('invariant_retention', 0.0):.1%}",
            f"- Replay Stability: {summary.get('replay_stability', 0.0):.1%}",
            f"- Dominant Failure: {summary.get('dominant_failure', 'none')}",
            "",
            "## Policy Context",
            f"- Policy Schema: {policy.get('schema_version', 'unknown') or 'unknown'}",
            f"- Enforcement: {policy.get('enforcement_mode', 'permissive') or 'permissive'}",
            f"- Legacy Read: {'enabled' if policy.get('legacy_read') else 'disabled'}",
            f"- Legacy Write: {'enabled' if policy.get('legacy_write') else 'disabled'}",
            f"- Required Fingerprint: {'yes' if policy.get('required_fingerprint') else 'no'}",
            "",
            "## Per-Layer Breakdown",
            "",
            "| Layer | Exact Match | Invariant Retention | Dominant Failure |",
            "|-------|-------------|--------------------|------------------|",
        ]
        for layer, metrics in sorted(report.get("per_layer", {}).items(), key=lambda item: str(item[0])):
            lines.append(
                f"| {layer} | {metrics.get('exact_match_rate', 0.0):.1%} | {metrics.get('invariant_retention_rate', 0.0):.1%} | {metrics.get('dominant_failure', 'none')} |"
            )
        lines.extend(
            [
                "",
                "## Failure Taxonomy",
                "",
                "| Failure Type | Count |",
                "|--------------|-------|",
            ]
        )
        for failure_type, count in sorted(report.get("failure_summary", {}).get("failure_counts", {}).items()):
            lines.append(f"| {failure_type} | {count} |")
        return "\n".join(lines)

    def save(self, path: str | Path, *, markdown: bool = False) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = self.to_markdown() if markdown else self.to_json()
        path.write_text(text, encoding="utf-8")
        return path


def build_evaluation_report(
    evaluation_run: dict[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    dataset_manifest_path: str | Path | None = None,
    policy: ArtifactPolicy | None = None,
    strict_policy: bool = False,
    seed: int = 0,
) -> EvaluationReport:
    evaluation_run = dict(evaluation_run)
    evaluation = dict(evaluation_run.get("evaluation", {}))
    metrics = dict(evaluation.get("metrics", {}))
    arena_results = list(evaluation_run.get("arena", {}).get("results", []))
    failure_summary = summarize_failures(arena_results).to_dict()
    checkpoint_summary_payload = _load_checkpoint_summary(Path(checkpoint_path) if checkpoint_path else None)
    dataset_manifest = _load_dataset_manifest(evaluation_run, Path(dataset_manifest_path) if dataset_manifest_path else None)
    per_layer = dict(evaluation_run.get("arena", {}).get("by_layer", {}))
    per_module = dict(evaluation_run.get("arena", {}).get("by_module", {}))
    registry = ArtifactRegistry(path=Path("artifacts") / "artifact_registry.json")
    policy_context = _policy_context(policy, strict_policy=strict_policy)

    summary = {
        "model_type": evaluation_run.get("model_type", evaluation.get("model_type", "unknown")),
        "dataset_name": Path(evaluation_run.get("dataset_path", "unknown")).stem,
        "dataset_path": evaluation_run.get("dataset_path", ""),
        "dataset_manifest_hash": dataset_manifest.get("fingerprint", ""),
        "snapshot_hash": checkpoint_summary_payload.get("snapshot_hash", ""),
        "checkpoint_schema_version": checkpoint_summary_payload.get("schema_version", ""),
        "checkpoint_artifact_fingerprint": checkpoint_summary_payload.get("artifact_fingerprint", ""),
        "model_checkpoint_hash": checkpoint_summary_payload.get("weights_hash", ""),
        "evaluation_fingerprint": evaluation.get("fingerprint", ""),
        "report_seed": int(seed),
        "total_samples": int(evaluation_run.get("total_samples", 0)),
        "pass_count": int(sum(1 for item in arena_results if not item.get("failure_type"))),
        "fail_count": int(sum(1 for item in arena_results if item.get("failure_type"))),
        "exact_match": _metric_value(metrics, "exact_match_rate"),
        "structural_match": _metric_value(metrics, "token_or_struct_match_rate"),
        "invariant_retention": _metric_value(metrics, "invariant_retention_rate"),
        "replay_stability": _metric_value(metrics, "replay_stability"),
        "ood_exact_match": _metric_value(metrics, "ood_exact_match_rate"),
        "extrapolation_robustness": _metric_value(metrics, "extrapolation_robustness"),
        "dominant_failure": failure_summary.get("dominant_failure", "none"),
        "policy_schema_version": policy_context.get("schema_version", ""),
        "policy_fingerprint": policy_context.get("fingerprint", ""),
        "policy_enforcement_mode": policy_context.get("enforcement_mode", ""),
        "policy_legacy_read": policy_context.get("legacy_read", False),
        "policy_legacy_write": policy_context.get("legacy_write", False),
        "policy_required_fingerprint": policy_context.get("required_fingerprint", False),
    }

    per_layer_summary: dict[str, Any] = {}
    for layer, metrics_by_group in sorted(per_layer.items(), key=lambda item: str(item[0])):
        layer_results = [item for item in arena_results if str(item.get("curriculum_layer", -1)) == str(layer)]
        layer_failure_summary = summarize_failures(layer_results).to_dict()
        per_layer_summary[str(layer)] = {
            "sample_count": int(metrics_by_group.get("sample_count", 0)),
            "exact_match_rate": float(metrics_by_group.get("exact_match_rate", 0.0)),
            "invariant_retention_rate": float(metrics_by_group.get("invariant_retention_rate", metrics.get("invariant_retention_rate", 0.0))),
            "replay_stability": float(metrics_by_group.get("replay_consistency", metrics.get("replay_stability", 0.0))),
            "failure_counts": layer_failure_summary.get("failure_counts", {}),
            "dominant_failure": layer_failure_summary.get("dominant_failure", "none"),
            "invariant_violations": int(layer_failure_summary.get("by_layer", {}).get(str(layer), {}).get("invariant_violations", 0)),
            "replay_instability": int(layer_failure_summary.get("by_layer", {}).get(str(layer), {}).get("replay_instability", 0)),
            "compositional_failures": int(layer_failure_summary.get("by_layer", {}).get(str(layer), {}).get("compositional_failure", 0)),
        }

    per_module_summary: dict[str, Any] = {}
    for module, metrics_by_group in sorted(per_module.items(), key=lambda item: str(item[0])):
        module_results = [item for item in arena_results if str(item.get("module_source", "unknown")) == str(module)]
        module_failure_summary = summarize_failures(module_results).to_dict()
        per_module_summary[str(module)] = {
            "sample_count": int(metrics_by_group.get("sample_count", 0)),
            "exact_match_rate": float(metrics_by_group.get("exact_match_rate", 0.0)),
            "invariant_retention_rate": float(metrics_by_group.get("invariant_retention_rate", metrics.get("invariant_retention_rate", 0.0))),
            "replay_stability": float(metrics_by_group.get("replay_consistency", metrics.get("replay_stability", 0.0))),
            "failure_counts": module_failure_summary.get("failure_counts", {}),
            "dominant_failure": module_failure_summary.get("dominant_failure", "none"),
        }

    ood_results = [item for item in arena_results if item.get("is_ood")]
    ind_results = [item for item in arena_results if not item.get("is_ood")]
    ood_summary = {
        "ood_sample_count": len(ood_results),
        "in_distribution_sample_count": len(ind_results),
        "ood_exact_match_rate": _rate(ood_results, lambda item: item.get("exact_match")),
        "in_distribution_exact_match_rate": _rate(ind_results, lambda item: item.get("exact_match")),
    }
    summary["ood_exact_match"] = ood_summary["ood_exact_match_rate"]
    summary["in_distribution_exact_match"] = ood_summary["in_distribution_exact_match_rate"]

    payload = {
        "artifact_type": "evaluation_report",
        "schema_version": "2.7.7",
        "compatibility_status": "compatible",
        "policy": policy_context,
        "summary": summary,
        "per_layer": per_layer_summary,
        "per_module": per_module_summary,
        "failure_summary": failure_summary,
        "ood": ood_summary,
        "dataset_manifest": dataset_manifest,
        "checkpoint": checkpoint_summary_payload,
        "evaluation": evaluation,
        "artifacts": registry.to_dict(),
    }
    _validate_report_policy(payload, policy, strict_policy=strict_policy)
    report = EvaluationReport(payload)
    report.to_dict()
    return report
