"""Diffing for evaluation reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_float_delta(a: Any, b: Any) -> float:
    return float(b or 0.0) - float(a or 0.0)


def _group_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(baseline) | set(candidate))
    delta: dict[str, Any] = {}
    for key in keys:
        left = baseline.get(key, {})
        right = candidate.get(key, {})
        if isinstance(left, dict) and isinstance(right, dict):
            metric_keys = sorted(set(left) | set(right))
            delta[key] = {
                metric: _stable_float_delta(left.get(metric, 0.0), right.get(metric, 0.0))
                for metric in metric_keys
                if isinstance(left.get(metric, 0.0), (int, float)) or isinstance(right.get(metric, 0.0), (int, float))
            }
    return delta


@dataclass
class EvalDiffResult:
    baseline_fingerprint: str
    candidate_fingerprint: str
    baseline_checkpoint_version: str
    candidate_checkpoint_version: str
    same_fingerprint: bool
    metric_deltas: dict[str, float] = field(default_factory=dict)
    layer_deltas: dict[str, dict[str, float]] = field(default_factory=dict)
    module_deltas: dict[str, dict[str, float]] = field(default_factory=dict)
    failure_deltas: dict[str, int] = field(default_factory=dict)
    invariant_retention_delta: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_fingerprint": self.baseline_fingerprint,
            "candidate_fingerprint": self.candidate_fingerprint,
            "baseline_checkpoint_version": self.baseline_checkpoint_version,
            "candidate_checkpoint_version": self.candidate_checkpoint_version,
            "same_fingerprint": self.same_fingerprint,
            "metric_deltas": dict(sorted(self.metric_deltas.items())),
            "layer_deltas": {key: dict(sorted(value.items())) for key, value in sorted(self.layer_deltas.items())},
            "module_deltas": {key: dict(sorted(value.items())) for key, value in sorted(self.module_deltas.items())},
            "failure_deltas": dict(sorted(self.failure_deltas.items())),
            "invariant_retention_delta": self.invariant_retention_delta,
            "warnings": sorted(self.warnings),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        payload = self.to_dict()
        lines = [
            "# CPT Evaluation Diff",
            "",
            "## Summary",
            f"- Baseline fingerprint: {payload['baseline_fingerprint']}",
            f"- Candidate fingerprint: {payload['candidate_fingerprint']}",
            f"- Same fingerprint: {payload['same_fingerprint']}",
            f"- Baseline checkpoint version: {payload['baseline_checkpoint_version'] or 'unknown'}",
            f"- Candidate checkpoint version: {payload['candidate_checkpoint_version'] or 'unknown'}",
            f"- Invariant retention delta: {payload['invariant_retention_delta']:+.4f}",
            "",
            "## Metric Deltas",
            "",
            "| Metric | Delta |",
            "|--------|-------|",
        ]
        for metric, delta in sorted(payload["metric_deltas"].items()):
            lines.append(f"| {metric} | {delta:+.4f} |")
        lines.extend(
            [
                "",
                "## Failure Deltas",
                "",
                "| Failure Type | Delta |",
                "|--------------|-------|",
            ]
        )
        for failure_type, delta in sorted(payload["failure_deltas"].items()):
            lines.append(f"| {failure_type} | {delta:+d} |")
        return "\n".join(lines)


def diff_eval_reports(baseline: Path, candidate: Path) -> EvalDiffResult:
    baseline_payload = _load(Path(baseline))
    candidate_payload = _load(Path(candidate))
    baseline_eval = baseline_payload.get("evaluation", {})
    candidate_eval = candidate_payload.get("evaluation", {})
    baseline_metrics = baseline_eval.get("metrics", {})
    candidate_metrics = candidate_eval.get("metrics", {})

    metric_keys = sorted(set(baseline_metrics) | set(candidate_metrics))
    metric_deltas = {
        key: _stable_float_delta(baseline_metrics.get(key, 0.0), candidate_metrics.get(key, 0.0))
        for key in metric_keys
        if isinstance(baseline_metrics.get(key, 0.0), (int, float)) or isinstance(candidate_metrics.get(key, 0.0), (int, float))
    }

    baseline_layers = baseline_payload.get("per_layer", {})
    candidate_layers = candidate_payload.get("per_layer", {})
    baseline_modules = baseline_payload.get("per_module", {})
    candidate_modules = candidate_payload.get("per_module", {})

    baseline_failures = baseline_payload.get("failure_summary", {}).get("failure_counts", {})
    candidate_failures = candidate_payload.get("failure_summary", {}).get("failure_counts", {})
    failure_keys = sorted(set(baseline_failures) | set(candidate_failures))
    failure_deltas = {key: int(candidate_failures.get(key, 0)) - int(baseline_failures.get(key, 0)) for key in failure_keys}

    result = EvalDiffResult(
        baseline_fingerprint=str(baseline_payload.get("report_fingerprint", baseline_eval.get("fingerprint", ""))),
        candidate_fingerprint=str(candidate_payload.get("report_fingerprint", candidate_eval.get("fingerprint", ""))),
        baseline_checkpoint_version=str(baseline_payload.get("checkpoint", {}).get("schema_version", "")),
        candidate_checkpoint_version=str(candidate_payload.get("checkpoint", {}).get("schema_version", "")),
        same_fingerprint=str(baseline_payload.get("report_fingerprint", "")) == str(candidate_payload.get("report_fingerprint", "")),
        metric_deltas=dict(sorted(metric_deltas.items())),
        layer_deltas=_group_delta(baseline_layers, candidate_layers),
        module_deltas=_group_delta(baseline_modules, candidate_modules),
        failure_deltas=dict(sorted(failure_deltas.items())),
        invariant_retention_delta=_stable_float_delta(
            baseline_metrics.get("invariant_retention_rate", 0.0),
            candidate_metrics.get("invariant_retention_rate", 0.0),
        ),
        warnings=[],
    )
    if result.baseline_checkpoint_version != result.candidate_checkpoint_version:
        result.warnings.append("checkpoint_version_changed")
    return result
