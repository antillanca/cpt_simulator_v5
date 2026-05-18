"""Stable failure taxonomy summaries for evaluation reports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _coerce_result(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    return dict(result)


def _dominant_failure(counts: dict[str, int]) -> str:
    failure_counts = {key: count for key, count in counts.items() if key != "pass"}
    if not failure_counts:
        return "none"
    return sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _top_failures(counts: dict[str, int], limit: int = 5) -> list[dict[str, Any]]:
    ordered = sorted(((failure_type, count) for failure_type, count in counts.items() if failure_type != "pass"), key=lambda item: (-item[1], item[0]))
    return [{"failure_type": failure_type, "count": count} for failure_type, count in ordered[:limit]]


def _failure_counts(results: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        failure_type = str(result.get("failure_type") or "pass")
        counts[failure_type] = counts.get(failure_type, 0) + 1
    return counts


def _group_results(results: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        key = str(key_fn(result))
        groups.setdefault(key, []).append(result)
    summary: dict[str, dict[str, Any]] = {}
    for key in sorted(groups):
        bucket = groups[key]
        counts = _failure_counts(bucket)
        total = float(len(bucket)) or 1.0
        summary[key] = {
            "sample_count": float(len(bucket)),
            "failure_counts": dict(sorted(counts.items())),
            "dominant_failure": _dominant_failure(counts),
            "invariant_violations": sum(1 for item in bucket if item.get("invariant_violation")),
            "replay_instability": sum(1 for item in bucket if not item.get("replay_consistency", True)),
            "compositional_failure": sum(1 for item in bucket if item.get("failure_type") == "compositional_failure"),
            "exact_match_rate": sum(1 for item in bucket if item.get("exact_match")) / total,
        }
    return summary


@dataclass
class FailureSummary:
    total_samples: int
    failure_counts: dict[str, int] = field(default_factory=dict)
    dominant_failure: str = "none"
    top_failures: list[dict[str, Any]] = field(default_factory=list)
    by_layer: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_module: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_ood: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_replay: dict[str, dict[str, Any]] = field(default_factory=dict)
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "total_samples": self.total_samples,
            "failure_counts": dict(sorted(self.failure_counts.items())),
            "dominant_failure": self.dominant_failure,
            "top_failures": list(self.top_failures),
            "by_layer": {key: _normalize(value) for key, value in sorted(self.by_layer.items())},
            "by_module": {key: _normalize(value) for key, value in sorted(self.by_module.items())},
            "by_ood": {key: _normalize(value) for key, value in sorted(self.by_ood.items())},
            "by_replay": {key: _normalize(value) for key, value in sorted(self.by_replay.items())},
        }
        payload["fingerprint"] = _stable_hash(payload)
        self.fingerprint = payload["fingerprint"]
        return payload


def summarize_failures(results: list[Any]) -> FailureSummary:
    records = [_coerce_result(result) for result in results]
    counts = _failure_counts(records)
    by_layer = _group_results(records, lambda r: r.get("curriculum_layer", -1))
    by_module = _group_results(records, lambda r: r.get("module_source", "unknown"))
    by_ood = _group_results(records, lambda r: "ood" if r.get("is_ood") else "in_distribution")
    by_replay = _group_results(records, lambda r: "stable" if r.get("replay_consistency", False) else "unstable")
    summary = FailureSummary(
        total_samples=len(records),
        failure_counts=dict(sorted(counts.items())),
        dominant_failure=_dominant_failure(counts),
        top_failures=_top_failures(counts),
        by_layer=by_layer,
        by_module=by_module,
        by_ood=by_ood,
        by_replay=by_replay,
    )
    summary.to_dict()
    return summary


def detect_failure_trends(baseline: FailureSummary | dict[str, Any], candidate: FailureSummary | dict[str, Any]) -> dict[str, Any]:
    baseline_payload = baseline.to_dict() if hasattr(baseline, "to_dict") else dict(baseline)
    candidate_payload = candidate.to_dict() if hasattr(candidate, "to_dict") else dict(candidate)
    baseline_counts = baseline_payload.get("failure_counts", {})
    candidate_counts = candidate_payload.get("failure_counts", {})
    keys = sorted(set(baseline_counts) | set(candidate_counts))
    deltas = {key: int(candidate_counts.get(key, 0)) - int(baseline_counts.get(key, 0)) for key in keys}
    trend = "stable"
    if any(delta > 0 for delta in deltas.values()):
        trend = "regression"
    elif any(delta < 0 for delta in deltas.values()):
        trend = "improvement"
    return {
        "trend": trend,
        "failure_deltas": dict(sorted(deltas.items())),
        "baseline_dominant_failure": baseline_payload.get("dominant_failure", "none"),
        "candidate_dominant_failure": candidate_payload.get("dominant_failure", "none"),
    }
