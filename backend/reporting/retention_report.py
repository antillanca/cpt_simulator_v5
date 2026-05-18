"""Retention reporting for artifact lifecycle governance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.governance.artifact_policy import ArtifactPolicy, artifact_policy_fingerprint
from backend.governance.retention_sweeper import SweepCandidate, build_retention_plan, scan_retention_candidates


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _human_bytes(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} B"


@dataclass
class RetentionReport:
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
        lines = [
            "# CPT Retention Report",
            "",
            "## Summary",
            f"- Total Artifacts: {summary.get('total_artifacts', 0)}",
            f"- Storage Usage: {summary.get('storage_usage_human', '0 B')}",
            f"- Reclaimable: {summary.get('reclaimable_storage_human', '0 B')}",
            f"- Pinned: {summary.get('pinned_artifacts', 0)}",
            f"- Archive Candidates: {summary.get('archive_candidates', 0)}",
            "",
            "## By Artifact Type",
            "",
            "| Type | Count | Size | Reclaimable |",
            "|------|------:|------:|------------:|",
        ]
        for artifact_type, metrics in sorted(report.get("by_type", {}).items(), key=lambda item: str(item[0])):
            lines.append(
                f"| {artifact_type} | {metrics.get('count', 0)} | {metrics.get('size_human', '0 B')} | {metrics.get('reclaimable_human', '0 B')} |"
            )
        return "\n".join(lines)


def build_retention_report(root: Path, policy: ArtifactPolicy) -> RetentionReport:
    candidates = scan_retention_candidates(root, policy)
    plan = build_retention_plan(candidates, policy)
    by_type: dict[str, dict[str, Any]] = {}
    total_bytes = sum(candidate.size_bytes for candidate in candidates)
    reclaimable_bytes = sum(candidate.size_bytes for candidate in plan if candidate.retention_reason not in {None, "retained_by_policy", "retained_pinned", "retained_failed_run", "retained_age_window"})
    pinned = sum(1 for candidate in candidates if candidate.pinned)
    archive_candidates = sum(1 for candidate in plan if candidate.retention_reason == "archive_before_delete")
    for candidate in candidates:
        bucket = by_type.setdefault(candidate.artifact_type, {"count": 0, "size_bytes": 0, "reclaimable_bytes": 0})
        bucket["count"] += 1
        bucket["size_bytes"] += int(candidate.size_bytes)
    for candidate in plan:
        if candidate.retention_reason not in {None, "retained_by_policy", "retained_pinned", "retained_failed_run", "retained_age_window"}:
            bucket = by_type.setdefault(candidate.artifact_type, {"count": 0, "size_bytes": 0, "reclaimable_bytes": 0})
            bucket["reclaimable_bytes"] += int(candidate.size_bytes)
    for artifact_type in sorted(by_type):
        bucket = by_type[artifact_type]
        bucket["size_human"] = _human_bytes(bucket["size_bytes"])
        bucket["reclaimable_human"] = _human_bytes(bucket["reclaimable_bytes"])
    payload = {
        "root": str(root),
        "policy_schema_version": policy.schema_version,
        "policy_fingerprint": artifact_policy_fingerprint(policy),
        "summary": {
            "total_artifacts": len(candidates),
            "storage_usage_bytes": total_bytes,
            "storage_usage_human": _human_bytes(total_bytes),
            "reclaimable_storage_bytes": reclaimable_bytes,
            "reclaimable_storage_human": _human_bytes(reclaimable_bytes),
            "pinned_artifacts": pinned,
            "archive_candidates": archive_candidates,
        },
        "by_type": by_type,
        "violations": [candidate.to_dict() for candidate in plan if candidate.retention_reason == "unknown_artifact_type"],
    }
    return RetentionReport(payload)
