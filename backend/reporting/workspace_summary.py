"""Deterministic workspace summary generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from backend.governance.artifact_inventory import InventoryIndex, _human_bytes, build_inventory_index
from backend.governance.lineage_graph import build_lineage_graph


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


@dataclass
class WorkspaceSummary:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        normalized = _normalize(self.payload)
        normalized["summary_fingerprint"] = _stable_hash(normalized)
        return normalized

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        report = self.to_dict()
        summary = report["summary"]
        lines = [
            "# CPT Workspace Summary",
            "",
            "## Overview",
            f"- Workspace Root: {summary.get('workspace_root', '.')}",
            f"- Total Artifacts: {summary.get('total_artifacts', 0)}",
            f"- Total Storage: {summary.get('total_storage_human', '0 B')}",
            f"- Reclaimable: {summary.get('reclaimable_storage_human', '0 B')}",
            "",
            "## Artifact Distribution",
            "",
            "| Type | Count | Size |",
            "|------|------:|------:|",
        ]
        for artifact_type, metrics in sorted(report.get("by_type", {}).items(), key=lambda item: str(item[0])):
            lines.append(f"| {artifact_type} | {metrics.get('count', 0)} | {metrics.get('size_human', '0 B')} |")
        lines.extend(
            [
                "",
                "## Schema Distribution",
                "",
                "| Schema | Count |",
                "|--------|------:|",
            ]
        )
        for schema_version, count in sorted(report.get("schema_distribution", {}).items(), key=lambda item: str(item[0])):
            lines.append(f"| {schema_version} | {count} |")
        return "\n".join(lines)


def _lineage_depths(index: InventoryIndex) -> list[int]:
    parents = {entry.artifact_id: tuple(entry.lineage_parents) for entry in index.entries}

    def depth(artifact_id: str, seen: set[str] | None = None) -> int:
        seen = set(seen or ())
        if artifact_id in seen:
            return 0
        seen.add(artifact_id)
        parent_ids = parents.get(artifact_id, ())
        if not parent_ids:
            return 0
        return 1 + max((depth(parent_id, seen) for parent_id in parent_ids if parent_id in parents), default=0)

    return [depth(entry.artifact_id) for entry in index.entries]


def build_workspace_summary(root: Path, *, policy=None, index: InventoryIndex | None = None) -> WorkspaceSummary:
    workspace_index = index or build_inventory_index(root, policy=policy)
    graph = build_lineage_graph(workspace_index)
    by_type: dict[str, dict[str, Any]] = {}
    schema_distribution: dict[str, int] = {}
    retention_distribution: dict[str, int] = {}
    archive_count = 0
    pinned_count = 0
    total_storage = 0
    reclaimable_storage = 0
    for entry in workspace_index.entries:
        total_storage += int(entry.size_bytes)
        schema_distribution[entry.schema_version] = schema_distribution.get(entry.schema_version, 0) + 1
        retention_distribution[entry.retention_status] = retention_distribution.get(entry.retention_status, 0) + 1
        bucket = by_type.setdefault(entry.artifact_type, {"count": 0, "size_bytes": 0, "reclaimable_bytes": 0})
        bucket["count"] += 1
        bucket["size_bytes"] += int(entry.size_bytes)
        if entry.retention_status not in {"active", "pinned", "retained_by_policy"}:
            bucket["reclaimable_bytes"] += int(entry.size_bytes)
            reclaimable_storage += int(entry.size_bytes)
        if entry.retention_status == "pinned" or "pinned" in entry.tags:
            pinned_count += 1
        if entry.artifact_type == "archive_bundle":
            archive_count += 1
    lineage_depths = _lineage_depths(workspace_index)
    for artifact_type in sorted(by_type):
        bucket = by_type[artifact_type]
        bucket["size_human"] = _human_bytes(bucket["size_bytes"])
        bucket["reclaimable_human"] = _human_bytes(bucket["reclaimable_bytes"])
    payload = {
        "workspace_root": workspace_index.workspace_root,
        "index_fingerprint": workspace_index.inventory_fingerprint,
        "policy_fingerprint": workspace_index.policy_fingerprint,
        "summary": {
            "workspace_root": workspace_index.workspace_root,
            "total_artifacts": workspace_index.entry_count,
            "total_storage_bytes": total_storage,
            "total_storage_human": _human_bytes(total_storage),
            "reclaimable_storage_bytes": reclaimable_storage,
            "reclaimable_storage_human": _human_bytes(reclaimable_storage),
            "pinned_artifacts": pinned_count,
            "archive_coverage": 0.0 if workspace_index.entry_count == 0 else archive_count / float(workspace_index.entry_count),
            "lineage_depth_statistics": {
                "max_depth": max(lineage_depths, default=0),
                "average_depth": mean(lineage_depths) if lineage_depths else 0.0,
            },
        },
        "by_type": by_type,
        "schema_distribution": dict(sorted(schema_distribution.items())),
        "retention_distribution": dict(sorted(retention_distribution.items())),
        "graph_fingerprint": graph.graph_fingerprint,
    }
    return WorkspaceSummary(payload)

