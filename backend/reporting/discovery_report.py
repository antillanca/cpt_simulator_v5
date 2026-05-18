"""Artifact discovery reports for operational navigation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from backend.governance.artifact_inventory import InventoryIndex
from backend.governance.reverse_dependencies import build_reverse_dependency_index, find_reverse_dependencies
from backend.reporting.search_facets import build_search_facets


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
class DiscoveryReport:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        normalized = _normalize(self.payload)
        normalized["discovery_fingerprint"] = _stable_hash(normalized)
        return normalized

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        report = self.to_dict()
        summary = report["summary"]
        lines = [
            "# CPT Artifact Discovery Report",
            "",
            "## Overview",
            f"- Total Artifacts: {summary.get('total_artifacts', 0)}",
            f"- Root Artifacts: {summary.get('root_artifacts', 0)}",
            f"- Orphan Artifacts: {summary.get('orphan_artifacts', 0)}",
            f"- Deepest Lineage Depth: {summary.get('deepest_lineage_depth', 0)}",
            "",
            "## Largest Dependency Hubs",
            "",
            "| Artifact | Dependents |",
            "|----------|-------------:|",
        ]
        for item in report.get("dependency_hubs", []):
            lines.append(f"| {item['artifact_id']} | {item['dependent_count']} |")
        lines.extend(
            [
                "",
                "## Orphan Artifacts",
                "",
                "| Artifact | Type |",
                "|----------|------|",
            ]
        )
        for item in report.get("orphans", []):
            lines.append(f"| {item['artifact_id']} | {item['artifact_type']} |")
        return "\n".join(lines)


def build_discovery_report(index: InventoryIndex) -> DiscoveryReport:
    reverse_index, _edges = build_reverse_dependency_index(index.entries)
    depth_map = {entry.artifact_id: 0 for entry in index.entries}
    parent_map = {entry.artifact_id: tuple(entry.lineage_parents) for entry in index.entries}

    def depth(artifact_id: str, seen: set[str] | None = None) -> int:
        seen = set(seen or ())
        if artifact_id in seen:
            return 0
        seen.add(artifact_id)
        parents = parent_map.get(artifact_id, ())
        if not parents:
            return 0
        return 1 + max((depth(parent_id, seen) for parent_id in parents if parent_id in parent_map), default=0)

    for entry in index.entries:
        depth_map[entry.artifact_id] = depth(entry.artifact_id)

    root_artifacts = [entry for entry in index.entries if not entry.lineage_parents]
    orphan_artifacts = [entry for entry in index.entries if not reverse_index.get(entry.artifact_id, ())]
    deepest = max(depth_map.values(), default=0)
    dependency_hubs = []
    for artifact_id, dependents in sorted(reverse_index.items(), key=lambda item: (-len(item[1]), item[0])):
        dependency_hubs.append({"artifact_id": artifact_id, "dependent_count": len(dependents)})
    payload = {
        "summary": {
            "total_artifacts": index.entry_count,
            "root_artifacts": len(root_artifacts),
            "orphan_artifacts": len(orphan_artifacts),
            "deepest_lineage_depth": deepest,
            "archive_coverage": build_search_facets(index).to_dict()["facets"]["archive_status"].get("archived", 0) / float(index.entry_count or 1),
        },
        "dependency_hubs": dependency_hubs[:10],
        "orphans": [{"artifact_id": entry.artifact_id, "artifact_type": entry.artifact_type} for entry in sorted(orphan_artifacts, key=lambda item: (item.artifact_type, item.relative_path, item.fingerprint))],
        "root_artifacts": [{"artifact_id": entry.artifact_id, "artifact_type": entry.artifact_type} for entry in sorted(root_artifacts, key=lambda item: (item.artifact_type, item.relative_path, item.fingerprint))],
        "facets": build_search_facets(index).to_dict()["facets"],
        "index_fingerprint": index.inventory_fingerprint,
    }
    return DiscoveryReport(payload)

