"""Aggregate facets for deterministic artifact discovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from backend.governance.artifact_inventory import InventoryIndex


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
class SearchFacets:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        normalized = _normalize(self.payload)
        normalized["facets_fingerprint"] = _stable_hash(normalized)
        return normalized

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        payload = self.to_dict()
        lines = [
            "# CPT Search Facets",
            "",
            "| Facet | Key | Count |",
            "|------|-----|------:|",
        ]
        for facet_name, values in sorted(payload.get("facets", {}).items(), key=lambda item: str(item[0])):
            for key, count in sorted(values.items(), key=lambda item: str(item[0])):
                lines.append(f"| {facet_name} | {key} | {count} |")
        return "\n".join(lines)


def _depth_map(index: InventoryIndex) -> dict[str, int]:
    parents = {entry.artifact_id: tuple(entry.lineage_parents) for entry in index.entries}
    cache: dict[str, int] = {}

    def depth(artifact_id: str, seen: set[str] | None = None) -> int:
        if artifact_id in cache:
            return cache[artifact_id]
        seen = set(seen or ())
        if artifact_id in seen:
            return 0
        seen.add(artifact_id)
        parent_ids = parents.get(artifact_id, ())
        if not parent_ids:
            cache[artifact_id] = 0
            return 0
        result = 1 + max((depth(parent_id, seen) for parent_id in parent_ids if parent_id in parents), default=0)
        cache[artifact_id] = result
        return result

    return {entry.artifact_id: depth(entry.artifact_id) for entry in index.entries}


def build_search_facets(index: InventoryIndex) -> SearchFacets:
    depth_map = _depth_map(index)
    facets = {
        "artifact_type": {},
        "schema_version": {},
        "workspace": {},
        "retention_status": {},
        "lineage_depth": {},
        "archive_status": {},
        "pinned_state": {},
    }
    for entry in index.entries:
        facets["artifact_type"][entry.artifact_type] = facets["artifact_type"].get(entry.artifact_type, 0) + 1
        facets["schema_version"][entry.schema_version] = facets["schema_version"].get(entry.schema_version, 0) + 1
        facets["workspace"][entry.workspace_root] = facets["workspace"].get(entry.workspace_root, 0) + 1
        facets["retention_status"][entry.retention_status] = facets["retention_status"].get(entry.retention_status, 0) + 1
        facets["lineage_depth"][str(depth_map.get(entry.artifact_id, 0))] = facets["lineage_depth"].get(str(depth_map.get(entry.artifact_id, 0)), 0) + 1
        archive_status = "archived" if entry.artifact_type == "archive_bundle" or entry.retention_status == "archived" else "active"
        facets["archive_status"][archive_status] = facets["archive_status"].get(archive_status, 0) + 1
        pinned_state = "pinned" if entry.retention_status == "pinned" or "pinned" in entry.tags else "unpinned"
        facets["pinned_state"][pinned_state] = facets["pinned_state"].get(pinned_state, 0) + 1
    return SearchFacets({"facets": facets, "index_fingerprint": index.inventory_fingerprint})

