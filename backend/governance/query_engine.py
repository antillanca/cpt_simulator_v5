"""Deterministic query helpers for artifact inventory indices."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.governance.artifact_inventory import InventoryEntry, InventoryIndex


@dataclass(frozen=True)
class QueryResult:
    entries: tuple[InventoryEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [entry.to_dict() for entry in self.entries], "entry_count": len(self.entries)}


def _within_range(value: float, start: float | None, end: float | None) -> bool:
    if start is not None and value < start:
        return False
    if end is not None and value > end:
        return False
    return True


def _lineage_depth_map(index: InventoryIndex) -> dict[str, int]:
    parent_map = {entry.artifact_id: tuple(entry.lineage_parents) for entry in index.entries}
    cache: dict[str, int] = {}

    def depth(artifact_id: str, seen: set[str] | None = None) -> int:
        if artifact_id in cache:
            return cache[artifact_id]
        seen = set(seen or ())
        if artifact_id in seen:
            return 0
        seen.add(artifact_id)
        parents = parent_map.get(artifact_id, ())
        if not parents:
            cache[artifact_id] = 0
            return 0
        result = 1 + max((depth(parent_id, seen) for parent_id in parents if parent_id in parent_map), default=0)
        cache[artifact_id] = result
        return result

    return {entry.artifact_id: depth(entry.artifact_id) for entry in index.entries}


def query_inventory(
    index: InventoryIndex,
    *,
    artifact_type: str | None = None,
    schema_version: str | None = None,
    fingerprint: str | None = None,
    tag: str | None = None,
    parent_id: str | None = None,
    retention_status: str | None = None,
    workspace_root: str | Path | None = None,
    relative_prefix: str | None = None,
    pinned: bool | None = None,
    lineage_depth: int | None = None,
    archive_status: str | None = None,
    created_after: float | None = None,
    created_before: float | None = None,
) -> list[InventoryEntry]:
    workspace_root_text = str(workspace_root) if workspace_root is not None else None
    depth_map = _lineage_depth_map(index) if lineage_depth is not None else {}
    entries = []
    for entry in index.entries:
        if artifact_type is not None and entry.artifact_type != artifact_type:
            continue
        if schema_version is not None and entry.schema_version != schema_version:
            continue
        if fingerprint is not None and entry.fingerprint != fingerprint:
            continue
        if tag is not None and tag not in entry.tags:
            continue
        if parent_id is not None and parent_id not in entry.lineage_parents:
            continue
        if retention_status is not None and entry.retention_status != retention_status:
            continue
        if workspace_root_text is not None and entry.workspace_root != workspace_root_text:
            continue
        if relative_prefix is not None and not entry.relative_path.startswith(relative_prefix):
            continue
        if pinned is not None and pinned != (entry.retention_status == "pinned" or "pinned" in entry.tags):
            continue
        if lineage_depth is not None and depth_map.get(entry.artifact_id, 0) != int(lineage_depth):
            continue
        if archive_status is not None:
            actual_status = "archived" if entry.artifact_type == "archive_bundle" or entry.retention_status == "archived" else "active"
            if actual_status != archive_status:
                continue
        if not _within_range(entry.created_at, created_after, created_before):
            continue
        entries.append(entry)
    return sorted(entries, key=lambda item: (item.artifact_type, item.relative_path, item.fingerprint))
