"""Lightweight drift detection for inventory indices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.governance.artifact_inventory import InventoryIndex, build_inventory_index
from backend.governance.lineage_graph import build_lineage_graph


def detect_inventory_drift(index: InventoryIndex, workspace_root: Path) -> list[str]:
    workspace_root = Path(workspace_root).resolve()
    current_index = build_inventory_index(workspace_root)
    current_by_path = {entry.relative_path: entry for entry in current_index.entries}
    indexed_by_path = {entry.relative_path: entry for entry in index.entries}
    issues: list[str] = []

    for relative_path in sorted(indexed_by_path):
        if relative_path not in current_by_path:
            issues.append(f"missing_artifact:{relative_path}")
            continue
        if current_by_path[relative_path].fingerprint != indexed_by_path[relative_path].fingerprint:
            issues.append(f"changed_fingerprint:{relative_path}")

    for relative_path in sorted(current_by_path):
        if relative_path not in indexed_by_path:
            issues.append(f"unindexed_artifact:{relative_path}")

    indexed_ids = {entry.artifact_id for entry in index.entries}
    for entry in index.entries:
        for parent_id in entry.lineage_parents:
            if parent_id.startswith("ref:"):
                continue
            if parent_id not in indexed_ids:
                issues.append(f"broken_lineage:{entry.artifact_id}->{parent_id}")

    graph = build_lineage_graph(index)
    if graph.graph_fingerprint != build_lineage_graph(current_index).graph_fingerprint:
        issues.append("inventory_graph_changed")

    if current_index.inventory_fingerprint != index.inventory_fingerprint:
        issues.append("stale_inventory_index")

    return sorted(dict.fromkeys(issues))
