"""Deterministic reverse dependency traversal for artifact inventories."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from backend.governance.artifact_inventory import InventoryEntry, InventoryIndex


@dataclass(frozen=True)
class DependencyEdge:
    source_id: str
    target_id: str
    relationship: str


@dataclass(frozen=True)
class ReverseDependencyResult:
    root_artifact: str
    dependent_count: int
    impacted_artifacts: tuple[str, ...]
    dependency_depth: int

    def to_dict(self) -> dict[str, object]:
        return {
            "root_artifact": self.root_artifact,
            "dependent_count": self.dependent_count,
            "impacted_artifacts": list(self.impacted_artifacts),
            "dependency_depth": self.dependency_depth,
        }


def _relationship_for(entry: InventoryEntry) -> str:
    if entry.artifact_type == "checkpoint":
        return "derived_from"
    if entry.artifact_type == "evaluation_report":
        return "evaluated_by"
    if entry.artifact_type == "archive_bundle":
        return "archived_into"
    if entry.artifact_type == "manifest":
        return "derived_from"
    if entry.artifact_type == "benchmark_snapshot":
        return "benchmarked_against"
    if entry.artifact_type == "workspace_summary":
        return "derived_from"
    if entry.artifact_type == "inventory_index":
        return "derived_from"
    if entry.artifact_type == "retention_report":
        return "derived_from"
    return "derived_from"


def build_reverse_dependency_index(entries: Iterable[InventoryEntry]) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[DependencyEdge, ...]]]:
    reverse_map: dict[str, set[str]] = defaultdict(set)
    edge_map: dict[str, list[DependencyEdge]] = defaultdict(list)
    for entry in sorted(entries, key=lambda item: (item.artifact_type, item.relative_path, item.fingerprint)):
        for parent_id in entry.lineage_parents:
            reverse_map[parent_id].add(entry.artifact_id)
            edge_map[parent_id].append(
                DependencyEdge(
                    source_id=entry.artifact_id,
                    target_id=parent_id,
                    relationship=_relationship_for(entry),
                )
            )
    return (
        {key: tuple(sorted(value)) for key, value in sorted(reverse_map.items())},
        {key: tuple(sorted(value, key=lambda edge: (edge.source_id, edge.target_id, edge.relationship))) for key, value in sorted(edge_map.items())},
    )


def find_reverse_dependencies(artifact_id: str, reverse_index) -> ReverseDependencyResult:
    reverse_map = reverse_index[0] if isinstance(reverse_index, tuple) and len(reverse_index) == 2 else reverse_index
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(artifact_id, 0)])
    max_depth = 0
    while queue:
        current, depth = queue.popleft()
        max_depth = max(max_depth, depth)
        for dependent in reverse_map.get(current, ()):
            if dependent in visited:
                continue
            visited.add(dependent)
            queue.append((dependent, depth + 1))
    return ReverseDependencyResult(
        root_artifact=artifact_id,
        dependent_count=len(visited),
        impacted_artifacts=tuple(sorted(visited)),
        dependency_depth=max_depth,
    )


def reverse_dependency_closure(artifact_id: str, reverse_index) -> tuple[str, ...]:
    return find_reverse_dependencies(artifact_id, reverse_index).impacted_artifacts

