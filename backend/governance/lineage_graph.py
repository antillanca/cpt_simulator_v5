"""Deterministic lineage graph generation for CPT artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.governance.artifact_inventory import InventoryEntry, InventoryIndex


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class LineageNode:
    artifact_id: str
    artifact_type: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class LineageEdge:
    source_id: str
    target_id: str
    relationship: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship,
        }


@dataclass(frozen=True)
class LineageGraph:
    nodes: tuple[LineageNode, ...]
    edges: tuple[LineageEdge, ...]
    adjacency: dict[str, tuple[str, ...]] = field(default_factory=dict)
    graph_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "adjacency": {key: list(value) for key, value in sorted(self.adjacency.items())},
        }
        payload["graph_fingerprint"] = self.graph_fingerprint or _stable_hash(payload)
        return payload

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        payload = self.to_dict()
        lines = [
            "# CPT Artifact Lineage Graph",
            "",
            "## Summary",
            f"- Nodes: {len(payload['nodes'])}",
            f"- Edges: {len(payload['edges'])}",
            f"- Fingerprint: {payload['graph_fingerprint']}",
            "",
            "## Edges",
            "",
            "| Source | Target | Relationship |",
            "|--------|--------|--------------|",
        ]
        for edge in payload["edges"]:
            lines.append(f"| {edge['source_id']} | {edge['target_id']} | {edge['relationship']} |")
        return "\n".join(lines)


def _relationship_for_child(entry: InventoryEntry) -> str:
    if entry.artifact_type == "evaluation_report":
        return "evaluated_by"
    if entry.artifact_type == "archive_bundle":
        return "archived_into"
    if entry.artifact_type == "manifest":
        return "derived_from"
    if entry.artifact_type == "benchmark_snapshot":
        return "benchmarked_against"
    if entry.artifact_type == "training_snapshot":
        return "trained_from"
    if entry.artifact_type == "workspace_summary":
        return "derived_from"
    if entry.artifact_type == "inventory_index":
        return "derived_from"
    if entry.artifact_type == "retention_report":
        return "derived_from"
    return "derived_from"


def build_lineage_graph(index: InventoryIndex) -> LineageGraph:
    nodes = tuple(sorted((LineageNode(entry.artifact_id, entry.artifact_type, entry.fingerprint) for entry in index.entries), key=lambda node: (node.artifact_type, node.artifact_id, node.fingerprint)))
    node_ids = {node.artifact_id for node in nodes}
    edges: list[LineageEdge] = []
    adjacency: dict[str, list[str]] = {}
    entry_by_id = {entry.artifact_id: entry for entry in index.entries}
    for entry in sorted(index.entries, key=lambda item: (item.artifact_type, item.relative_path, item.fingerprint)):
        relationship = _relationship_for_child(entry)
        for parent_id in entry.lineage_parents:
            edges.append(LineageEdge(source_id=entry.artifact_id, target_id=parent_id, relationship=relationship))
            adjacency.setdefault(entry.artifact_id, []).append(parent_id)
    edges = sorted(edges, key=lambda edge: (edge.source_id, edge.target_id, edge.relationship))
    adjacency_sorted = {key: tuple(sorted(values)) for key, values in sorted(adjacency.items())}
    payload = {
        "nodes": [node.to_dict() for node in nodes],
        "edges": [edge.to_dict() for edge in edges],
        "adjacency": {key: list(value) for key, value in adjacency_sorted.items()},
    }
    graph_fingerprint = _stable_hash(payload)
    return LineageGraph(nodes=nodes, edges=tuple(edges), adjacency=adjacency_sorted, graph_fingerprint=graph_fingerprint)


def save_lineage_graph(graph: LineageGraph, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(graph.to_json(), encoding="utf-8")
