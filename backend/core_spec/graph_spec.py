"""CPT Core Specification — Canonical Circuit Graph Contract.

Defines the stable, serializable, deterministic graph format that ALL CPT
domains (circuits, KiCad, FreeCAD, mathematics, logic) must conform to.
This contract is FROZEN — changes require a schema version bump.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Topology family enum — canonical, ordered, deterministic
# ---------------------------------------------------------------------------

class TopologyFamily(str, Enum):
    RADIAL = "radial"
    MESH = "mesh"
    BRIDGE = "bridge"
    CURRENT_SOURCE = "current_source"
    MIXED = "mixed"
    UNKNOWN = "unknown"

    @classmethod
    def classify(cls, cycle_count: int, connected_components: int, source_nodes: list[int]) -> "TopologyFamily":
        """Heuristic topology classification from graph statistics."""
        if connected_components > 1:
            return cls.UNKNOWN
        if cycle_count == 0:
            return cls.RADIAL
        if cycle_count == 1:
            return cls.BRIDGE
        if cycle_count >= 2 and any(s >= 0 for s in source_nodes):
            return cls.CURRENT_SOURCE
        if cycle_count >= 2:
            return cls.MESH
        return cls.MIXED


# ---------------------------------------------------------------------------
# Canonical Circuit Graph — frozen, serializable, validated
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalCircuitGraph:
    """Stable graph contract for the CPT ecosystem.

    Every field is immutable. Tensors are included for computation but
    serialized as lists. The fingerprint is a deterministic SHA-256 over
    canonical JSON (sorted keys, no tensor references).
    """

    graph_id: str
    fingerprint: str
    num_nodes: int
    num_edges: int
    node_features: torch.Tensor        # (num_nodes, NODE_DIM)
    edge_index: torch.Tensor           # (2, num_edges)
    edge_features: torch.Tensor        # (num_edges, EDGE_DIM)
    topology_family: TopologyFamily
    cycle_count: int
    connected_components: int
    source_nodes: list[int]
    ground_node: int
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- Serialization -------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:
        """Deterministic JSON-serializable dict (tensors as lists, sorted keys)."""
        return {
            "graph_id": self.graph_id,
            "fingerprint": self.fingerprint,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "node_features": self.node_features.tolist(),
            "edge_index": self.edge_index.tolist(),
            "edge_features": self.edge_features.tolist(),
            "topology_family": self.topology_family.value,
            "cycle_count": self.cycle_count,
            "connected_components": self.connected_components,
            "source_nodes": sorted(self.source_nodes),
            "ground_node": self.ground_node,
            "metadata": _normalize_dict(self.metadata),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "CanonicalCircuitGraph":
        """Reconstruct from JSON dict (lists back to tensors)."""
        return cls(
            graph_id=data["graph_id"],
            fingerprint=data["fingerprint"],
            num_nodes=data["num_nodes"],
            num_edges=data["num_edges"],
            node_features=torch.tensor(data["node_features"], dtype=torch.float32),
            edge_index=torch.tensor(data["edge_index"], dtype=torch.long),
            edge_features=torch.tensor(data["edge_features"], dtype=torch.float32),
            topology_family=TopologyFamily(data["topology_family"]),
            cycle_count=data["cycle_count"],
            connected_components=data["connected_components"],
            source_nodes=data["source_nodes"],
            ground_node=data["ground_node"],
            metadata=data.get("metadata", {}),
        )

    # -- Validation ----------------------------------------------------------

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors: list[str] = []
        if self.num_nodes <= 0:
            errors.append("num_nodes must be positive")
        if self.num_edges < 0:
            errors.append("num_edges must be non-negative")
        if self.node_features.shape[0] != self.num_nodes:
            errors.append(f"node_features rows {self.node_features.shape[0]} != num_nodes {self.num_nodes}")
        if self.edge_index.shape[1] != self.num_edges:
            errors.append(f"edge_index cols {self.edge_index.shape[1]} != num_edges {self.num_edges}")
        if self.edge_features.shape[0] != self.num_edges:
            errors.append(f"edge_features rows {self.edge_features.shape[0]} != num_edges {self.num_edges}")
        if self.ground_node < 0 or self.ground_node >= self.num_nodes:
            errors.append(f"ground_node {self.ground_node} out of range [0, {self.num_nodes})")
        expected_fp = compute_graph_fingerprint(self)
        if self.fingerprint != expected_fp:
            errors.append(f"fingerprint mismatch: stored={self.fingerprint[:16]} computed={expected_fp[:16]}")
        return errors


# ---------------------------------------------------------------------------
# Fingerprint — deterministic SHA-256 over canonical JSON
# ---------------------------------------------------------------------------

def compute_graph_fingerprint(graph: CanonicalCircuitGraph) -> str:
    """Deterministic SHA-256 fingerprint over graph structure (no tensor data)."""
    payload = json.dumps({
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "edge_index": graph.edge_index.tolist(),
        "topology_family": graph.topology_family.value,
        "cycle_count": graph.cycle_count,
        "connected_components": graph.connected_components,
        "source_nodes": sorted(graph.source_nodes),
        "ground_node": graph.ground_node,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def compute_graph_fingerprint_from_dict(data: dict[str, Any]) -> str:
    """Fingerprint from raw dict (for pre-construction validation)."""
    payload = json.dumps({
        "num_nodes": data["num_nodes"],
        "num_edges": data["num_edges"],
        "edge_index": data["edge_index"],
        "topology_family": data["topology_family"],
        "cycle_count": data["cycle_count"],
        "connected_components": data["connected_components"],
        "source_nodes": sorted(data["source_nodes"]),
        "ground_node": data["ground_node"],
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Conversion helpers — existing CircuitGraph → CanonicalCircuitGraph
# ---------------------------------------------------------------------------

def from_circuit_graph(
    cg: Any,  # backend.circuits.graph_dataset.CircuitGraph (avoid circular import)
    graph_id: str = "",
    topology_family: TopologyFamily | None = None,
    cycle_count: int = 0,
    connected_components: int = 1,
    source_nodes: list[int] | None = None,
    ground_node: int = 0,
    metadata: dict[str, Any] | None = None,
) -> CanonicalCircuitGraph:
    """Convert existing CircuitGraph to canonical contract.

    Does NOT break existing code — pure conversion layer.
    """
    src_nodes = source_nodes or []
    meta = metadata or {}

    num_nodes = cg.node_features.shape[0]
    num_edges = cg.edge_index.shape[1]
    family = topology_family or TopologyFamily.classify(cycle_count, connected_components, src_nodes)

    canonical = CanonicalCircuitGraph(
        graph_id=graph_id or cg.fingerprint[:16],
        fingerprint="",  # placeholder, computed below
        num_nodes=num_nodes,
        num_edges=num_edges,
        node_features=cg.node_features,
        edge_index=cg.edge_index,
        edge_features=cg.edge_features,
        topology_family=family,
        cycle_count=cycle_count,
        connected_components=connected_components,
        source_nodes=src_nodes,
        ground_node=ground_node,
        metadata={**meta, "source_fingerprint": cg.fingerprint},
    )
    # Compute and set fingerprint
    object.__setattr__(canonical, "fingerprint", compute_graph_fingerprint(canonical))
    return canonical


def validate_graph(graph: CanonicalCircuitGraph) -> CanonicalCircuitGraph:
    """Validate and return. Raises ValueError if invalid."""
    errors = graph.validate()
    if errors:
        raise ValueError(f"Invalid CanonicalCircuitGraph: {errors}")
    return graph


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Sort dict keys for deterministic serialization."""
    result: dict[str, Any] = {}
    for k in sorted(d):
        v = d[k]
        if isinstance(v, dict):
            result[k] = _normalize_dict(v)
        elif isinstance(v, list):
            result[k] = sorted(v) if all(isinstance(x, str) for x in v) else v
        else:
            result[k] = v
    return result
