"""CPT Core Specification — Memory / Replay Foundation Schema.

Defines the canonical memory entry schema for future FAISS/replay/LoRA
integration. ONLY the schema — no vector store, no retrieval logic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MemoryEntry:
    """Canonical memory record for replay-based learning.

    Captures the relationship between a graph, its projection effort,
    dominant failure, and oracle/projection timing. This is the schema
    that FAISS will index when implemented.
    """
    entry_id: str
    graph_fingerprint: str
    topology_family: str
    projection_iterations: int
    initial_residual: float
    final_residual: float
    dominant_failure: str | None
    oracle_time_ms: float
    projection_time_ms: float
    used_lora_expert: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "2.11"

    # -- Fingerprint ---------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_json_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    # -- Serialization -------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "graph_fingerprint": self.graph_fingerprint,
            "topology_family": self.topology_family,
            "projection_iterations": self.projection_iterations,
            "initial_residual": round(self.initial_residual, 12),
            "final_residual": round(self.final_residual, 12),
            "dominant_failure": self.dominant_failure,
            "oracle_time_ms": round(self.oracle_time_ms, 3),
            "projection_time_ms": round(self.projection_time_ms, 3),
            "used_lora_expert": self.used_lora_expert,
            "metadata": _sort_dict(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(
            entry_id=data["entry_id"],
            graph_fingerprint=data["graph_fingerprint"],
            topology_family=data["topology_family"],
            projection_iterations=data["projection_iterations"],
            initial_residual=data["initial_residual"],
            final_residual=data["final_residual"],
            dominant_failure=data.get("dominant_failure"),
            oracle_time_ms=data["oracle_time_ms"],
            projection_time_ms=data["projection_time_ms"],
            used_lora_expert=data.get("used_lora_expert"),
            metadata=data.get("metadata", {}),
            schema_version=data.get("schema_version", "2.11"),
        )

    # -- Validation ----------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.entry_id:
            errors.append("entry_id must not be empty")
        if not self.graph_fingerprint:
            errors.append("graph_fingerprint must not be empty")
        if self.projection_iterations < 0:
            errors.append("projection_iterations must be non-negative")
        if self.initial_residual < 0:
            errors.append("initial_residual must be non-negative")
        if self.final_residual < 0:
            errors.append("final_residual must be non-negative")
        if self.dominant_failure is not None and self.dominant_failure not in _VALID_FAILURES:
            errors.append(f"unknown dominant_failure: {self.dominant_failure}")
        return errors


# Back-reference to failure taxonomy (lazy to avoid circular import)
_VALID_FAILURES: set[str] = set()

def _ensure_failure_set() -> None:
    global _VALID_FAILURES
    if not _VALID_FAILURES:
        from backend.core_spec.failure_taxonomy import FAILURE_TYPES
        _VALID_FAILURES = set(FAILURE_TYPES)

# Eager load on module import
_ensure_failure_set()


def _sort_dict(d: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k in sorted(d):
        v = d[k]
        result[k] = _sort_dict(v) if isinstance(v, dict) else v
    return result
