"""CPT Runtime — Operational Experience Schema (v2.15 / v2.16 prep).

SCHEMA-ONLY. No replay learning. No training. No LoRA.
Defines the frozen schema for v2.16 consumption.

This schema captures the complete operational fingerprint of each execution,
sufficient for future accumulation in v2.16 (LoRA, replay, continual learning).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.runtime.projection_scheduler import VALID_TRAJECTORY_CLASSES
from backend.runtime.execution_scheduler import VALID_OUTCOMES, VALID_ROUTES


# ---------------------------------------------------------------------------
# OperationalExperienceEntry (v2.16 input schema)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperationalExperienceEntry:
    """Complete operational fingerprint of a single execution.

    This is the schema that v2.16 will consume for:
    - LoRA specialization per topology family
    - Replay learning from successful executions
    - Continual learning loop
    - Adaptive routing refinement

    DO NOT implement any of these yet. Only define the schema.
    """

    execution_id: str
    task_hash: str
    topology_family: str

    # Projection budget and usage
    projection_budget: int
    projection_iterations: int

    # Trajectory classification
    convergence_class: str  # From VALID_TRAJECTORY_CLASSES

    # Assist methods
    warmstart_used: bool
    retrieval_used: bool

    # Degradation
    degraded: bool

    # Latency breakdown (milliseconds)
    oracle_latency_ms: float
    surrogate_latency_ms: float
    projection_latency_ms: float

    # Metadata
    timestamp: str
    metadata: dict[str, Any]  # Frozen via __post_init__ copy

    def __post_init__(self) -> None:
        if self.convergence_class not in VALID_TRAJECTORY_CLASSES:
            raise ValueError(
                f"Invalid convergence_class: {self.convergence_class}, "
                f"expected one of {VALID_TRAJECTORY_CLASSES}"
            )
        # Deep-copy metadata to maintain immutability
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "task_hash": self.task_hash,
            "topology_family": self.topology_family,
            "projection_budget": self.projection_budget,
            "projection_iterations": self.projection_iterations,
            "convergence_class": self.convergence_class,
            "warmstart_used": self.warmstart_used,
            "retrieval_used": self.retrieval_used,
            "degraded": self.degraded,
            "oracle_latency_ms": round(self.oracle_latency_ms, 3),
            "surrogate_latency_ms": round(self.surrogate_latency_ms, 3),
            "projection_latency_ms": round(self.projection_latency_ms, 3),
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "OperationalExperienceEntry":
        return cls(
            execution_id=data["execution_id"],
            task_hash=data["task_hash"],
            topology_family=data["topology_family"],
            projection_budget=data["projection_budget"],
            projection_iterations=data["projection_iterations"],
            convergence_class=data["convergence_class"],
            warmstart_used=data.get("warmstart_used", False),
            retrieval_used=data.get("retrieval_used", False),
            degraded=data.get("degraded", False),
            oracle_latency_ms=data.get("oracle_latency_ms", 0.0),
            surrogate_latency_ms=data.get("surrogate_latency_ms", 0.0),
            projection_latency_ms=data.get("projection_latency_ms", 0.0),
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )

    @property
    def budget_efficiency(self) -> float:
        """Fraction of allocated budget actually used (0-1)."""
        if self.projection_budget <= 0:
            return 0.0
        return self.projection_iterations / self.projection_budget

    @property
    def iterations_saved(self) -> int:
        """Iterations saved vs allocated budget."""
        return max(0, self.projection_budget - self.projection_iterations)

    @property
    def total_latency_ms(self) -> float:
        """Total execution latency."""
        return self.oracle_latency_ms + self.surrogate_latency_ms + self.projection_latency_ms


# ---------------------------------------------------------------------------
# OperationalExperienceAccumulator (v2.16 will implement this)
# ---------------------------------------------------------------------------

class OperationalExperienceAccumulator:
    """Stub for v2.16. Only schema + accumulation, no learning.

    In v2.16 this will:
    - Accumulate OperationalExperienceEntry records
    - Compute per-family statistics for LoRA specialization
    - Generate replay buffers for continual learning
    - Export training-ready datasets

    For now, it only stores entries and provides basic stats.
    """

    def __init__(self) -> None:
        self._entries: list[OperationalExperienceEntry] = []

    def add(self, entry: OperationalExperienceEntry) -> None:
        """Add an operational experience entry."""
        self._entries.append(entry)

    @property
    def count(self) -> int:
        return len(self._entries)

    def family_stats(self, topology_family: str) -> dict[str, Any]:
        """Basic stats for a topology family. No learning."""
        family = [e for e in self._entries if e.topology_family == topology_family]
        n = len(family)
        if n == 0:
            return {"topology_family": topology_family, "count": 0}

        avg_iters = sum(e.projection_iterations for e in family) / n
        avg_budget = sum(e.projection_budget for e in family) / n
        warmstart_rate = sum(1 for e in family if e.warmstart_used) / n
        retrieval_rate = sum(1 for e in family if e.retrieval_used) / n
        degraded_rate = sum(1 for e in family if e.degraded) / n

        return {
            "topology_family": topology_family,
            "count": n,
            "avg_iterations": avg_iters,
            "avg_budget": avg_budget,
            "warmstart_rate": warmstart_rate,
            "retrieval_rate": retrieval_rate,
            "degraded_rate": degraded_rate,
            "avg_budget_efficiency": sum(e.budget_efficiency for e in family) / n,
        }

    def all_family_stats(self) -> dict[str, dict[str, Any]]:
        """Stats for all families."""
        families: set[str] = set()
        for e in self._entries:
            families.add(e.topology_family)
        return {f: self.family_stats(f) for f in sorted(families)}

    def trajectory_distribution(self) -> dict[str, int]:
        """Distribution of trajectory classes across all entries."""
        dist: dict[str, int] = {}
        for e in self._entries:
            dist[e.convergence_class] = dist.get(e.convergence_class, 0) + 1
        return dist

    def export_jsonl(self, path: str) -> None:
        """Export all entries as JSONL."""
        import json
        with open(path, "w") as f:
            for entry in self._entries:
                f.write(json.dumps(entry.to_json_dict(), sort_keys=True) + "\n")
