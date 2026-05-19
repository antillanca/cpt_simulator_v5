"""CPT Runtime — Projection Experience Memory.

Stores convergence behavior for future:
- LoRA specialization
- Replay learning
- Adaptive routing

DO NOT implement learning yet. ONLY collect experience.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.runtime.retrieval_memory import RetrievalMemory


# ---------------------------------------------------------------------------
# ProjectionExperienceEntry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectionExperienceEntry:
    """One projection experience record.

    Captures the convergence behavior of a single projection execution.
    """

    task_hash: str
    topology_family: str
    initial_residual: float
    final_residual: float
    residual_slope: float        # Average residual decrease per iteration
    iterations: int
    converged: bool
    kcl_residual: float
    kvl_residual: float
    used_warmstart: bool
    warmstart_similarity: float
    timestamp: str

    @property
    def fingerprint(self) -> str:
        blob = json.dumps({
            "task_hash": self.task_hash,
            "topology_family": self.topology_family,
            "initial_residual": round(self.initial_residual, 12),
            "final_residual": round(self.final_residual, 12),
            "residual_slope": round(self.residual_slope, 12),
            "iterations": self.iterations,
            "converged": self.converged,
            "kcl_residual": round(self.kcl_residual, 12),
            "kvl_residual": round(self.kvl_residual, 12),
            "used_warmstart": self.used_warmstart,
            "warmstart_similarity": round(self.warmstart_similarity, 8),
            "timestamp": self.timestamp,
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "task_hash": self.task_hash,
            "topology_family": self.topology_family,
            "initial_residual": round(self.initial_residual, 12),
            "final_residual": round(self.final_residual, 12),
            "residual_slope": round(self.residual_slope, 12),
            "iterations": self.iterations,
            "converged": self.converged,
            "kcl_residual": round(self.kcl_residual, 12),
            "kvl_residual": round(self.kvl_residual, 12),
            "used_warmstart": self.used_warmstart,
            "warmstart_similarity": round(self.warmstart_similarity, 8),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "ProjectionExperienceEntry":
        return cls(
            task_hash=data["task_hash"],
            topology_family=data["topology_family"],
            initial_residual=data["initial_residual"],
            final_residual=data["final_residual"],
            residual_slope=data["residual_slope"],
            iterations=data["iterations"],
            converged=data["converged"],
            kcl_residual=data["kcl_residual"],
            kvl_residual=data["kvl_residual"],
            used_warmstart=data.get("used_warmstart", False),
            warmstart_similarity=data.get("warmstart_similarity", 0.0),
            timestamp=data["timestamp"],
        )


# ---------------------------------------------------------------------------
# ProjectionExperienceMemory
# ---------------------------------------------------------------------------

class ProjectionExperienceMemory:
    """Store convergence behavior for future learning.

    Persists to JSONL with atomic writes.
    Provides family-level statistics queries.
    """

    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._path = self._base / "projection_experience.jsonl"
        self._entries: list[ProjectionExperienceEntry] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with open(self._path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = ProjectionExperienceEntry.from_json_dict(data)
                    self._entries.append(entry)
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    def _atomic_write(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for entry in self._entries:
                f.write(json.dumps(entry.to_json_dict(), sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._path)

    def add(self, entry: ProjectionExperienceEntry) -> None:
        """Record a projection experience."""
        self._entries.append(entry)
        self._atomic_write()

    def family_stats(self, topology_family: str) -> dict[str, Any]:
        """Get family-level projection statistics."""
        family_entries = [
            e for e in self._entries
            if e.topology_family == topology_family
        ]
        n = len(family_entries)
        if n == 0:
            return {
                "topology_family": topology_family,
                "count": 0,
                "avg_iterations": 0,
                "convergence_rate": 0.0,
                "avg_residual_slope": 0.0,
                "warmstart_usage_rate": 0.0,
            }

        converged = sum(1 for e in family_entries if e.converged)
        total_iters = sum(e.iterations for e in family_entries)
        total_slope = sum(e.residual_slope for e in family_entries)
        warmstarted = sum(1 for e in family_entries if e.used_warmstart)

        return {
            "topology_family": topology_family,
            "count": n,
            "avg_iterations": total_iters / n,
            "convergence_rate": converged / n,
            "avg_residual_slope": total_slope / n,
            "warmstart_usage_rate": warmstarted / n,
        }

    def all_family_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all topology families."""
        families: set[str] = set()
        for e in self._entries:
            families.add(e.topology_family)
        return {f: self.family_stats(f) for f in sorted(families)}

    @property
    def count(self) -> int:
        return len(self._entries)

    def recent_entries(self, n: int = 10) -> list[ProjectionExperienceEntry]:
        """Return the n most recent entries."""
        return self._entries[-n:]
