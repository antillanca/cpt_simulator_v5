"""CORE v3.3 — Uncertainty Memory.

Store executions that reached transitional or indeterminate trust levels.
This is NOT retrieval memory. This is NOT exact cache.
This is memory of indeterminacy and operational risk.

SAFETY GUARANTEE:
  - Uncertainty memory entries are NEVER used as clean cache hits
  - Degraded executions can be stored here, but NEVER in clean exact cache
  - This memory does not alter execution outputs
  - Same input -> same record ordering
  - All entries are SHA-256 anchored
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# UncertaintyEntry — frozen, hash-anchored
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UncertaintyEntry:
    """Record of an execution with transitional or indeterminate trust."""
    task_hash: str
    reason: str
    topology_signature: str
    routing_action: str
    projection_iterations: int
    residual: float
    confidence_score: float
    trust_level: str
    timestamp: str
    scheduler_context: dict[str, Any]
    metadata: dict[str, Any]

    def anchor(self) -> str:
        """SHA-256 anchor for integrity verification."""
        payload = json.dumps({
            "task_hash": self.task_hash,
            "reason": self.reason,
            "topology_signature": self.topology_signature,
            "routing_action": self.routing_action,
            "projection_iterations": self.projection_iterations,
            "residual": round(self.residual, 12),
            "confidence_score": round(self.confidence_score, 12),
            "trust_level": self.trust_level,
            "timestamp": self.timestamp,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_hash": self.task_hash,
            "reason": self.reason,
            "topology_signature": self.topology_signature,
            "routing_action": self.routing_action,
            "projection_iterations": self.projection_iterations,
            "residual": self.residual,
            "confidence_score": self.confidence_score,
            "trust_level": self.trust_level,
            "timestamp": self.timestamp,
            "anchor": self.anchor(),
            "scheduler_context": self.scheduler_context,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UncertaintyEntry:
        """Reconstruct from a dict (e.g., JSONL line)."""
        return cls(
            task_hash=d["task_hash"],
            reason=d["reason"],
            topology_signature=d["topology_signature"],
            routing_action=d["routing_action"],
            projection_iterations=d["projection_iterations"],
            residual=d["residual"],
            confidence_score=d["confidence_score"],
            trust_level=d["trust_level"],
            timestamp=d["timestamp"],
            scheduler_context=d.get("scheduler_context", {}),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Known reason codes
# ---------------------------------------------------------------------------

UNCERTAINTY_REASONS = frozenset({
    "NaN",
    "timeout",
    "instability",
    "divergence",
    "low_confidence",
    "high_residual",
    "ood_execution",
})


# ---------------------------------------------------------------------------
# UncertaintyMemory — deterministic, ordered, atomic
# ---------------------------------------------------------------------------

class UncertaintyMemory:
    """Store and query operational uncertainty records.

    Invariants:
    - Same task_hash -> at most one entry (latest wins, but deterministic)
    - Ordering is deterministic (by insertion order + task_hash tiebreak)
    - Atomic file writes (write-to-temp, then rename)
    - Entries are NEVER used as clean exact cache hits
    """

    def __init__(self) -> None:
        self._entries: dict[str, UncertaintyEntry] = {}
        self._order: list[str] = []  # deterministic insertion order

    def add_entry(self, entry: UncertaintyEntry) -> None:
        """Add an uncertainty entry. Overwrites if same task_hash exists."""
        if entry.task_hash in self._entries:
            # Replace in-place to preserve ordering
            self._entries[entry.task_hash] = entry
        else:
            self._entries[entry.task_hash] = entry
            self._order.append(entry.task_hash)

    def contains(self, task_hash: str) -> bool:
        """Check if a task hash has an uncertainty record."""
        return task_hash in self._entries

    def get(self, task_hash: str) -> UncertaintyEntry | None:
        """Retrieve uncertainty entry by task hash."""
        return self._entries.get(task_hash)

    def search_by_reason(self, reason: str) -> list[UncertaintyEntry]:
        """Find all entries matching a reason code."""
        return [
            self._entries[h] for h in self._order
            if self._entries[h].reason == reason
        ]

    def search_by_trust_level(self, trust_level: str) -> list[UncertaintyEntry]:
        """Find all entries matching a trust level."""
        return [
            self._entries[h] for h in self._order
            if self._entries[h].trust_level == trust_level
        ]

    def all_entries(self) -> list[UncertaintyEntry]:
        """Return all entries in deterministic insertion order."""
        return [self._entries[h] for h in self._order]

    def __len__(self) -> int:
        return len(self._entries)

    # -----------------------------------------------------------------------
    # Persistence — atomic JSONL
    # -----------------------------------------------------------------------

    def export_jsonl(self, path: str | Path) -> None:
        """Write all entries to a JSONL file atomically."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(self._entries[h].to_dict(), sort_keys=True)
            for h in self._order
        ]
        content = "\n".join(lines) + ("\n" if lines else "")
        # Atomic write: temp file then rename
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)

    def load_jsonl(self, path: str | Path) -> None:
        """Load entries from a JSONL file. Appends to existing entries."""
        path = Path(path)
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            entry = UncertaintyEntry.from_dict(d)
            self.add_entry(entry)

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def reason_distribution(self) -> dict[str, int]:
        """Count entries by reason."""
        dist: dict[str, int] = {}
        for entry in self.all_entries():
            dist[entry.reason] = dist.get(entry.reason, 0) + 1
        return dist

    def trust_level_distribution(self) -> dict[str, int]:
        """Count entries by trust level."""
        dist: dict[str, int] = {}
        for entry in self.all_entries():
            dist[entry.trust_level] = dist.get(entry.trust_level, 0) + 1
        return dist
