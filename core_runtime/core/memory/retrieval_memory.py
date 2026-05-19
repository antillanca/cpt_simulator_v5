"""CPT Runtime — Retrieval Memory Foundation.

Deterministic retrieval memory layer. Stores execution experience
indexed by embedding SHA-256 for future similarity retrieval.

LAYERS REMAIN SEPARATED:
- Knowledge: frozen specs/taxonomy/contracts
- Memory: exact executions, JSONL traces, deterministic outputs
- Experience: embeddings, similarity retrieval, warm-start states
  ← THIS MODULE

DO NOT mix these layers.
"""

from __future__ import annotations

import hashlib
import json
import os
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# RetrievalEntry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalEntry:
    """One entry in the retrieval memory index.

    Links a task hash to its embedding, topology, projection metrics,
    and physical residuals. Used for semantic warm-start and routing.
    """

    task_hash: str
    embedding_sha256: str
    topology_family: str
    node_count: int
    edge_count: int
    confidence: float
    projection_iterations: int
    kcl_residual: float
    kvl_residual: float
    timestamp: str
    embedding_path: str
    trace_path: str

    # -- Fingerprint ---------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        blob = json.dumps({
            "task_hash": self.task_hash,
            "embedding_sha256": self.embedding_sha256,
            "topology_family": self.topology_family,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "confidence": round(self.confidence, 8),
            "projection_iterations": self.projection_iterations,
            "kcl_residual": round(self.kcl_residual, 12),
            "kvl_residual": round(self.kvl_residual, 12),
            "timestamp": self.timestamp,
            "embedding_path": self.embedding_path,
            "trace_path": self.trace_path,
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    # -- Serialization -------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "task_hash": self.task_hash,
            "embedding_sha256": self.embedding_sha256,
            "topology_family": self.topology_family,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "confidence": round(self.confidence, 8),
            "projection_iterations": self.projection_iterations,
            "kcl_residual": round(self.kcl_residual, 12),
            "kvl_residual": round(self.kvl_residual, 12),
            "timestamp": self.timestamp,
            "embedding_path": self.embedding_path,
            "trace_path": self.trace_path,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "RetrievalEntry":
        return cls(
            task_hash=data["task_hash"],
            embedding_sha256=data["embedding_sha256"],
            topology_family=data["topology_family"],
            node_count=data["node_count"],
            edge_count=data["edge_count"],
            confidence=data["confidence"],
            projection_iterations=data["projection_iterations"],
            kcl_residual=data["kcl_residual"],
            kvl_residual=data["kvl_residual"],
            timestamp=data["timestamp"],
            embedding_path=data.get("embedding_path", ""),
            trace_path=data.get("trace_path", ""),
        )

    # -- Validation ----------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.task_hash:
            errors.append("task_hash must not be empty")
        if not self.embedding_sha256:
            errors.append("embedding_sha256 must not be empty")
        if self.node_count < 0:
            errors.append("node_count must be non-negative")
        if self.edge_count < 0:
            errors.append("edge_count must be non-negative")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append("confidence must be in [0, 1]")
        if self.projection_iterations < 0:
            errors.append("projection_iterations must be non-negative")
        if self.kcl_residual < 0:
            errors.append("kcl_residual must be non-negative")
        if self.kvl_residual < 0:
            errors.append("kvl_residual must be non-negative")
        return errors


# ---------------------------------------------------------------------------
# RetrievalMemory
# ---------------------------------------------------------------------------

class RetrievalMemory:
    """Deterministic retrieval memory store.

    Persists RetrievalEntry records to JSONL. No duplicate task hashes.
    Deterministic ordering by task_hash.
    """

    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._index_path = self._base / "retrieval_index.jsonl"
        self._entries: dict[str, RetrievalEntry] = {}
        self._load()

    # -- Load / Save ---------------------------------------------------------

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        with open(self._index_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = RetrievalEntry.from_json_dict(data)
                    self._entries[entry.task_hash] = entry
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    def _atomic_write(self) -> None:
        """Atomic write: temp → fsync → rename."""
        tmp = self._index_path.with_suffix(".tmp")
        ordered = sorted(self._entries.values(), key=lambda e: e.task_hash)
        with open(tmp, "w") as f:
            for entry in ordered:
                f.write(json.dumps(entry.to_json_dict(), sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._index_path)

    # -- CRUD ----------------------------------------------------------------

    def add(self, entry: RetrievalEntry) -> bool:
        """Add entry. Returns False if duplicate (no silent overwrite)."""
        errs = entry.validate()
        if errs:
            raise ValueError(f"Invalid RetrievalEntry: {errs}")
        if entry.task_hash in self._entries:
            return False  # No silent duplicate insertion
        self._entries[entry.task_hash] = entry
        self._atomic_write()
        return True

    def search(self, task_hash: str) -> RetrievalEntry | None:
        """Lookup by exact task hash."""
        return self._entries.get(task_hash)

    def search_by_topology(self, topology_family: str) -> list[RetrievalEntry]:
        """Return all entries for a topology family, sorted by task_hash."""
        results = [
            e for e in self._entries.values()
            if e.topology_family == topology_family
        ]
        return sorted(results, key=lambda e: e.task_hash)

    def rebuild_index(self) -> int:
        """Rebuild index from JSONL (dedup + sort). Returns entry count."""
        old_count = len(self._entries)
        self._entries.clear()
        self._load()
        self._atomic_write()
        return len(self._entries)

    def compact(self) -> int:
        """Remove entries whose embedding files no longer exist."""
        to_remove = []
        for task_hash, entry in self._entries.items():
            if entry.embedding_path and not Path(entry.embedding_path).exists():
                to_remove.append(task_hash)
        for th in to_remove:
            del self._entries[th]
        if to_remove:
            self._atomic_write()
        return len(to_remove)

    def remove(self, task_hash: str) -> bool:
        """Remove entry by task hash. Returns True if existed."""
        if task_hash in self._entries:
            del self._entries[task_hash]
            self._atomic_write()
            return True
        return False

    # -- Stats ---------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        entries = list(self._entries.values())
        topo_counts: dict[str, int] = {}
        total_proj_iters = 0
        for e in entries:
            topo_counts[e.topology_family] = topo_counts.get(e.topology_family, 0) + 1
            total_proj_iters += e.projection_iterations

        n = len(entries)
        return {
            "total_entries": n,
            "topology_families": len(topo_counts),
            "topology_distribution": dict(sorted(topo_counts.items())),
            "avg_projection_iterations": total_proj_iters / n if n else 0.0,
            "avg_confidence": sum(e.confidence for e in entries) / n if n else 0.0,
        }

    @property
    def count(self) -> int:
        return len(self._entries)

    def all_entries(self) -> list[RetrievalEntry]:
        """Return all entries sorted deterministically."""
        return sorted(self._entries.values(), key=lambda e: e.task_hash)
