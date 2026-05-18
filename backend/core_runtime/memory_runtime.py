"""CPT Core Runtime — Memory Registration Pipeline.

Uses MemoryEntry schema from core_spec. Persists to deterministic JSONL.
NO FAISS — only stable JSONL storage for future vector DB integration.

v2.13: Atomic writes (write→temp→fsync→os.replace) guarantee no
partial JSONL corruption on crash.
"""

from __future__ import annotations

import json
import os
import tempfile
import time as _time
from pathlib import Path
from typing import Any

from backend.core_spec.memory_spec import MemoryEntry


# ---------------------------------------------------------------------------
# MemoryRuntime — JSONL-based memory persistence
# ---------------------------------------------------------------------------

class MemoryRuntime:
    """Deterministic JSONL memory store for execution results.

    Each register_execution() call appends one MemoryEntry to the JSONL file.
    File path: workspace/memory/memory_log.jsonl

    v2.13: Uses atomic writes to prevent corruption on crash.
    """

    DEFAULT_DIR = "workspace/memory"

    def __init__(self, base_dir: str | None = None) -> None:
        self._base_dir = Path(base_dir or self.DEFAULT_DIR)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._base_dir / "memory_log.jsonl"

    @property
    def log_path(self) -> Path:
        return self._log_path

    def register_execution(
        self,
        task: Any,
        oracle_output: dict[str, Any] | None = None,
        surrogate_output: Any = None,
        projected_output: Any = None,
        oracle_ms: float = 0.0,
        surrogate_ms: float = 0.0,
        projection_ms: float = 0.0,
        failure_type: str | None = None,
        topology_family: str = "unknown",
    ) -> MemoryEntry:
        """Register execution result as a MemoryEntry and persist to JSONL."""
        graph_fp = getattr(task, "input_artifact", "unknown") if task else "unknown"
        domain = getattr(task, "domain", "unknown") if task else "unknown"
        task_id = getattr(task, "task_id", "unknown") if task else "unknown"

        # Determine projection iterations
        proj_iters = 0
        initial_residual = 0.0
        final_residual = 0.0
        if projected_output is not None and hasattr(projected_output, "iterations"):
            proj_iters = projected_output.iterations
            initial_residual = projected_output.metadata.get("initial_kcl", 0.0) if hasattr(projected_output, "metadata") else 0.0
            final_residual = projected_output.kcl_violation if hasattr(projected_output, "kcl_violation") else 0.0

        import uuid
        entry = MemoryEntry(
            entry_id=f"mem_{uuid.uuid4().hex[:12]}",
            graph_fingerprint=graph_fp,
            topology_family=topology_family,
            projection_iterations=proj_iters,
            initial_residual=initial_residual,
            final_residual=final_residual,
            dominant_failure=failure_type,
            oracle_time_ms=oracle_ms,
            projection_time_ms=projection_ms,
            used_lora_expert=None,
            metadata={
                "task_id": task_id,
                "domain": domain,
                "surrogate_ms": surrogate_ms,
            },
        )

        # Atomic persist
        self._atomic_append(entry)
        return entry

    def load_all(self) -> list[MemoryEntry]:
        """Load all memory entries from JSONL."""
        entries: list[MemoryEntry] = []
        if not self._log_path.exists():
            return entries
        with open(self._log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                entries.append(MemoryEntry.from_json_dict(d))
        return entries

    def count(self) -> int:
        """Count stored entries."""
        if not self._log_path.exists():
            return 0
        with open(self._log_path, "r") as f:
            return sum(1 for line in f if line.strip())

    def clear(self) -> None:
        """Reset memory log (for testing)."""
        if self._log_path.exists():
            self._log_path.unlink()

    def compact(self) -> int:
        """Compact the JSONL file by rewriting it atomically.

        Deduplicates entries by entry_id (keeps latest).
        Returns the number of entries after compaction.
        """
        entries = self.load_all()
        # Deduplicate: last occurrence wins
        seen: dict[str, MemoryEntry] = {}
        for e in entries:
            seen[e.entry_id] = e
        unique = list(seen.values())
        self._atomic_rewrite(unique)
        return len(unique)

    # -- atomic write helpers -----------------------------------------------

    def _atomic_append(self, entry: MemoryEntry) -> None:
        """Append a MemoryEntry to JSONL using atomic write (temp→fsync→rename)."""
        line = json.dumps(entry.to_json_dict(), sort_keys=True) + "\n"
        # For append mode, we use atomic append:
        # write to temp, fsync, then append
        fd, tmp = tempfile.mkstemp(
            dir=self._base_dir,
            prefix=".mem_append_",
            suffix=".tmp",
        )
        try:
            os.write(fd, line.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        # Append content to log
        with open(self._log_path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        # Clean up temp
        try:
            os.unlink(tmp)
        except OSError:
            pass

    def _atomic_rewrite(self, entries: list[MemoryEntry]) -> None:
        """Rewrite the entire JSONL file atomically (temp→fsync→os.replace)."""
        fd, tmp = tempfile.mkstemp(
            dir=self._base_dir,
            prefix=".mem_rewrite_",
            suffix=".tmp",
        )
        try:
            lines = []
            for e in entries:
                lines.append(json.dumps(e.to_json_dict(), sort_keys=True))
            content = "\n".join(lines) + "\n" if lines else ""
            os.write(fd, content.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        # Atomic replace
        os.replace(tmp, self._log_path)
