"""CPT Core Runtime — Execution Trace System.

Deterministic execution traces for reproducibility and debugging.
Traces saved to workspace/runtime_traces/ as JSONL.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hashlib


# ---------------------------------------------------------------------------
# ExecutionTrace — canonical trace record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionTrace:
    """Immutable execution trace for a single runtime pipeline run."""

    trace_id: str
    task_id: str
    runtime_ms: float
    oracle_runtime_ms: float
    surrogate_runtime_ms: float
    projection_runtime_ms: float
    projection_iterations: int
    topology_family: str
    failure_type: str | None
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            object.__setattr__(
                self, "timestamp",
                datetime.now(timezone.utc).isoformat(),
            )

    def fingerprint(self) -> str:
        """Deterministic SHA-256 of trace content."""
        blob = json.dumps({
            "task_id": self.task_id,
            "runtime_ms": round(self.runtime_ms, 6),
            "oracle_ms": round(self.oracle_runtime_ms, 6),
            "surrogate_ms": round(self.surrogate_runtime_ms, 6),
            "projection_ms": round(self.projection_runtime_ms, 6),
            "proj_iters": self.projection_iterations,
            "topology": self.topology_family,
            "failure": self.failure_type,
            "metadata": _sorted(self.metadata),
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps({
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "runtime_ms": self.runtime_ms,
            "oracle_runtime_ms": self.oracle_runtime_ms,
            "surrogate_runtime_ms": self.surrogate_runtime_ms,
            "projection_runtime_ms": self.projection_runtime_ms,
            "projection_iterations": self.projection_iterations,
            "topology_family": self.topology_family,
            "failure_type": self.failure_type,
            "timestamp": self.timestamp,
            "fingerprint": self.fingerprint(),
            "metadata": _sorted(self.metadata),
        }, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ExecutionTrace":
        d = json.loads(text)
        return cls(
            trace_id=d["trace_id"],
            task_id=d["task_id"],
            runtime_ms=d["runtime_ms"],
            oracle_runtime_ms=d["oracle_runtime_ms"],
            surrogate_runtime_ms=d["surrogate_runtime_ms"],
            projection_runtime_ms=d["projection_runtime_ms"],
            projection_iterations=d["projection_iterations"],
            topology_family=d["topology_family"],
            failure_type=d.get("failure_type"),
            timestamp=d.get("timestamp", ""),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# TraceStore — JSONL persistence
# ---------------------------------------------------------------------------

class TraceStore:
    """Deterministic JSONL trace storage."""

    DEFAULT_DIR = "workspace/runtime_traces"

    def __init__(self, base_dir: str | None = None) -> None:
        self._base_dir = Path(base_dir or self.DEFAULT_DIR)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, trace: ExecutionTrace) -> Path:
        """Save trace to JSONL file (one file per task_id)."""
        path = self._base_dir / f"{trace.task_id}.jsonl"
        with open(path, "a") as f:
            f.write(trace.to_json() + "\n")
        return path

    def load(self, task_id: str) -> list[ExecutionTrace]:
        """Load all traces for a task_id."""
        path = self._base_dir / f"{task_id}.jsonl"
        if not path.exists():
            return []
        traces = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(ExecutionTrace.from_json(line))
        return traces

    def load_all(self) -> list[ExecutionTrace]:
        """Load all traces from all files."""
        all_traces: list[ExecutionTrace] = []
        for path in sorted(self._base_dir.glob("*.jsonl")):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_traces.append(ExecutionTrace.from_json(line))
        return all_traces

    def clear(self) -> None:
        """Remove all traces (for testing)."""
        for path in self._base_dir.glob("*.jsonl"):
            path.unlink()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted(d: dict) -> dict:
    if not isinstance(d, dict):
        return d
    return {k: _sorted(v) if isinstance(v, dict) else v for k, v in sorted(d.items())}


def make_trace_id() -> str:
    """Generate deterministic-ish trace ID."""
    return f"trace_{uuid.uuid4().hex[:12]}"
