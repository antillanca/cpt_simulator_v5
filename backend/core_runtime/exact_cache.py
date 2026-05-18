"""CPT Core Runtime — Exact-Match Cache.

Deterministic SHA-256 exact-match cache. Equivalent circuits produce
identical hashes, enabling instant result reuse without re-execution.

Persistence: workspace/exact_cache/ (JSONL)
"""

from __future__ import annotations

import hashlib
import json
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.core_runtime.task_runtime import RuntimeResult


# ---------------------------------------------------------------------------
# ExactCacheEntry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExactCacheEntry:
    """One entry in the exact-match cache.

    task_hash: SHA-256 of canonicalized task (topology, values, config)
    runtime_result_hash: SHA-256 of the RuntimeResult (for integrity check)
    """
    task_hash: str
    runtime_result_hash: str
    topology_family: str
    projection_iterations: int
    failure_type: str | None
    created_at: str

    @property
    def fingerprint(self) -> str:
        blob = json.dumps({
            "task_hash": self.task_hash,
            "runtime_result_hash": self.runtime_result_hash,
            "topology_family": self.topology_family,
            "projection_iterations": self.projection_iterations,
            "failure_type": self.failure_type,
            "created_at": self.created_at,
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps({
            "task_hash": self.task_hash,
            "runtime_result_hash": self.runtime_result_hash,
            "topology_family": self.topology_family,
            "projection_iterations": self.projection_iterations,
            "failure_type": self.failure_type,
            "created_at": self.created_at,
        }, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> ExactCacheEntry:
        d = json.loads(text)
        return cls(**d)


# ---------------------------------------------------------------------------
# RuntimeResult hash — deterministic fingerprint of a result
# ---------------------------------------------------------------------------

def _tensor_to_list(v: Any) -> Any:
    """Convert tensors / SurrogatePrediction to JSON-serializable lists."""
    import torch

    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    # SurrogatePrediction or similar dataclass — extract .prediction tensor
    if hasattr(v, "prediction") and isinstance(v.prediction, torch.Tensor):
        return v.prediction.detach().cpu().tolist()
    if v is None:
        return None
    return v


def _hash_runtime_result(result: RuntimeResult) -> str:
    """Deterministic SHA-256 of a RuntimeResult (tensor-safe)."""
    blob = json.dumps({
        "task_id": result.task_id,
        "task_fingerprint": result.task_fingerprint,
        "oracle_voltages": _tensor_to_list(result.oracle_voltages),
        "surrogate_voltages": _tensor_to_list(result.surrogate_voltages),
        "projected_voltages": _tensor_to_list(result.projected_voltages),
        "total_runtime_ms": round(result.total_runtime_ms, 6),
        "oracle_runtime_ms": round(result.oracle_runtime_ms, 6),
        "surrogate_runtime_ms": round(result.surrogate_runtime_ms, 6),
        "projection_runtime_ms": round(result.projection_runtime_ms, 6),
        "failure_type": result.failure_type,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# ExactMatchCache
# ---------------------------------------------------------------------------

class ExactMatchCache:
    """Deterministic exact-match cache backed by JSONL.

    Index: task_hash -> RuntimeResult + ExactCacheEntry
    """

    DEFAULT_DIR = "workspace/exact_cache"

    def __init__(self, base_dir: str | None = None) -> None:
        self._base_dir = Path(base_dir or self.DEFAULT_DIR)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._base_dir / "cache_index.jsonl"
        self._results_dir = self._base_dir / "results"
        self._results_dir.mkdir(parents=True, exist_ok=True)
        # In-memory index for fast lookup
        self._index: dict[str, ExactCacheEntry] = {}
        self._load_index()

    # -- public API ---------------------------------------------------------

    def get(self, task_hash: str) -> RuntimeResult | None:
        """Retrieve a cached RuntimeResult by task_hash, or None."""
        entry = self._index.get(task_hash)
        if entry is None:
            return None
        result_path = self._results_dir / f"{task_hash}.json"
        if not result_path.exists():
            return None
        return self._load_result(result_path)

    def put(self, task_hash: str, result: RuntimeResult) -> ExactCacheEntry:
        """Cache a RuntimeResult. Returns the ExactCacheEntry."""
        result_hash = _hash_runtime_result(result)
        # Extract metadata from result
        topology_family = "unknown"
        projection_iterations = 0
        failure_type = result.failure_type
        if result.projection_result is not None:
            projection_iterations = getattr(result.projection_result, "iterations", 0)
        if result.memory_entry is not None:
            topology_family = getattr(result.memory_entry, "topology_family", topology_family)
        if result.metadata:
            topology_family = result.metadata.get("topology_family", topology_family)

        entry = ExactCacheEntry(
            task_hash=task_hash,
            runtime_result_hash=result_hash,
            topology_family=topology_family,
            projection_iterations=projection_iterations,
            failure_type=failure_type,
            created_at=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        )

        # Persist result + index
        self._save_result(task_hash, result)
        self._index[task_hash] = entry
        self._append_index(entry)
        return entry

    def contains(self, task_hash: str) -> bool:
        return task_hash in self._index

    def count(self) -> int:
        return len(self._index)

    def clear(self) -> None:
        """Reset cache (for testing)."""
        self._index.clear()
        if self._index_path.exists():
            self._index_path.unlink()
        for f in self._results_dir.glob("*.json"):
            f.unlink()

    def entries(self) -> list[ExactCacheEntry]:
        return list(self._index.values())

    # -- internal -----------------------------------------------------------

    def _load_index(self) -> None:
        if not self._index_path.exists():
            return
        with open(self._index_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = ExactCacheEntry.from_json(line)
                self._index[entry.task_hash] = entry

    def _append_index(self, entry: ExactCacheEntry) -> None:
        with open(self._index_path, "a") as f:
            f.write(entry.to_json() + "\n")

    def _save_result(self, task_hash: str, result: RuntimeResult) -> None:
        data = {
            "task_id": result.task_id,
            "task_fingerprint": result.task_fingerprint,
            "oracle_voltages": _tensor_to_list(result.oracle_voltages),
            "surrogate_voltages": _tensor_to_list(result.surrogate_voltages),
            "projected_voltages": _tensor_to_list(result.projected_voltages),
            "total_runtime_ms": round(result.total_runtime_ms, 6),
            "oracle_runtime_ms": round(result.oracle_runtime_ms, 6),
            "surrogate_runtime_ms": round(result.surrogate_runtime_ms, 6),
            "projection_runtime_ms": round(result.projection_runtime_ms, 6),
            "failure_type": result.failure_type,
        }
        path = self._results_dir / f"{task_hash}.json"
        with open(path, "w") as f:
            json.dump(data, f, sort_keys=True, indent=2)

    def _load_result(self, path: Path) -> RuntimeResult:
        import torch

        with open(path, "r") as f:
            data = json.load(f)

        # Reconstruct tensors
        oracle_v = data.get("oracle_voltages")
        surr_v = data.get("surrogate_voltages")
        proj_v = data.get("projected_voltages")
        if isinstance(oracle_v, list):
            oracle_v = torch.tensor(oracle_v, dtype=torch.float32)
        if isinstance(surr_v, list):
            surr_v = torch.tensor(surr_v, dtype=torch.float32)
        if isinstance(proj_v, list):
            proj_v = torch.tensor(proj_v, dtype=torch.float32)

        return RuntimeResult(
            task_id=data["task_id"],
            task_fingerprint=data["task_fingerprint"],
            oracle_voltages=oracle_v,
            surrogate_voltages=surr_v,
            projected_voltages=proj_v,
            projection_result=None,
            evaluation_report=None,
            memory_entry=None,
            total_runtime_ms=data["total_runtime_ms"],
            oracle_runtime_ms=data["oracle_runtime_ms"],
            surrogate_runtime_ms=data["surrogate_runtime_ms"],
            projection_runtime_ms=data["projection_runtime_ms"],
            failure_type=data.get("failure_type"),
        )
