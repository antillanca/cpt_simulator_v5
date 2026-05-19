"""CPT Core Runtime — Dataset Governance.

Deterministic dataset manifests. Every runtime execution must reference
a dataset manifest ID for full reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# DatasetManifest — immutable dataset record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetManifest:
    """Canonical dataset manifest with deterministic SHA-256."""

    dataset_id: str
    sha256: str
    sample_count: int
    created_at: str
    domain: str = "circuit"
    description: str = ""
    topology_families: tuple[str, ...] = ()
    node_range: tuple[int, int] = (0, 0)
    edge_range: tuple[int, int] = (0, 0)
    metadata: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Deterministic fingerprint (same as sha256 field)."""
        return self.sha256

    def to_json(self) -> str:
        return json.dumps({
            "dataset_id": self.dataset_id,
            "sha256": self.sha256,
            "sample_count": self.sample_count,
            "created_at": self.created_at,
            "domain": self.domain,
            "description": self.description,
            "topology_families": list(self.topology_families),
            "node_range": list(self.node_range),
            "edge_range": list(self.edge_range),
            "metadata": _sorted(self.metadata),
        }, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "DatasetManifest":
        d = json.loads(text)
        return cls(
            dataset_id=d["dataset_id"],
            sha256=d["sha256"],
            sample_count=d["sample_count"],
            created_at=d["created_at"],
            domain=d.get("domain", "circuit"),
            description=d.get("description", ""),
            topology_families=tuple(d.get("topology_families", [])),
            node_range=tuple(d.get("node_range", [0, 0])),
            edge_range=tuple(d.get("edge_range", [0, 0])),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# DatasetRegistry — JSONL-based registry
# ---------------------------------------------------------------------------

class DatasetRegistry:
    """Deterministic dataset manifest registry.

    Stores manifests in workspace/datasets/registry.jsonl
    """

    DEFAULT_PATH = "workspace/datasets/registry.jsonl"

    def __init__(self, path: str | None = None) -> None:
        self._path = Path(path or self.DEFAULT_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def register(self, manifest: DatasetManifest) -> None:
        """Append manifest to registry."""
        with open(self._path, "a") as f:
            f.write(manifest.to_json() + "\n")

    def load_all(self) -> list[DatasetManifest]:
        """Load all registered manifests."""
        if not self._path.exists():
            return []
        manifests = []
        with open(self._path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    manifests.append(DatasetManifest.from_json(line))
        return manifests

    def find_by_id(self, dataset_id: str) -> DatasetManifest | None:
        """Find manifest by dataset_id."""
        for m in self.load_all():
            if m.dataset_id == dataset_id:
                return m
        return None

    def find_by_sha256(self, sha256: str) -> DatasetManifest | None:
        """Find manifest by content hash."""
        for m in self.load_all():
            if m.sha256 == sha256:
                return m
        return None

    def clear(self) -> None:
        """Reset registry (for testing)."""
        if self._path.exists():
            self._path.unlink()


# ---------------------------------------------------------------------------
# Dataset hashing utility
# ---------------------------------------------------------------------------

def compute_dataset_sha256(data: Any, seed: int = 42) -> str:
    """Compute deterministic SHA-256 hash for a dataset.

    Works with:
      - Path to a .pt / .json / .jsonl file
      - list of items (each item json-serialized deterministically)
    """
    h = hashlib.sha256()
    h.update(str(seed).encode())

    if isinstance(data, Path) or isinstance(data, str):
        p = Path(data)
        if p.exists() and p.is_file():
            for chunk in _read_chunks(p):
                h.update(chunk)
            return h.hexdigest()

    # Fallback: serialize items deterministically
    if isinstance(data, (list, tuple)):
        for item in data:
            h.update(json.dumps(item, sort_keys=True, default=str).encode())
        return h.hexdigest()

    # Single object
    h.update(json.dumps(data, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _read_chunks(path: Path, chunk_size: int = 8192) -> Any:
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted(d: dict) -> dict:
    if not isinstance(d, dict):
        return d
    return {k: _sorted(v) if isinstance(v, dict) else v for k, v in sorted(d.items())}
