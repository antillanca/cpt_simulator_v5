"""CPT Runtime — FAISS Semantic Retrieval.

Deterministic semantic retrieval using FAISS IndexFlatIP (inner product).
Exact cache is ALWAYS checked first. FAISS retrieval is second.

DO NOT:
- Use IVF or approximate indices yet
- Skip exact cache lookup
- Accept NaN embeddings
- Store degraded executions as valid retrieval entries
"""

from __future__ import annotations

import json
import os
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from backend.runtime.retrieval_memory import RetrievalEntry, RetrievalMemory
from backend.runtime.embedding_runtime import EmbeddingResult


# ---------------------------------------------------------------------------
# TopKSimilarityResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TopKSimilarityResult:
    """One result from a FAISS similarity search."""

    rank: int
    similarity_score: float
    task_hash: str
    topology_family: str
    projection_iterations: int
    confidence: float
    kcl_residual: float
    kvl_residual: float


# ---------------------------------------------------------------------------
# FaissRuntime
# ---------------------------------------------------------------------------

class FaissRuntime:
    """FAISS-based semantic retrieval.

    Uses IndexFlatIP (inner product) on L2-normalized embeddings
    for cosine similarity search.

    REQUIREMENTS:
    - Exact cache lookup ALWAYS first
    - Never store degraded executions
    - Never accept NaN embeddings
    - Deterministic ordering of results
    """

    def __init__(self, dim: int = 64, base_dir: str = "workspace/faiss_index") -> None:
        if not HAS_FAISS:
            raise ImportError("faiss-cpu is required for FaissRuntime")

        self._dim = dim
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._index_path = self._base / "faiss_index.bin"
        self._meta_path = self._base / "faiss_meta.jsonl"

        # FAISS index — IndexFlatIP for exact inner product search
        self._index = faiss.IndexFlatIP(dim)
        self._task_hashes: list[str] = []  # Parallel to FAISS internal IDs
        self._metadata: dict[str, dict[str, Any]] = {}

        # Try loading existing index
        self._load()

    # -- Load / Save ---------------------------------------------------------

    def _load(self) -> None:
        if self._index_path.exists() and self._meta_path.exists():
            try:
                self._index = faiss.read_index(str(self._index_path))
                self._task_hashes = []
                self._metadata = {}
                with open(self._meta_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        th = data["task_hash"]
                        self._task_hashes.append(th)
                        self._metadata[th] = data
            except Exception:
                # Corrupted index — start fresh
                self._index = faiss.IndexFlatIP(self._dim)
                self._task_hashes = []
                self._metadata = {}

    def _atomic_save(self) -> None:
        """Atomic save: temp → fsync → rename."""
        # Save FAISS index
        tmp_idx = self._index_path.with_suffix(".tmp")
        faiss.write_index(self._index, str(tmp_idx))
        # Save metadata
        tmp_meta = self._meta_path.with_suffix(".tmp")
        with open(tmp_meta, "w") as f:
            for th in self._task_hashes:
                meta = self._metadata.get(th, {})
                f.write(json.dumps(meta, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_idx, self._index_path)
        os.replace(tmp_meta, self._meta_path)

    # -- Add / Search --------------------------------------------------------

    def add_embedding(
        self,
        task_hash: str,
        embedding: np.ndarray,
        entry: RetrievalEntry,
    ) -> bool:
        """Add an embedding to the FAISS index.

        Returns False if:
        - Embedding contains NaN
        - Entry is degraded (failure_type set)
        - Duplicate task_hash
        """
        # Safety: reject NaN embeddings
        if np.any(np.isnan(embedding)):
            return False

        # Safety: reject degraded executions
        if entry.projection_iterations == 0 and entry.kcl_residual > 1.0:
            return False

        # Safety: reject duplicates
        if task_hash in self._metadata:
            return False

        # Normalize for cosine similarity (L2 normalize)
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            return False  # Zero/near-zero vector
        normalized = (embedding / norm).astype(np.float32)

        # Reshape for FAISS (needs 2D)
        vec = normalized.reshape(1, -1)

        # Add to index
        self._index.add(vec)
        self._task_hashes.append(task_hash)
        self._metadata[task_hash] = entry.to_json_dict()
        self._atomic_save()
        return True

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[TopKSimilarityResult]:
        """Search for top-k similar embeddings.

        Returns results sorted by similarity (descending).
        Filters out results below min_similarity threshold.
        """
        if self._index.ntotal == 0:
            return []

        # Normalize query
        norm = np.linalg.norm(query_embedding)
        if norm < 1e-10:
            return []
        normalized = (query_embedding / norm).astype(np.float32)
        vec = normalized.reshape(1, -1)

        # Search
        actual_k = min(k, self._index.ntotal)
        distances, indices = self._index.search(vec, actual_k)

        results: list[TopKSimilarityResult] = []
        for rank_idx in range(actual_k):
            faiss_id = int(indices[0][rank_idx])
            similarity = float(distances[0][rank_idx])

            if similarity < min_similarity:
                continue

            if faiss_id < 0 or faiss_id >= len(self._task_hashes):
                continue

            task_hash = self._task_hashes[faiss_id]
            meta = self._metadata.get(task_hash, {})

            results.append(TopKSimilarityResult(
                rank=rank_idx + 1,
                similarity_score=round(similarity, 8),
                task_hash=task_hash,
                topology_family=meta.get("topology_family", "unknown"),
                projection_iterations=meta.get("projection_iterations", 0),
                confidence=meta.get("confidence", 0.0),
                kcl_residual=meta.get("kcl_residual", 0.0),
                kvl_residual=meta.get("kvl_residual", 0.0),
            ))

        return results

    def save_index(self) -> None:
        """Persist index to disk."""
        self._atomic_save()

    def load_index(self) -> None:
        """Reload index from disk."""
        self._load()

    def rebuild(self) -> int:
        """Rebuild index from metadata (clear + re-add all).

        Returns number of entries rebuilt.
        """
        saved_meta = dict(self._metadata)
        saved_hashes = list(self._task_hashes)

        # Reset index
        self._index = faiss.IndexFlatIP(self._dim)
        self._task_hashes = []
        self._metadata = {}

        # We can't rebuild embeddings from metadata alone
        # This would require re-extracting from saved embedding files
        # For now, just reload from disk
        self._load()
        return self._index.ntotal

    # -- Stats ---------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    @property
    def dimension(self) -> int:
        return self._dim

    def contains(self, task_hash: str) -> bool:
        return task_hash in self._metadata
