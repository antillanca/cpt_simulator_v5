"""CPT Runtime — Retrieval Memory, Semantic Warm-Start & Cost Estimation.

v2.14 adds semantic retrieval memory and warm-start capabilities.

MEMORY LAYERS (STRICTLY SEPARATED):
- Knowledge: frozen specs/taxonomy/contracts (core_spec)
- Memory: exact executions, JSONL traces, deterministic outputs (core_runtime)
- Experience: embeddings, similarity retrieval, warm-start states (runtime)
  ← THIS PACKAGE

DO NOT MIX THESE LAYERS.
"""

from backend.runtime.retrieval_memory import RetrievalEntry, RetrievalMemory
from backend.runtime.embedding_runtime import EmbeddingResult, extract_graph_embedding, normalize_embedding, compute_embedding_sha256
from backend.runtime.cost_estimator import ExecutionCostEstimate, CostEstimator
from backend.runtime.warmstart_runtime import WarmStartResult, WarmstartRuntime
from backend.runtime.projection_experience import ProjectionExperienceEntry, ProjectionExperienceMemory

# FAISS is optional
try:
    from backend.runtime.faiss_runtime import FaissRuntime, TopKSimilarityResult
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

__all__ = [
    # Retrieval
    "RetrievalEntry", "RetrievalMemory",
    # Embedding
    "EmbeddingResult", "extract_graph_embedding", "normalize_embedding", "compute_embedding_sha256",
    # Cost
    "ExecutionCostEstimate", "CostEstimator",
    # Warmstart
    "WarmStartResult", "WarmstartRuntime",
    # Projection Experience
    "ProjectionExperienceEntry", "ProjectionExperienceMemory",
    # FAISS (optional)
    "FaissRuntime", "TopKSimilarityResult",
]
