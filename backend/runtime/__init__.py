"""CPT Runtime — Retrieval Memory, Semantic Warm-Start, Cost Estimation & Adaptive Scheduling.

v2.14: Semantic retrieval memory and warm-start capabilities.
v2.15: Adaptive projection budgeting and operational scheduling.

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

# v2.15: Adaptive scheduling
from backend.runtime.projection_scheduler import (
    ProjectionBudget,
    ProjectionScheduler,
    StopDecision,
    # Trajectory classes
    TRAJECTORY_FAST_CONVERGING,
    TRAJECTORY_STABLE_LINEAR,
    TRAJECTORY_OSCILLATORY,
    TRAJECTORY_STALLED,
    TRAJECTORY_DIVERGENCE_RISK,
    TRAJECTORY_RETRIEVAL_ASSISTED,
    VALID_TRAJECTORY_CLASSES,
    # Stop reasons
    STOP_CONTINUE,
    STOP_CONVERGED,
    STOP_STAGNATED,
    STOP_DIMINISHING,
    STOP_DIVERGENCE,
    STOP_ESCALATE,
    STOP_BUDGET_EXHAUSTED,
    VALID_STOP_REASONS,
)
from backend.runtime.trajectory_analysis import (
    TrajectoryMetrics,
    TrajectoryAnalysisResult,
    TrajectoryAnalyzer,
)
from backend.runtime.execution_scheduler import (
    ExecutionSchedule,
    ExecutionOutcome,
    ExecutionScheduler,
    # Routes
    ROUTE_CACHE_HIT,
    ROUTE_RETRIEVAL_WARMSTART,
    ROUTE_RETRIEVAL_SEMANTIC,
    ROUTE_STANDARD,
    ROUTE_OOD_ESCALATED,
    ROUTE_ORACLE_FORCED,
    ROUTE_DEGRADED,
    VALID_ROUTES,
    # Outcomes
    OUTCOME_SUCCESS,
    OUTCOME_CONVERGED_EARLY,
    OUTCOME_STAGNATED,
    OUTCOME_DIVERGED,
    OUTCOME_ESCALATED,
    OUTCOME_BUDGET_EXHAUSTED,
    OUTCOME_DEGRADED,
    OUTCOME_CACHE_HIT,
    VALID_OUTCOMES,
)

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
    # v2.15: Projection Scheduler
    "ProjectionBudget", "ProjectionScheduler", "StopDecision",
    "TRAJECTORY_FAST_CONVERGING", "TRAJECTORY_STABLE_LINEAR",
    "TRAJECTORY_OSCILLATORY", "TRAJECTORY_STALLED",
    "TRAJECTORY_DIVERGENCE_RISK", "TRAJECTORY_RETRIEVAL_ASSISTED",
    "VALID_TRAJECTORY_CLASSES",
    "STOP_CONTINUE", "STOP_CONVERGED", "STOP_STAGNATED",
    "STOP_DIMINISHING", "STOP_DIVERGENCE", "STOP_ESCALATE",
    "STOP_BUDGET_EXHAUSTED", "VALID_STOP_REASONS",
    # v2.15: Trajectory Analysis
    "TrajectoryMetrics", "TrajectoryAnalysisResult", "TrajectoryAnalyzer",
    # v2.15: Execution Scheduler
    "ExecutionSchedule", "ExecutionOutcome", "ExecutionScheduler",
    "ROUTE_CACHE_HIT", "ROUTE_RETRIEVAL_WARMSTART", "ROUTE_RETRIEVAL_SEMANTIC",
    "ROUTE_STANDARD", "ROUTE_OOD_ESCALATED", "ROUTE_ORACLE_FORCED", "ROUTE_DEGRADED",
    "VALID_ROUTES",
    "OUTCOME_SUCCESS", "OUTCOME_CONVERGED_EARLY", "OUTCOME_STAGNATED",
    "OUTCOME_DIVERGED", "OUTCOME_ESCALATED", "OUTCOME_BUDGET_EXHAUSTED",
    "OUTCOME_DEGRADED", "OUTCOME_CACHE_HIT", "VALID_OUTCOMES",
    # FAISS (optional)
    "FaissRuntime", "TopKSimilarityResult",
]
