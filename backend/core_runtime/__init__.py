"""CPT Core Runtime — Execution Engine & Task Standardization.

This package implements the canonical execution pipeline:
  task -> oracle -> surrogate -> projection -> evaluation -> memory

v2.12: RuntimeTask, RuntimeResult, RuntimeExecutor, Protocols
v2.13: ExactMatchCache, CanonicalHashing, ExecutionPolicy,
       ConfidenceEstimation, CapabilityRouter, AtomicMemory
v2.14: Upgraded CapabilityRouter (retrieval, warmstart, cost-aware)
"""

from backend.core_runtime.task_runtime import (
    RuntimeTask,
    RuntimeResult,
    RuntimeExecutor,
    OracleProtocol,
    SurrogateProtocol,
    ProjectionProtocol,
    EvaluatorProtocol,
)
from backend.core_runtime.oracle_protocol import MNAOracleAdapter
from backend.core_runtime.surrogate_runtime import SurrogateRuntime, SurrogatePrediction
from backend.core_runtime.projection_runtime import ProjectionRuntime, ProjectionExecution
from backend.core_runtime.memory_runtime import MemoryRuntime
from backend.core_runtime.execution_trace import ExecutionTrace, TraceStore
from backend.core_runtime.dataset_registry import DatasetManifest, DatasetRegistry

# v2.13 additions
from backend.core_runtime.exact_cache import ExactMatchCache, ExactCacheEntry
from backend.core_runtime.task_hashing import compute_task_hash, compute_circuit_hash, canonicalize_task, HASH_SCHEMA_VERSION
from backend.core_runtime.execution_policy import ExecutionPolicy, RecoveryHandler
from backend.core_runtime.confidence_runtime import ConfidenceRuntime, ConfidenceEstimate
from backend.core_runtime.capability_router import CapabilityRouter, RoutingDecision

__all__ = [
    # v2.12
    "RuntimeTask", "RuntimeResult", "RuntimeExecutor",
    "OracleProtocol", "SurrogateProtocol", "ProjectionProtocol", "EvaluatorProtocol",
    "MNAOracleAdapter",
    "SurrogateRuntime", "SurrogatePrediction",
    "ProjectionRuntime", "ProjectionExecution",
    "MemoryRuntime",
    "ExecutionTrace", "TraceStore",
    "DatasetManifest", "DatasetRegistry",
    # v2.13
    "ExactMatchCache", "ExactCacheEntry",
    "compute_task_hash", "compute_circuit_hash", "canonicalize_task", "HASH_SCHEMA_VERSION",
    "ExecutionPolicy", "RecoveryHandler",
    "ConfidenceRuntime", "ConfidenceEstimate",
    "CapabilityRouter", "RoutingDecision",
]
