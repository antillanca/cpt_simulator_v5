"""CPT Core Specification — Stable contracts for the CPT ecosystem.

This package defines ONLY contracts, schemas, validation, and dataclasses.
NO heavy business logic. All dataclasses are frozen (immutable).
All fingerprints are deterministic SHA-256 over canonical JSON.

Contracts:
    graph_spec       — CanonicalCircuitGraph, TopologyFamily
    projection_spec  — ProjectionResult
    model_spec       — CPTModel protocol, ModelMetadata, CircuitGNNAdapter
    experiment_spec  — ExperimentSpec
    report_spec      — EvaluationReport
    failure_taxonomy — FAILURE_TYPES, FailureCategory
    memory_spec      — MemoryEntry
"""

from backend.core_spec.graph_spec import (
    CanonicalCircuitGraph,
    TopologyFamily,
    compute_graph_fingerprint,
    from_circuit_graph,
    validate_graph,
)
from backend.core_spec.projection_spec import (
    ProjectionResult,
    from_projection_effort,
)
from backend.core_spec.model_spec import (
    CPTModel,
    CircuitGNNAdapter,
    ModelMetadata,
)
from backend.core_spec.experiment_spec import ExperimentSpec
from backend.core_spec.report_spec import EvaluationReport
from backend.core_spec.failure_taxonomy import (
    FAILURE_TYPES,
    FailureCategory,
    category_of,
    is_valid_failure_type,
    validate_failure_type,
    validate_taxonomy_consistency,
)
from backend.core_spec.memory_spec import MemoryEntry

__all__ = [
    "CanonicalCircuitGraph",
    "TopologyFamily",
    "compute_graph_fingerprint",
    "from_circuit_graph",
    "validate_graph",
    "ProjectionResult",
    "from_projection_effort",
    "CPTModel",
    "CircuitGNNAdapter",
    "ModelMetadata",
    "ExperimentSpec",
    "EvaluationReport",
    "FAILURE_TYPES",
    "FailureCategory",
    "category_of",
    "is_valid_failure_type",
    "validate_failure_type",
    "validate_taxonomy_consistency",
    "MemoryEntry",
]
