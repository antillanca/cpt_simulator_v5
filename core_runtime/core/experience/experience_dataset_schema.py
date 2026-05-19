#!/usr/bin/env python3
"""CPT v2.15 -- Experience Dataset Schema for v2.16 preparation.

FROZEN SCHEMAS ONLY. No implementation of replay, LoRA, continual training,
or distributed execution. These schemas define the data format that v2.16
will consume for experience-based learning.

Each schema is immutable (frozen dataclass) with JSON roundtrip support.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════
# Schema version
# ═══════════════════════════════════════════════════════════════

SCHEMA_VERSION = "v2.15"


# ═══════════════════════════════════════════════════════════════
# 1. ConvergenceTraceSchema
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ConvergenceTraceSchema:
    """Per-execution residual trace with metadata."""
    schema_version: str = field(default=SCHEMA_VERSION)
    task_hash: str = ""
    topology_family: str = ""
    node_count: int = 0
    edge_count: int = 0
    initial_residual: float = 0.0
    final_residual: float = 0.0
    iterations: int = 0
    convergence_class: str = ""
    residual_history_json: str = "[]"  # JSON-encoded list[float]
    used_warmstart: bool = False
    warmstart_similarity: float = 0.0
    projection_budget: int = 0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.node_count < 0:
            raise ValueError(f"node_count must be >= 0, got {self.node_count}")
        if self.edge_count < 0:
            raise ValueError(f"edge_count must be >= 0, got {self.edge_count}")
        if self.iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {self.iterations}")
        if self.projection_budget < 0:
            raise ValueError(f"projection_budget must be >= 0, got {self.projection_budget}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConvergenceTraceSchema:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════
# 2. RoutingOutcomeSchema
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RoutingOutcomeSchema:
    """Scheduler routing decision with context."""
    schema_version: str = field(default=SCHEMA_VERSION)
    task_hash: str = ""
    route: str = ""
    reason: str = ""
    confidence: float = 0.0
    retrieval_similarity: float = 0.0
    max_iterations_allocated: int = 0
    stagnation_patience: int = 0
    convergence_target: float = 0.0
    escalation_threshold: float = 0.0
    cost_estimate_ms: float = 0.0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.max_iterations_allocated < 0:
            raise ValueError(f"max_iterations_allocated must be >= 0")
        if self.stagnation_patience < 0:
            raise ValueError(f"stagnation_patience must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RoutingOutcomeSchema:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════
# 3. RetrievalOutcomeSchema
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RetrievalOutcomeSchema:
    """Retrieval decision + effectiveness metrics."""
    schema_version: str = field(default=SCHEMA_VERSION)
    task_hash: str = ""
    embedding_sha256: str = ""
    topology_family: str = ""
    best_similarity: float = 0.0
    warmstart_applied: bool = False
    warmstart_source_hash: str = ""
    coldstart_iterations: int = 0
    warmstart_iterations: int = 0
    iterations_saved: int = 0
    coldstart_residual: float = 0.0
    warmstart_residual: float = 0.0
    faiss_index_size: int = 0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.coldstart_iterations < 0:
            raise ValueError(f"coldstart_iterations must be >= 0")
        if self.warmstart_iterations < 0:
            raise ValueError(f"warmstart_iterations must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RetrievalOutcomeSchema:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════
# 4. EscalationEventSchema
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EscalationEventSchema:
    """Escalation trigger + context + resolution."""
    schema_version: str = field(default=SCHEMA_VERSION)
    task_hash: str = ""
    topology_family: str = ""
    trigger_reason: str = ""  # divergence, stagnation, budget_exhausted, escalation_threshold
    trigger_iteration: int = 0
    trigger_residual: float = 0.0
    budget_allocated: int = 0
    budget_used: int = 0
    oracle_called: bool = False
    oracle_residual: float = 0.0
    resolution: str = ""  # oracle_verified, oracle_corrected, oracle_failed
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.trigger_iteration < 0:
            raise ValueError(f"trigger_iteration must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EscalationEventSchema:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════
# 5. WarmstartPerformanceSchema
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WarmstartPerformanceSchema:
    """Warmstart vs coldstart comparison metrics."""
    schema_version: str = field(default=SCHEMA_VERSION)
    topology_family: str = ""
    sample_count: int = 0
    avg_coldstart_iterations: float = 0.0
    avg_warmstart_iterations: float = 0.0
    avg_iterations_saved: float = 0.0
    warmstart_win_rate: float = 0.0
    coldstart_win_rate: float = 0.0
    tie_rate: float = 0.0
    avg_coldstart_residual: float = 0.0
    avg_warmstart_residual: float = 0.0
    statistical_significance_p: float = 1.0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError(f"sample_count must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WarmstartPerformanceSchema:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════
# 6. TopologyClusterSchema
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TopologyClusterSchema:
    """Topology family clustering metadata."""
    schema_version: str = field(default=SCHEMA_VERSION)
    family_name: str = ""
    member_count: int = 0
    avg_node_count: float = 0.0
    avg_edge_count: float = 0.0
    avg_projection_iterations: float = 0.0
    avg_final_residual: float = 0.0
    convergence_class_distribution_json: str = "{}"  # JSON dict
    failure_rate: float = 0.0
    warmstart_effectiveness: float = 0.0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.member_count < 0:
            raise ValueError(f"member_count must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TopologyClusterSchema:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════
# 7. RuntimeCostDistributionSchema
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RuntimeCostDistributionSchema:
    """Runtime cost distribution per family/route."""
    schema_version: str = field(default=SCHEMA_VERSION)
    topology_family: str = ""
    route: str = ""
    sample_count: int = 0
    avg_total_runtime_ms: float = 0.0
    avg_projection_runtime_ms: float = 0.0
    avg_oracle_runtime_ms: float = 0.0
    avg_surrogate_runtime_ms: float = 0.0
    avg_scheduler_overhead_ms: float = 0.0
    p50_runtime_ms: float = 0.0
    p90_runtime_ms: float = 0.0
    p99_runtime_ms: float = 0.0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError(f"sample_count must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RuntimeCostDistributionSchema:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════
# 8. DatasetManifestSchema
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DatasetManifestSchema:
    """Describes a full dataset export for v2.16 consumption."""
    schema_version: str = field(default=SCHEMA_VERSION)
    dataset_id: str = ""
    description: str = ""
    created_at: str = ""
    schema_types: tuple[str, ...] = ()
    total_entries: int = 0
    file_paths: tuple[str, ...] = ()
    sha256: str = ""
    seed: int = 0
    runtime_version: str = "v2.15"

    def __post_init__(self) -> None:
        if self.total_entries < 0:
            raise ValueError(f"total_entries must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_types"] = list(self.schema_types)
        d["file_paths"] = list(self.file_paths)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetManifestSchema:
        d = dict(d)
        d["schema_types"] = tuple(d.get("schema_types", []))
        d["file_paths"] = tuple(d.get("file_paths", []))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════
# 9. SchemaRegistry
# ═══════════════════════════════════════════════════════════════

class SchemaRegistry:
    """Maps schema names to classes for v2.16 dataset consumption."""

    _registry: dict[str, type] = {
        "convergence_trace": ConvergenceTraceSchema,
        "routing_outcome": RoutingOutcomeSchema,
        "retrieval_outcome": RetrievalOutcomeSchema,
        "escalation_event": EscalationEventSchema,
        "warmstart_performance": WarmstartPerformanceSchema,
        "topology_cluster": TopologyClusterSchema,
        "runtime_cost_distribution": RuntimeCostDistributionSchema,
        "dataset_manifest": DatasetManifestSchema,
    }

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._registry:
            raise KeyError(f"Unknown schema: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def all_schemas(cls) -> dict[str, type]:
        return dict(cls._registry)

    @classmethod
    def from_dict(cls, schema_name: str, d: dict[str, Any]) -> Any:
        schema_cls = cls.get(schema_name)
        return schema_cls.from_dict(d)

    @classmethod
    def validate_entry(cls, schema_name: str, d: dict[str, Any]) -> list[str]:
        """Validate a dict against a schema, returning list of errors."""
        errors = []
        schema_cls = cls.get(schema_name)
        required_fields = set(schema_cls.__dataclass_fields__.keys())
        provided_fields = set(d.keys())
        missing = required_fields - provided_fields
        if missing:
            errors.append(f"Missing fields: {missing}")
        try:
            schema_cls.from_dict(d)
        except (ValueError, TypeError) as e:
            errors.append(f"Validation error: {e}")
        return errors
