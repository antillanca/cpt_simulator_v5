"""CPT Circuit Domain -- first validated domain for CORE.

This package contains all circuit-specific logic: MNA oracle,
KCL/KVL projection, CircuitGraph surrogate, and confidence
scoring based on KCL residuals.

Circuit domain version: 2.15.0
Carries forward the frozen v2.15 runtime lineage.
"""

from __future__ import annotations

from core_runtime.core.domain_sdk import (
    DomainTaskBase,
    DomainOracle,
    DomainSurrogate,
    DomainProjection,
    DomainEvaluator,
    DomainConfidence,
    register_domain,
)

# Circuit-specific types (re-exported for compatibility)
from core_runtime.domains.circuits.graph_spec import (
    CanonicalCircuitGraph,
    TopologyFamily,
)
from core_runtime.domains.circuits.oracle_adapter import MNAOracleAdapter
from core_runtime.domains.circuits.projection_runtime import PhysicsProjection
from core_runtime.domains.circuits.confidence_runtime import ConfidenceRuntime as CircuitConfidence

__all__ = [
    "CanonicalCircuitGraph",
    "TopologyFamily",
    "MNAOracleAdapter",
    "PhysicsProjection",
    "CircuitConfidence",
    "CircuitTask",
]


# ============================================================
# Circuit-specific DomainTask
# ============================================================

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CircuitTask(DomainTaskBase):
    """Circuit domain task -- extends DomainTaskBase with circuit fields."""
    domain_name: str = "circuits"

    def fingerprint(self) -> str:
        """Deterministic fingerprint for cache lookups."""
        from core_runtime.core.specs.task_hashing import deterministic_hash
        return deterministic_hash(self.input_artifact, self.metadata)

    def node_count(self) -> int:
        return self.metadata.get("num_nodes", 0)

    def edge_count(self) -> int:
        return self.metadata.get("num_edges", 0)


# Register circuit domain
register_domain(
    domain_name="circuits",
    oracle=MNAOracleAdapter,
    projection=PhysicsProjection,
    confidence=CircuitConfidence,
)

__version__ = "2.15.0"
