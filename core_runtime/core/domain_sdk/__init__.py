"""CORE Domain SDK -- Canonical interfaces for domain integration.

Every domain that plugs into CORE must implement these protocols.
The core runtime ONLY knows these interfaces, never domain-specific
types like CircuitGraph or MNA.

Usage:
    from core_runtime.core.domain_sdk import (
        DomainTask, DomainOracle, DomainSurrogate,
        DomainProjection, DomainEvaluator, DomainTaskBase,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ============================================================
# Domain Task -- the universal unit of work
# ============================================================

@dataclass(frozen=True)
class DomainTaskBase:
    """Base dataclass for all domain tasks.

    Every domain task MUST have at minimum these fields.
    Domains may subclass to add domain-specific fields.
    """
    task_id: str
    domain_name: str
    input_artifact: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DomainTask(Protocol):
    """Protocol for domain tasks.

    Any object with these attributes and methods satisfies
    the DomainTask protocol and can be executed by the runtime.
    """
    task_id: str
    domain_name: str
    input_artifact: str
    metadata: dict[str, Any]

    def fingerprint(self) -> str:
        """Deterministic fingerprint for cache lookups."""
        ...

    def node_count(self) -> int:
        """Graph/problem size metric (nodes)."""
        ...

    def edge_count(self) -> int:
        """Graph/problem size metric (edges/constraints)."""
        ...


# ============================================================
# Domain Oracle -- ground-truth solver
# ============================================================

@runtime_checkable
class DomainOracle(Protocol):
    """Protocol for domain oracles.

    An oracle provides the ground-truth solution for a task.
    It is always correct but may be slow or expensive.
    """

    def solve(self, task: DomainTaskBase) -> dict[str, Any]:
        """Solve the task and return a solution dictionary.

        The returned dict must contain at minimum:
          - 'solution': the ground-truth solution
          - 'residual': scalar residual after solving
          - 'runtime_ms': wall-clock solve time
        """
        ...


# ============================================================
# Domain Surrogate -- fast approximate solver
# ============================================================

@runtime_checkable
class DomainSurrogate(Protocol):
    """Protocol for domain surrogates.

    A surrogate provides a fast approximate solution.
    It may be wrong; projection corrects it.
    """

    def predict(self, task: DomainTaskBase) -> dict[str, Any]:
        """Predict an approximate solution.

        The returned dict must contain at minimum:
          - 'prediction': the approximate solution
          - 'residual': scalar residual before projection
          - 'runtime_ms': wall-clock predict time
        """
        ...


# ============================================================
# Domain Projection -- correctness enforcement
# ============================================================

@runtime_checkable
class DomainProjection(Protocol):
    """Protocol for domain projection operators.

    Projection takes a surrogate prediction and enforces
    domain-specific correctness constraints (e.g., KCL/KVL
    for circuits, residual bounds for linear systems).

    Projection is ALWAYS the final authority.
    """

    def project(
        self, task: DomainTaskBase, prediction: dict[str, Any], budget: int,
    ) -> dict[str, Any]:
        """Project a prediction toward correctness.

        Args:
            task: The domain task
            prediction: Surrogate prediction dict
            budget: Maximum projection iterations

        Returns dict with at minimum:
          - 'solution': the projected solution
          - 'residual': scalar residual after projection
          - 'iterations': actual iterations used
          - 'converged': whether projection converged
          - 'runtime_ms': wall-clock projection time
        """
        ...


# ============================================================
# Domain Evaluator -- quality measurement
# ============================================================

@runtime_checkable
class DomainEvaluator(Protocol):
    """Protocol for domain evaluators.

    An evaluator measures the quality of a solution against
    domain-specific criteria.
    """

    def evaluate(
        self, task: DomainTaskBase, solution: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a solution.

        Returns dict with at minimum:
          - 'residual': primary quality metric
          - 'correct': whether solution meets acceptance threshold
          - 'metrics': dict of domain-specific quality metrics
        """
        ...


# ============================================================
# Domain Confidence -- optional per-domain confidence scoring
# ============================================================

@runtime_checkable
class DomainConfidence(Protocol):
    """Protocol for domain confidence scoring.

    Optional: domains can implement this to provide confidence
    signals for the scheduler. If not provided, the scheduler
    uses default confidence estimation.
    """

    def score(self, task: DomainTaskBase, prediction: dict[str, Any]) -> float:
        """Return a confidence score in [0, 1].

        Higher = more confident the surrogate prediction is close.
        """
        ...


# ============================================================
# Domain Registry -- registration of domain components
# ============================================================

_DOMAIN_REGISTRY: dict[str, dict[str, type]] = {}


def register_domain(
    domain_name: str,
    oracle: type | None = None,
    surrogate: type | None = None,
    projection: type | None = None,
    evaluator: type | None = None,
    confidence: type | None = None,
) -> None:
    """Register domain components in the global registry."""
    if domain_name not in _DOMAIN_REGISTRY:
        _DOMAIN_REGISTRY[domain_name] = {}
    entry = _DOMAIN_REGISTRY[domain_name]
    if oracle is not None:
        entry["oracle"] = oracle
    if surrogate is not None:
        entry["surrogate"] = surrogate
    if projection is not None:
        entry["projection"] = projection
    if evaluator is not None:
        entry["evaluator"] = evaluator
    if confidence is not None:
        entry["confidence"] = confidence


def get_domain_components(domain_name: str) -> dict[str, type]:
    """Get registered components for a domain."""
    return _DOMAIN_REGISTRY.get(domain_name, {})


def list_domains() -> list[str]:
    """List all registered domain names."""
    return sorted(_DOMAIN_REGISTRY.keys())
