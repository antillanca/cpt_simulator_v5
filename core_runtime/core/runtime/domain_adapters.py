"""CORE Domain Adapters -- backward compatibility layer.

Adapts old circuit-specific types to the new domain SDK interfaces.
This preserves compatibility while the new architecture becomes canonical.
"""

from __future__ import annotations

from typing import Any
import warnings

from core_runtime.core.domain_sdk import DomainTaskBase
from core_runtime.core.runtime.task_runtime import RuntimeTask


def circuit_task_to_domain_task(circuit_task: Any) -> DomainTaskBase:
    """Convert an old CircuitTask/RuntimeTask to DomainTaskBase.

    This adapter preserves backward compatibility for code that
    constructs tasks using the old circuit-specific API.
    """
    from core_runtime.domains.circuits import CircuitTask

    metadata = dict(getattr(circuit_task, "metadata", {}))

    # Extract circuit-specific fields if present
    for attr in ("num_nodes", "num_edges", "topology_family"):
        val = getattr(circuit_task, attr, None)
        if val is not None and attr not in metadata:
            metadata[attr] = val

    return CircuitTask(
        task_id=getattr(circuit_task, "task_id", "unknown"),
        domain_name="circuits",
        input_artifact=getattr(circuit_task, "input_artifact", ""),
        metadata=metadata,
    )


def circuit_graph_to_domain_task(graph: Any) -> DomainTaskBase:
    """Convert a CanonicalCircuitGraph to a DomainTaskBase."""
    from core_runtime.domains.circuits import CircuitTask

    return CircuitTask(
        task_id=getattr(graph, "graph_id", "unknown"),
        domain_name="circuits",
        input_artifact=getattr(graph, "fingerprint", ""),
        metadata={
            "num_nodes": getattr(graph, "num_nodes", 0),
            "num_edges": getattr(graph, "num_edges", 0),
            "topology_family": str(getattr(graph, "topology_family", "unknown")),
            "cycle_count": getattr(graph, "cycle_count", 0),
            "connected_components": getattr(graph, "connected_components", 1),
        },
    )


def adapt_old_runtime_task(old_task: Any) -> RuntimeTask:
    """Adapt an old backend.core_runtime.task_runtime.RuntimeTask to the new RuntimeTask.

    This is the primary compatibility adapter. It allows existing code
    that constructs old-style RuntimeTask objects to work with the new
    domain-agnostic runtime.
    """
    domain_task = circuit_task_to_domain_task(old_task)
    return RuntimeTask(
        task_id=old_task.task_id,
        domain_name=getattr(old_task, "domain", "circuits"),
        task=domain_task,
        oracle_name=getattr(old_task, "oracle_name", ""),
        surrogate_name=getattr(old_task, "surrogate_name", ""),
        projection_enabled=getattr(old_task, "projection_enabled", True),
        metadata=dict(getattr(old_task, "metadata", {})),
    )


# ============================================================
# Oracle/Surrogate/Projection adapters
# ============================================================

class OracleAdapter:
    """Wraps an old-style oracle function as a DomainOracle."""

    def __init__(self, solve_fn):
        self._solve_fn = solve_fn

    def solve(self, task: DomainTaskBase) -> dict[str, Any]:
        return self._solve_fn(task)


class SurrogateAdapter:
    """Wraps an old-style surrogate model as a DomainSurrogate."""

    def __init__(self, predict_fn):
        self._predict_fn = predict_fn

    def predict(self, task: DomainTaskBase) -> dict[str, Any]:
        return self._predict_fn(task)


class ProjectionAdapter:
    """Wraps an old-style projection as a DomainProjection."""

    def __init__(self, project_fn):
        self._project_fn = project_fn

    def project(self, task: DomainTaskBase, prediction: dict, budget: int) -> dict[str, Any]:
        return self._project_fn(task, prediction, budget)
