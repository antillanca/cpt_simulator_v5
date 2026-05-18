"""CPT Core Runtime — Task Specification & Execution Engine.

Defines RuntimeTask and RuntimeExecutor: the canonical execution pipeline
that orchestrates oracle → surrogate → projection → evaluation → memory
for ANY domain (circuits, KiCad, FreeCAD, mathematics, logic, language).

Uses frozen contracts from backend.core_spec.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from backend.core_spec.graph_spec import CanonicalCircuitGraph
from backend.core_spec.projection_spec import ProjectionResult
from backend.core_spec.report_spec import EvaluationReport
from backend.core_spec.memory_spec import MemoryEntry


# ---------------------------------------------------------------------------
# RuntimeTask — canonical task specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeTask:
    """Immutable task specification for the execution runtime.

    domain: 'circuit', 'kicad', 'freecad', 'math', 'logic', 'language', etc.
    input_artifact: fingerprint or path of the input graph/circuit.
    """

    task_id: str
    domain: str
    input_artifact: str
    oracle_name: str
    surrogate_name: str
    projection_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint of the task spec."""
        import hashlib, json
        blob = json.dumps({
            "task_id": self.task_id,
            "domain": self.domain,
            "input_artifact": self.input_artifact,
            "oracle_name": self.oracle_name,
            "surrogate_name": self.surrogate_name,
            "projection_enabled": self.projection_enabled,
            "metadata": _sorted(self.metadata),
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# RuntimeResult — canonical execution result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeResult:
    """Immutable result from RuntimeExecutor.execute()."""

    task_id: str
    task_fingerprint: str
    oracle_voltages: Any | None  # torch.Tensor or domain-specific
    surrogate_voltages: Any | None
    projected_voltages: Any | None
    projection_result: ProjectionResult | None
    evaluation_report: EvaluationReport | None
    memory_entry: MemoryEntry | None
    total_runtime_ms: float
    oracle_runtime_ms: float
    surrogate_runtime_ms: float
    projection_runtime_ms: float
    failure_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OracleProtocol — generic oracle interface
# ---------------------------------------------------------------------------

@runtime_checkable
class OracleProtocol(Protocol):
    """Any oracle that solves a domain problem exactly."""

    def solve(self, graph: Any) -> dict[str, Any]:
        """Return exact solution as a dict with 'voltages' key (or domain equivalent)."""
        ...

    def name(self) -> str:
        """Return canonical oracle identifier."""
        ...


# ---------------------------------------------------------------------------
# SurrogateProtocol — generic surrogate interface
# ---------------------------------------------------------------------------

@runtime_checkable
class SurrogateProtocol(Protocol):
    """Any surrogate model that approximates an oracle."""

    def predict(self, graph: Any) -> Any:
        """Return approximate prediction (tensor or domain-specific)."""
        ...

    def name(self) -> str:
        """Return canonical surrogate identifier."""
        ...


# ---------------------------------------------------------------------------
# ProjectionProtocol — generic projection interface
# ---------------------------------------------------------------------------

@runtime_checkable
class ProjectionProtocol(Protocol):
    """Any physics/constraint projection layer."""

    def project(self, graph: Any, context: Any, prediction: Any) -> Any:
        """Return corrected prediction."""
        ...

    def name(self) -> str:
        """Return canonical projection identifier."""
        ...


# ---------------------------------------------------------------------------
# EvaluatorProtocol — generic evaluation interface
# ---------------------------------------------------------------------------

@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Any evaluator that measures quality of a prediction."""

    def evaluate(
        self,
        graph: Any,
        oracle_output: dict[str, Any],
        surrogate_output: Any,
        projected_output: Any | None,
    ) -> EvaluationReport:
        ...


# ---------------------------------------------------------------------------
# RuntimeExecutor — orchestrates the full pipeline
# ---------------------------------------------------------------------------

class RuntimeExecutor:
    """Canonical execution engine: oracle → surrogate → projection → eval → memory.

    Domain-agnostic: inject oracle/surrogate/projection/evaluator via constructor.
    """

    def __init__(
        self,
        oracle: OracleProtocol,
        surrogate: SurrogateProtocol,
        projection: ProjectionProtocol | None = None,
        evaluator: EvaluatorProtocol | None = None,
        memory_sink: Any | None = None,  # MemoryRuntime or similar
    ) -> None:
        self._oracle = oracle
        self._surrogate = surrogate
        self._projection = projection
        self._evaluator = evaluator
        self._memory_sink = memory_sink

    def execute(self, task: RuntimeTask) -> RuntimeResult:
        """Execute the full pipeline for a single task.

        Order: Load → Oracle → Surrogate → Projection → Evaluate → Memory.
        """
        t0 = time.perf_counter()

        # 1. Oracle
        t_oracle = time.perf_counter()
        oracle_output = self._oracle.solve(task)
        oracle_ms = (time.perf_counter() - t_oracle) * 1000.0

        # 2. Surrogate
        t_surr = time.perf_counter()
        surrogate_output = self._surrogate.predict(task)
        surrogate_ms = (time.perf_counter() - t_surr) * 1000.0

        # 3. Optional Projection
        projected_output = None
        projection_result = None
        projection_ms = 0.0
        if task.projection_enabled and self._projection is not None:
            t_proj = time.perf_counter()
            projected_output = self._projection.project(task, oracle_output, surrogate_output)
            projection_ms = (time.perf_counter() - t_proj) * 1000.0

        # 4. Optional Evaluation
        eval_report = None
        if self._evaluator is not None:
            eval_report = self._evaluator.evaluate(
                task, oracle_output, surrogate_output, projected_output
            )

        # 5. Optional Memory Registration
        memory_entry = None
        if self._memory_sink is not None:
            memory_entry = self._memory_sink.register_execution(
                task=task,
                oracle_output=oracle_output,
                surrogate_output=surrogate_output,
                projected_output=projected_output,
                oracle_ms=oracle_ms,
                surrogate_ms=surrogate_ms,
                projection_ms=projection_ms,
            )

        total_ms = (time.perf_counter() - t0) * 1000.0

        return RuntimeResult(
            task_id=task.task_id,
            task_fingerprint=task.fingerprint(),
            oracle_voltages=oracle_output.get("voltages") if isinstance(oracle_output, dict) else oracle_output,
            surrogate_voltages=surrogate_output,
            projected_voltages=projected_output,
            projection_result=projection_result,
            evaluation_report=eval_report,
            memory_entry=memory_entry,
            total_runtime_ms=total_ms,
            oracle_runtime_ms=oracle_ms,
            surrogate_runtime_ms=surrogate_ms,
            projection_runtime_ms=projection_ms,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted(d: dict) -> dict:
    """Recursively sort dict keys for deterministic serialization."""
    if not isinstance(d, dict):
        return d
    return {k: _sorted(v) if isinstance(v, dict) else v for k, v in sorted(d.items())}
