"""CPT Core Runtime — Execution Policy & Runtime Recovery.

Defines ExecutionPolicy and implements retry/fallback/degradation logic.
NO silent failures — every failure is recorded as a trace event.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Any

from backend.core_runtime.task_runtime import RuntimeTask, RuntimeResult


# ---------------------------------------------------------------------------
# ExecutionPolicy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionPolicy:
    """Immutable execution policy governing retries, timeouts, and fallbacks.

    Attributes:
        oracle_timeout_s:          Max seconds for oracle solve before timeout.
        max_retries:               Max retries for any failing step.
        fallback_to_cache:         Use exact cache on oracle timeout/failure.
        projection_budget_high:    Max iterations for OOD / low-confidence tasks.
        projection_budget_low:     Max iterations for high-confidence tasks.
        nan_is_degraded:           Treat NaN outputs as degraded (not error).
        surrogate_instability_threshold: Max relative error before marking unstable.
    """
    oracle_timeout_s: float = 30.0
    max_retries: int = 2
    fallback_to_cache: bool = True
    projection_budget_high: int = 20
    projection_budget_low: int = 5
    nan_is_degraded: bool = True
    surrogate_instability_threshold: float = 100.0


# ---------------------------------------------------------------------------
# Degradation reasons
# ---------------------------------------------------------------------------

DEGRADED_ORACLE_TIMEOUT = "oracle_timeout"
DEGRADED_PROJECTION_DIVERGENCE = "projection_divergence"
DEGRADED_NAN_OUTPUT = "nan_output"
DEGRADED_SURROGATE_INSTABILITY = "surrogate_instability"
DEGRADED_CACHE_FALLBACK = "cache_fallback"


# ---------------------------------------------------------------------------
# Recovery handler — classifies and recovers from runtime failures
# ---------------------------------------------------------------------------

class RecoveryHandler:
    """Handles runtime failures: retry, fallback, degrade.

    Never raises silently — always produces a RuntimeResult (possibly degraded).
    """

    def __init__(self, policy: ExecutionPolicy | None = None) -> None:
        self._policy = policy or ExecutionPolicy()
        self._events: list[dict[str, Any]] = []

    @property
    def policy(self) -> ExecutionPolicy:
        return self._policy

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def check_oracle_timeout(
        self,
        oracle_ms: float,
        task: RuntimeTask,
    ) -> str | None:
        """Check if oracle exceeded timeout. Returns degradation reason or None."""
        if oracle_ms > self._policy.oracle_timeout_s * 1000.0:
            reason = DEGRADED_ORACLE_TIMEOUT
            self._record_event(task.task_id, reason, {"oracle_ms": oracle_ms})
            return reason
        return None

    def check_nan_output(
        self,
        values: Any,
        label: str,
        task: RuntimeTask,
    ) -> str | None:
        """Check if output contains NaN. Returns degradation reason or None."""
        import torch
        if isinstance(values, torch.Tensor) and torch.isnan(values).any():
            reason = DEGRADED_NAN_OUTPUT
            self._record_event(task.task_id, reason, {"label": label})
            return reason
        if isinstance(values, (list, tuple)):
            import math
            if any(isinstance(v, float) and math.isnan(v) for v in values):
                reason = DEGRADED_NAN_OUTPUT
                self._record_event(task.task_id, reason, {"label": label})
                return reason
        return None

    def check_surrogate_instability(
        self,
        surrogate_output: Any,
        oracle_output: Any,
        task: RuntimeTask,
    ) -> str | None:
        """Check if surrogate is wildly off (relative error > threshold)."""
        import torch
        if surrogate_output is None or oracle_output is None:
            return None
        surr = surrogate_output if isinstance(surrogate_output, torch.Tensor) else None
        oracle = oracle_output if isinstance(oracle_output, torch.Tensor) else None
        if surr is None or oracle is None:
            return None
        if oracle.abs().sum() == 0:
            return None
        rel_error = (surr - oracle).abs().sum() / oracle.abs().sum()
        if rel_error > self._policy.surrogate_instability_threshold:
            reason = DEGRADED_SURROGATE_INSTABILITY
            self._record_event(task.task_id, reason, {"rel_error": float(rel_error)})
            return reason
        return None

    def check_projection_divergence(
        self,
        projection_iterations: int,
        max_budget: int,
        converged: bool,
        task: RuntimeTask,
    ) -> str | None:
        """Check if projection diverged (hit budget without convergence)."""
        if not converged and projection_iterations >= max_budget:
            reason = DEGRADED_PROJECTION_DIVERGENCE
            self._record_event(task.task_id, reason, {
                "iterations": projection_iterations,
                "budget": max_budget,
            })
            return reason
        return None

    def record_cache_fallback(self, task_id: str, task_hash: str) -> None:
        """Record that a cache fallback was used."""
        self._record_event(task_id, DEGRADED_CACHE_FALLBACK, {"task_hash": task_hash})

    def make_degraded_result(
        self,
        task: RuntimeTask,
        reason: str,
        oracle_voltages: Any = None,
        surrogate_voltages: Any = None,
        total_runtime_ms: float = 0.0,
        oracle_runtime_ms: float = 0.0,
        surrogate_runtime_ms: float = 0.0,
        projection_runtime_ms: float = 0.0,
    ) -> RuntimeResult:
        """Create a degraded RuntimeResult with failure_type set."""
        return RuntimeResult(
            task_id=task.task_id,
            task_fingerprint=task.fingerprint(),
            oracle_voltages=oracle_voltages,
            surrogate_voltages=surrogate_voltages,
            projected_voltages=None,
            projection_result=None,
            evaluation_report=None,
            memory_entry=None,
            total_runtime_ms=total_runtime_ms,
            oracle_runtime_ms=oracle_runtime_ms,
            surrogate_runtime_ms=surrogate_runtime_ms,
            projection_runtime_ms=projection_runtime_ms,
            failure_type=reason,
            metadata={"degraded": True, "degradation_reason": reason},
        )

    def clear_events(self) -> None:
        self._events.clear()

    # -- internal -----------------------------------------------------------

    def _record_event(self, task_id: str, reason: str, details: dict) -> None:
        self._events.append({
            "task_id": task_id,
            "reason": reason,
            "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
            **details,
        })
