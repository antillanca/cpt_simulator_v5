"""CPT Runtime — Execution Scheduler (v2.15).

Coordinates the full execution pipeline:
1. exact cache check (bypasses scheduler entirely)
2. retrieval memory lookup
3. confidence estimation
4. cost estimation
5. scheduler budget allocation
6. warmstart evaluation
7. projection execution
8. escalation/fallback
9. trace commit

The scheduler outputs:
- selected budget
- route taken
- reason for route
- estimated cost
- final execution status

ALL outputs must be traceable and deterministic.
Exact cache ALWAYS bypasses the scheduler.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Any

from backend.core_runtime.confidence_runtime import ConfidenceEstimate, ConfidenceRuntime
from backend.core_runtime.execution_policy import ExecutionPolicy
from backend.core_runtime.task_runtime import RuntimeTask
from backend.runtime.cost_estimator import CostEstimator, ExecutionCostEstimate
from backend.runtime.projection_scheduler import (
    ProjectionBudget,
    ProjectionScheduler,
    StopDecision,
    TRAJECTORY_FAST_CONVERGING,
    TRAJECTORY_STABLE_LINEAR,
    TRAJECTORY_OSCILLATORY,
    TRAJECTORY_STALLED,
    TRAJECTORY_DIVERGENCE_RISK,
    TRAJECTORY_RETRIEVAL_ASSISTED,
    STOP_CONTINUE,
    STOP_CONVERGED,
    STOP_STAGNATED,
    STOP_DIMINISHING,
    STOP_DIVERGENCE,
    STOP_ESCALATE,
    STOP_BUDGET_EXHAUSTED,
)
from backend.runtime.trajectory_analysis import TrajectoryAnalyzer, TrajectoryAnalysisResult


# ---------------------------------------------------------------------------
# ExecutionRoute (v2.15)
# ---------------------------------------------------------------------------

ROUTE_CACHE_HIT = "cache_hit"
ROUTE_RETRIEVAL_WARMSTART = "retrieval_warmstart"
ROUTE_RETRIEVAL_SEMANTIC = "retrieval_semantic"
ROUTE_STANDARD = "standard"
ROUTE_OOD_ESCALATED = "ood_escalated"
ROUTE_ORACLE_FORCED = "oracle_forced"
ROUTE_DEGRADED = "degraded"

VALID_ROUTES = frozenset({
    ROUTE_CACHE_HIT,
    ROUTE_RETRIEVAL_WARMSTART,
    ROUTE_RETRIEVAL_SEMANTIC,
    ROUTE_STANDARD,
    ROUTE_OOD_ESCALATED,
    ROUTE_ORACLE_FORCED,
    ROUTE_DEGRADED,
})


# ---------------------------------------------------------------------------
# ExecutionSchedule (v2.15)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionSchedule:
    """Complete execution schedule produced by the scheduler.

    Attributes:
        route: The execution route selected.
        reason: Human-readable explanation of the route choice.
        budget: Allocated projection budget (None for cache hits).
        cost_estimate: Estimated execution cost.
        confidence: Confidence estimate used.
        retrieval_similarity: Retrieval similarity score (0 if no retrieval).
        force_oracle: Whether oracle verification is forced.
        estimated_total_runtime_ms: Estimated total runtime.
    """

    route: str
    reason: str
    budget: ProjectionBudget | None
    cost_estimate: ExecutionCostEstimate | None
    confidence: ConfidenceEstimate | None
    retrieval_similarity: float
    force_oracle: bool
    estimated_total_runtime_ms: float

    def __post_init__(self) -> None:
        if self.route not in VALID_ROUTES:
            raise ValueError(
                f"Invalid route: {self.route}, expected one of {VALID_ROUTES}"
            )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "reason": self.reason,
            "budget": self.budget.to_json_dict() if self.budget else None,
            "cost_estimate": self.cost_estimate.to_json_dict() if self.cost_estimate else None,
            "confidence": {
                "score": round(self.confidence.confidence_score, 8),
                "estimated_iterations": self.confidence.estimated_projection_iterations,
                "likely_ood": self.confidence.likely_ood,
            } if self.confidence else None,
            "retrieval_similarity": round(self.retrieval_similarity, 8),
            "force_oracle": self.force_oracle,
            "estimated_total_runtime_ms": round(self.estimated_total_runtime_ms, 3),
        }


# ---------------------------------------------------------------------------
# ExecutionOutcome (v2.15)
# ---------------------------------------------------------------------------

OUTCOME_SUCCESS = "success"
OUTCOME_CONVERGED_EARLY = "converged_early"
OUTCOME_STAGNATED = "stagnated"
OUTCOME_DIVERGED = "diverged"
OUTCOME_ESCALATED = "escalated"
OUTCOME_BUDGET_EXHAUSTED = "budget_exhausted"
OUTCOME_DEGRADED = "degraded"
OUTCOME_CACHE_HIT = "cache_hit"

VALID_OUTCOMES = frozenset({
    OUTCOME_SUCCESS,
    OUTCOME_CONVERGED_EARLY,
    OUTCOME_STAGNATED,
    OUTCOME_DIVERGED,
    OUTCOME_ESCALATED,
    OUTCOME_BUDGET_EXHAUSTED,
    OUTCOME_DEGRADED,
    OUTCOME_CACHE_HIT,
})


@dataclass(frozen=True)
class ExecutionOutcome:
    """Final outcome of a scheduled execution.

    Attributes:
        outcome: The execution outcome.
        iterations_used: Actual projection iterations used.
        iterations_allocated: Budget allocated (max_iterations).
        iterations_saved: iterations_allocated - iterations_used.
        trajectory_class: Classified trajectory.
        stop_reason: Why projection stopped (if applicable).
        final_residual: Final residual after projection.
        warmstart_used: Whether warmstart was applied.
        retrieval_used: Whether retrieval was used.
        runtime_ms: Actual total runtime.
        scheduler_overhead_ms: Time spent in scheduler logic.
    """

    outcome: str
    iterations_used: int
    iterations_allocated: int
    iterations_saved: int
    trajectory_class: str
    stop_reason: str | None
    final_residual: float
    warmstart_used: bool
    retrieval_used: bool
    runtime_ms: float
    scheduler_overhead_ms: float

    def __post_init__(self) -> None:
        if self.outcome not in VALID_OUTCOMES:
            raise ValueError(
                f"Invalid outcome: {self.outcome}, expected one of {VALID_OUTCOMES}"
            )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "iterations_used": self.iterations_used,
            "iterations_allocated": self.iterations_allocated,
            "iterations_saved": self.iterations_saved,
            "trajectory_class": self.trajectory_class,
            "stop_reason": self.stop_reason,
            "final_residual": round(self.final_residual, 12),
            "warmstart_used": self.warmstart_used,
            "retrieval_used": self.retrieval_used,
            "runtime_ms": round(self.runtime_ms, 3),
            "scheduler_overhead_ms": round(self.scheduler_overhead_ms, 3),
        }


# ---------------------------------------------------------------------------
# ExecutionScheduler (v2.15)
# ---------------------------------------------------------------------------

class ExecutionScheduler:
    """Deterministic execution scheduler.

    Coordinates: cache → retrieval → confidence → cost → budget →
                 warmstart → projection → escalation → trace.

    Exact cache ALWAYS bypasses the scheduler entirely.
    Projection remains the final authority.
    The scheduler only decides effort allocation and when to stop/escalate.

    DETERMINISTIC: same inputs → same schedule. Always.
    """

    def __init__(
        self,
        policy: ExecutionPolicy | None = None,
        confidence_runtime: ConfidenceRuntime | None = None,
        cost_estimator: CostEstimator | None = None,
        projection_scheduler: ProjectionScheduler | None = None,
        trajectory_analyzer: TrajectoryAnalyzer | None = None,
        family_stats: dict[str, dict[str, Any]] | None = None,
        failure_counts: dict[str, int] | None = None,
        min_similarity_warmstart: float = 0.5,
    ) -> None:
        self._policy = policy or ExecutionPolicy()
        self._confidence_rt = confidence_runtime or ConfidenceRuntime()
        self._cost_estimator = cost_estimator or CostEstimator()
        self._proj_scheduler = projection_scheduler or ProjectionScheduler(
            family_stats=family_stats,
        )
        self._trajectory_analyzer = trajectory_analyzer or TrajectoryAnalyzer()
        self._family_stats = family_stats or {}
        self._failure_counts = failure_counts or {}
        self._min_similarity = min_similarity_warmstart

    # -- Schedule computation ------------------------------------------------

    def schedule(
        self,
        task: RuntimeTask,
        cache_hit: bool = False,
        retrieval_similarity: float = 0.0,
        is_degraded: bool = False,
        node_count: int = 0,
        edge_count: int = 0,
        current_sources: int = 0,
        resistance_range: tuple[float, float] = (1.0, 1.0),
    ) -> ExecutionSchedule:
        """Compute a deterministic execution schedule.

        Priority:
        1. Exact cache hit → bypass everything
        2. Degraded → failure path
        3. Repeated failure topology → oracle forced
        4. OOD detection → escalated budget
        5. Retrieval warmstart → reduced budget
        6. Retrieval semantic → standard budget
        7. High confidence → small budget
        8. Default → standard budget
        """
        t_start = _time.monotonic()

        topo = task.metadata.get("topology_family", "unknown")

        # 1. Cache hit: bypass scheduler entirely
        if cache_hit:
            return ExecutionSchedule(
                route=ROUTE_CACHE_HIT,
                reason="Exact cache hit — scheduler bypassed",
                budget=None,
                cost_estimate=None,
                confidence=None,
                retrieval_similarity=0.0,
                force_oracle=False,
                estimated_total_runtime_ms=0.0,
            )

        # 2. Confidence estimation
        confidence = self._confidence_rt.estimate(
            task,
            graph_size=node_count,
            topology_family=topo,
        )

        # 3. Cost estimation
        cost_estimate = self._cost_estimator.estimate(
            node_count=node_count,
            edge_count=edge_count,
            topology_family=topo,
            current_sources=current_sources,
            resistance_range=resistance_range,
            likely_ood=confidence.likely_ood,
            confidence=confidence.confidence_score,
        )

        # 4. Degraded path
        if is_degraded:
            budget = self._proj_scheduler.allocate_budget(
                confidence=confidence,
                cost_estimate=cost_estimate,
                retrieval_similarity=0.0,
                topology_family=topo,
                is_ood=True,
                node_count=node_count,
            )
            return ExecutionSchedule(
                route=ROUTE_DEGRADED,
                reason="Degraded execution — forcing oracle",
                budget=budget,
                cost_estimate=cost_estimate,
                confidence=confidence,
                retrieval_similarity=0.0,
                force_oracle=True,
                estimated_total_runtime_ms=cost_estimate.estimated_runtime_ms,
            )

        # 5. Repeated failure topology → oracle forced
        failure_count = self._failure_counts.get(topo, 0)
        if failure_count >= 3:
            budget = self._proj_scheduler.allocate_budget(
                confidence=confidence,
                cost_estimate=cost_estimate,
                retrieval_similarity=retrieval_similarity,
                topology_family=topo,
                is_ood=True,
                node_count=node_count,
            )
            return ExecutionSchedule(
                route=ROUTE_ORACLE_FORCED,
                reason=f"Topology '{topo}' has {failure_count} failures — oracle forced",
                budget=budget,
                cost_estimate=cost_estimate,
                confidence=confidence,
                retrieval_similarity=retrieval_similarity,
                force_oracle=True,
                estimated_total_runtime_ms=cost_estimate.estimated_runtime_ms,
            )

        # 6. OOD detection
        if confidence.likely_ood:
            budget = self._proj_scheduler.allocate_budget(
                confidence=confidence,
                cost_estimate=cost_estimate,
                retrieval_similarity=retrieval_similarity,
                topology_family=topo,
                is_ood=True,
                node_count=node_count,
            )
            force_oracle = True
            if retrieval_similarity >= self._min_similarity:
                route = ROUTE_RETRIEVAL_WARMSTART
                reason = f"OOD + warmstart (sim={retrieval_similarity:.3f})"
            else:
                route = ROUTE_OOD_ESCALATED
                reason = f"OOD (conf={confidence.confidence_score:.3f})"
            return ExecutionSchedule(
                route=route,
                reason=reason,
                budget=budget,
                cost_estimate=cost_estimate,
                confidence=confidence,
                retrieval_similarity=retrieval_similarity,
                force_oracle=force_oracle,
                estimated_total_runtime_ms=cost_estimate.estimated_runtime_ms,
            )

        # 7. Retrieval warmstart
        if retrieval_similarity >= self._min_similarity:
            budget = self._proj_scheduler.allocate_budget(
                confidence=confidence,
                cost_estimate=cost_estimate,
                retrieval_similarity=retrieval_similarity,
                topology_family=topo,
                is_ood=False,
                node_count=node_count,
            )
            return ExecutionSchedule(
                route=ROUTE_RETRIEVAL_WARMSTART,
                reason=f"Warmstart available (sim={retrieval_similarity:.3f})",
                budget=budget,
                cost_estimate=cost_estimate,
                confidence=confidence,
                retrieval_similarity=retrieval_similarity,
                force_oracle=False,
                estimated_total_runtime_ms=cost_estimate.estimated_runtime_ms * 0.7,
            )

        # 8. Retrieval semantic (partial match)
        if retrieval_similarity >= 0.3:
            budget = self._proj_scheduler.allocate_budget(
                confidence=confidence,
                cost_estimate=cost_estimate,
                retrieval_similarity=retrieval_similarity,
                topology_family=topo,
                is_ood=False,
                node_count=node_count,
            )
            return ExecutionSchedule(
                route=ROUTE_RETRIEVAL_SEMANTIC,
                reason=f"Partial retrieval (sim={retrieval_similarity:.3f})",
                budget=budget,
                cost_estimate=cost_estimate,
                confidence=confidence,
                retrieval_similarity=retrieval_similarity,
                force_oracle=False,
                estimated_total_runtime_ms=cost_estimate.estimated_runtime_ms,
            )

        # 9. Standard
        budget = self._proj_scheduler.allocate_budget(
            confidence=confidence,
            cost_estimate=cost_estimate,
            retrieval_similarity=0.0,
            topology_family=topo,
            is_ood=False,
            node_count=node_count,
        )
        return ExecutionSchedule(
            route=ROUTE_STANDARD,
            reason=f"Standard (conf={confidence.confidence_score:.3f})",
            budget=budget,
            cost_estimate=cost_estimate,
            confidence=confidence,
            retrieval_similarity=0.0,
            force_oracle=False,
            estimated_total_runtime_ms=cost_estimate.estimated_runtime_ms,
        )

    # -- Outcome computation -------------------------------------------------

    def compute_outcome(
        self,
        schedule: ExecutionSchedule,
        iterations_used: int,
        final_residual: float,
        residual_history: list[float],
        warmstart_used: bool = False,
        was_degraded: bool = False,
        runtime_ms: float = 0.0,
        scheduler_overhead_ms: float = 0.0,
    ) -> ExecutionOutcome:
        """Compute the final execution outcome from actual results.

        Maps actual execution results to a canonical outcome.
        DETERMINISTIC: same inputs → same outcome.
        """
        # Cache hit special case
        if schedule.route == ROUTE_CACHE_HIT:
            return ExecutionOutcome(
                outcome=OUTCOME_CACHE_HIT,
                iterations_used=0,
                iterations_allocated=0,
                iterations_saved=0,
                trajectory_class=TRAJECTORY_STABLE_LINEAR,
                stop_reason=None,
                final_residual=0.0,
                warmstart_used=False,
                retrieval_used=False,
                runtime_ms=runtime_ms,
                scheduler_overhead_ms=scheduler_overhead_ms,
            )

        # Degraded
        if was_degraded:
            return ExecutionOutcome(
                outcome=OUTCOME_DEGRADED,
                iterations_used=iterations_used,
                iterations_allocated=schedule.budget.max_iterations if schedule.budget else 0,
                iterations_saved=0,
                trajectory_class=TRAJECTORY_DIVERGENCE_RISK,
                stop_reason=STOP_DIVERGENCE,
                final_residual=final_residual,
                warmstart_used=warmstart_used,
                retrieval_used=schedule.retrieval_similarity > 0.0,
                runtime_ms=runtime_ms,
                scheduler_overhead_ms=scheduler_overhead_ms,
            )

        # Analyze trajectory
        analysis = self._trajectory_analyzer.analyze(
            residual_history, used_warmstart=warmstart_used,
        )
        trajectory_class = analysis.trajectory_class

        # Compute iterations
        allocated = schedule.budget.max_iterations if schedule.budget else 0
        saved = max(0, allocated - iterations_used)

        # Determine outcome
        if analysis.divergence_detected:
            outcome = OUTCOME_DIVERGED
            stop_reason = STOP_DIVERGENCE
        elif analysis.stagnation_detected and iterations_used >= allocated:
            outcome = OUTCOME_STAGNATED
            stop_reason = STOP_STAGNATED
        elif iterations_used >= allocated:
            outcome = OUTCOME_BUDGET_EXHAUSTED
            stop_reason = STOP_BUDGET_EXHAUSTED
        elif final_residual <= (schedule.budget.convergence_target if schedule.budget else 1e-4):
            if iterations_used < allocated * 0.5:
                outcome = OUTCOME_CONVERGED_EARLY
            else:
                outcome = OUTCOME_SUCCESS
            stop_reason = STOP_CONVERGED
        elif schedule.force_oracle and final_residual > 0.1:
            outcome = OUTCOME_ESCALATED
            stop_reason = STOP_ESCALATE
        else:
            outcome = OUTCOME_SUCCESS
            stop_reason = None

        return ExecutionOutcome(
            outcome=outcome,
            iterations_used=iterations_used,
            iterations_allocated=allocated,
            iterations_saved=saved,
            trajectory_class=trajectory_class,
            stop_reason=stop_reason,
            final_residual=final_residual,
            warmstart_used=warmstart_used,
            retrieval_used=schedule.retrieval_similarity > 0.0,
            runtime_ms=runtime_ms,
            scheduler_overhead_ms=scheduler_overhead_ms,
        )

    # -- Failure recording ---------------------------------------------------

    def record_failure(self, topology_family: str) -> None:
        """Record a failure for a topology family."""
        self._failure_counts[topology_family] = self._failure_counts.get(topology_family, 0) + 1

    def record_success(self, topology_family: str) -> None:
        """Record a success (reduces failure count by 1, min 0)."""
        current = self._failure_counts.get(topology_family, 0)
        if current > 0:
            self._failure_counts[topology_family] = current - 1

    @property
    def failure_counts(self) -> dict[str, int]:
        return dict(self._failure_counts)
