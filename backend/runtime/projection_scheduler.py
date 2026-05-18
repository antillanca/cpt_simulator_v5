"""CPT Runtime — Projection Scheduler (v2.15).

DETERMINISTIC budget allocation for projection execution.
NO learning. NO randomness. NO hidden state.
NO changes to projection mathematics.

The scheduler ONLY decides:
- max_iterations: how many projection steps to allow
- stagnation_patience: how many steps without improvement before stopping
- min_improvement: threshold for what counts as meaningful improvement
- convergence_target: residual target for early stopping
- escalation_threshold: residual above which to escalate early

Projection remains the final authority.
The scheduler only allocates effort and decides when to stop or escalate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.core_runtime.confidence_runtime import ConfidenceEstimate
from backend.runtime.cost_estimator import ExecutionCostEstimate


# ---------------------------------------------------------------------------
# ProjectionBudget (v2.15)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectionBudget:
    """Immutable projection budget allocated by the scheduler.

    Attributes:
        max_iterations: Maximum projection iterations allowed.
        stagnation_patience: Iterations without improvement before stopping.
        min_improvement: Minimum residual improvement per patience window.
        convergence_target: Residual target for early convergence stop.
        escalation_threshold: Residual above which to escalate immediately.
        family: Topology family this budget was allocated for.
        confidence: Confidence score used in allocation.
        estimated_cost: Cost estimate used in allocation.
    """

    max_iterations: int
    stagnation_patience: int
    min_improvement: float
    convergence_target: float
    escalation_threshold: float
    family: str
    confidence: float
    estimated_cost: float

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.stagnation_patience < 1:
            raise ValueError(f"stagnation_patience must be >= 1, got {self.stagnation_patience}")
        if self.min_improvement < 0:
            raise ValueError(f"min_improvement must be >= 0, got {self.min_improvement}")
        if self.convergence_target < 0:
            raise ValueError(f"convergence_target must be >= 0, got {self.convergence_target}")
        if self.escalation_threshold < 0:
            raise ValueError(
                f"escalation_threshold must be >= 0, got {self.escalation_threshold}"
            )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "stagnation_patience": self.stagnation_patience,
            "min_improvement": round(self.min_improvement, 12),
            "convergence_target": round(self.convergence_target, 12),
            "escalation_threshold": round(self.escalation_threshold, 12),
            "family": self.family,
            "confidence": round(self.confidence, 8),
            "estimated_cost": round(self.estimated_cost, 3),
        }


# ---------------------------------------------------------------------------
# TrajectoryClass (v2.15)
# ---------------------------------------------------------------------------

TRAJECTORY_FAST_CONVERGING = "fast_converging"
TRAJECTORY_STABLE_LINEAR = "stable_linear"
TRAJECTORY_OSCILLATORY = "oscillatory"
TRAJECTORY_STALLED = "stalled"
TRAJECTORY_DIVERGENCE_RISK = "divergence_risk"
TRAJECTORY_RETRIEVAL_ASSISTED = "retrieval_assisted"

VALID_TRAJECTORY_CLASSES = frozenset({
    TRAJECTORY_FAST_CONVERGING,
    TRAJECTORY_STABLE_LINEAR,
    TRAJECTORY_OSCILLATORY,
    TRAJECTORY_STALLED,
    TRAJECTORY_DIVERGENCE_RISK,
    TRAJECTORY_RETRIEVAL_ASSISTED,
})


# ---------------------------------------------------------------------------
# StopDecision (v2.15)
# ---------------------------------------------------------------------------

STOP_CONTINUE = "continue"
STOP_CONVERGED = "converged"
STOP_STAGNATED = "stagnated"
STOP_DIMINISHING = "diminishing_returns"
STOP_DIVERGENCE = "divergence"
STOP_ESCALATE = "escalate"
STOP_BUDGET_EXHAUSTED = "budget_exhausted"

VALID_STOP_REASONS = frozenset({
    STOP_CONTINUE,
    STOP_CONVERGED,
    STOP_STAGNATED,
    STOP_DIMINISHING,
    STOP_DIVERGENCE,
    STOP_ESCALATE,
    STOP_BUDGET_EXHAUSTED,
})


@dataclass(frozen=True)
class StopDecision:
    """Immutable decision about whether to stop projection early.

    Attributes:
        should_stop: Whether to stop projection now.
        reason: Why (from VALID_STOP_REASONS).
        current_residual: Residual at this iteration.
        iteration: Current iteration number.
        trajectory_class: Classified trajectory type.
    """

    should_stop: bool
    reason: str
    current_residual: float
    iteration: int
    trajectory_class: str

    def __post_init__(self) -> None:
        if self.reason not in VALID_STOP_REASONS:
            raise ValueError(
                f"Invalid stop reason: {self.reason}, expected one of {VALID_STOP_REASONS}"
            )
        if self.trajectory_class not in VALID_TRAJECTORY_CLASSES:
            raise ValueError(
                f"Invalid trajectory class: {self.trajectory_class}, "
                f"expected one of {VALID_TRAJECTORY_CLASSES}"
            )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "should_stop": self.should_stop,
            "reason": self.reason,
            "current_residual": round(self.current_residual, 12),
            "iteration": self.iteration,
            "trajectory_class": self.trajectory_class,
        }


# ---------------------------------------------------------------------------
# ProjectionScheduler (v2.15)
# ---------------------------------------------------------------------------

class ProjectionScheduler:
    """Deterministic projection budget scheduler.

    Allocates projection effort based on:
    - topology family
    - node count / edge count
    - confidence estimate
    - retrieval similarity
    - cost estimate
    - projection experience history

    NEVER modifies physics equations.
    NEVER bypasses projection.
    Only decides: max_iterations, stagnation_patience, min_improvement,
    convergence_target, escalation_threshold.

    DETERMINISTIC: same inputs → same budget. Always.
    """

    # Default convergence targets
    _CONVERGENCE_TARGET = 1e-4
    _ESCALATION_THRESHOLD = 1.0
    _MIN_IMPROVEMENT = 1e-6

    # Budget tiers (iterations)
    _BUDGET_TRIVIAL = 3
    _BUDGET_EASY = 5
    _BUDGET_MODERATE = 10
    _BUDGET_HARD = 20
    _BUDGET_EXTREME = 50

    # Stagnation patience tiers
    _PATIENCE_TRIVIAL = 2
    _PATIENCE_EASY = 3
    _PATIENCE_MODERATE = 5
    _PATIENCE_HARD = 7
    _PATIENCE_EXTREME = 10

    def __init__(
        self,
        family_stats: dict[str, dict[str, Any]] | None = None,
        convergence_target: float = 1e-4,
        escalation_threshold: float = 1.0,
        min_improvement: float = 1e-6,
    ) -> None:
        self._family_stats = family_stats or {}
        self._convergence_target = convergence_target
        self._escalation_threshold = escalation_threshold
        self._min_improvement = min_improvement

    # -- Budget allocation ---------------------------------------------------

    def allocate_budget(
        self,
        confidence: ConfidenceEstimate,
        cost_estimate: ExecutionCostEstimate | None = None,
        retrieval_similarity: float = 0.0,
        topology_family: str = "unknown",
        is_ood: bool = False,
        node_count: int = 0,
    ) -> ProjectionBudget:
        """Allocate a deterministic projection budget.

        Policy:
        - exact cache hit: caller must bypass (not our job)
        - high confidence + good retrieval: small budget
        - medium confidence: moderate budget
        - likely OOD: larger budget
        - divergence risk: escalation early

        Conservative: if uncertain, allocate MORE, never less.
        """
        difficulty = self._compute_difficulty_tier(
            confidence, cost_estimate, retrieval_similarity,
            is_ood, topology_family, node_count,
        )

        # Map difficulty tier to budget parameters
        max_iters = self._tier_to_iterations(difficulty)
        patience = self._tier_to_patience(difficulty)

        # Retrieval reduces budget (but never below minimum for safety)
        if retrieval_similarity >= 0.5:
            max_iters = max(3, int(max_iters * 0.6))
            patience = max(2, int(patience * 0.7))

        # OOD increases budget
        if is_ood:
            max_iters = int(max_iters * 1.5)

        # Family experience adjustment
        family_data = self._family_stats.get(topology_family, {})
        avg_iters = family_data.get("avg_iterations", 0)
        if avg_iters > 0 and family_data.get("count", 0) >= 3:
            # Blend: 60% heuristic, 40% experience (at least 3 samples)
            max_iters = int(0.6 * max_iters + 0.4 * avg_iters)
            max_iters = max(max_iters, 3)  # Safety floor

        # Use cost estimate if available
        est_cost = 0.0
        if cost_estimate is not None:
            est_cost = cost_estimate.estimated_runtime_ms
            # If cost estimate suggests more iterations, respect it
            est_iters = cost_estimate.estimated_projection_iterations
            if est_iters > max_iters and not is_ood:
                # Only upgrade if cost says it's harder than we thought
                max_iters = min(est_iters, self._BUDGET_EXTREME)

        return ProjectionBudget(
            max_iterations=max_iters,
            stagnation_patience=patience,
            min_improvement=self._min_improvement,
            convergence_target=self._convergence_target,
            escalation_threshold=self._escalation_threshold,
            family=topology_family,
            confidence=confidence.confidence_score,
            estimated_cost=est_cost,
        )

    # -- Stopping decisions --------------------------------------------------

    def should_stop(
        self,
        iteration: int,
        current_residual: float,
        previous_residuals: list[float],
        budget: ProjectionBudget,
        trajectory_class: str = TRAJECTORY_STABLE_LINEAR,
    ) -> StopDecision:
        """Decide whether to stop projection at this iteration.

        Rules (in priority order):
        1. Budget exhausted → stop
        2. Convergence target reached → stop (converged)
        3. Escalation threshold exceeded → stop + escalate
        4. Divergence detected → stop + escalate
        5. Stagnation detected → stop (stagnated)
        6. Diminishing returns → stop
        7. Otherwise → continue

        NEVER stops before convergence target if residual is still unsafe.
        NEVER accepts a non-converged projection as final.
        """
        # Rule 1: Budget exhausted
        if iteration >= budget.max_iterations:
            return StopDecision(
                should_stop=True,
                reason=STOP_BUDGET_EXHAUSTED,
                current_residual=current_residual,
                iteration=iteration,
                trajectory_class=trajectory_class,
            )

        # Rule 2: Convergence target reached
        if current_residual <= budget.convergence_target:
            return StopDecision(
                should_stop=True,
                reason=STOP_CONVERGED,
                current_residual=current_residual,
                iteration=iteration,
                trajectory_class=trajectory_class,
            )

        # Rule 3: Escalation threshold exceeded
        if current_residual >= budget.escalation_threshold:
            return StopDecision(
                should_stop=True,
                reason=STOP_ESCALATE,
                current_residual=current_residual,
                iteration=iteration,
                trajectory_class=trajectory_class,
            )

        # Rule 4: Divergence detection
        if self._is_diverging(current_residual, previous_residuals):
            return StopDecision(
                should_stop=True,
                reason=STOP_DIVERGENCE,
                current_residual=current_residual,
                iteration=iteration,
                trajectory_class=trajectory_class,
            )

        # Need at least stagnation_patience steps to check stagnation
        if len(previous_residuals) >= budget.stagnation_patience:
            # Rule 5: Stagnation detection
            window = previous_residuals[-budget.stagnation_patience:]
            improvement = window[0] - window[-1]  # positive = improvement
            if improvement < budget.min_improvement:
                return StopDecision(
                    should_stop=True,
                    reason=STOP_STAGNATED,
                    current_residual=current_residual,
                    iteration=iteration,
                    trajectory_class=trajectory_class,
                )

            # Rule 6: Diminishing returns
            if len(previous_residuals) >= budget.stagnation_patience * 2:
                early_window = previous_residuals[-budget.stagnation_patience * 2:-budget.stagnation_patience]
                late_window = previous_residuals[-budget.stagnation_patience:]
                early_improvement = early_window[0] - early_window[-1]
                late_improvement = late_window[0] - late_window[-1]
                if early_improvement > 0 and late_improvement > 0:
                    ratio = late_improvement / early_improvement
                    if ratio < 0.1:  # Less than 10% of early improvement rate
                        return StopDecision(
                            should_stop=True,
                            reason=STOP_DIMINISHING,
                            current_residual=current_residual,
                            iteration=iteration,
                            trajectory_class=trajectory_class,
                        )

        # Rule 7: Continue
        return StopDecision(
            should_stop=False,
            reason=STOP_CONTINUE,
            current_residual=current_residual,
            iteration=iteration,
            trajectory_class=trajectory_class,
        )

    # -- Escalation check ----------------------------------------------------

    def should_escalate(
        self,
        current_residual: float,
        budget: ProjectionBudget,
        trajectory_class: str,
    ) -> bool:
        """Check if the current trajectory warrants early oracle escalation.

        Escalate if:
        - residual exceeds escalation_threshold
        - trajectory is classified as divergence_risk
        """
        if current_residual >= budget.escalation_threshold:
            return True
        if trajectory_class == TRAJECTORY_DIVERGENCE_RISK:
            return True
        return False

    # -- Trajectory classification -------------------------------------------

    def classify_trajectory(
        self,
        residuals: list[float],
        used_warmstart: bool = False,
    ) -> str:
        """Classify projection trajectory deterministically.

        Categories:
        - fast_converging: rapid residual decrease
        - stable_linear: steady linear decrease
        - oscillatory: residual oscillates
        - stalled: minimal improvement
        - divergence_risk: residual increasing
        - retrieval_assisted: warmstart with good convergence
        """
        if len(residuals) < 2:
            return TRAJECTORY_STABLE_LINEAR

        # Check divergence first (most urgent)
        if self._is_diverging(residuals[-1], residuals):
            return TRAJECTORY_DIVERGENCE_RISK

        # Retrieval-assisted: warmstart + fast convergence
        if used_warmstart and len(residuals) >= 2:
            total_improvement = residuals[0] - residuals[-1]
            if total_improvement > 0 and residuals[-1] < 0.01:
                return TRAJECTORY_RETRIEVAL_ASSISTED

        # Need at least 3 points for meaningful classification
        if len(residuals) < 3:
            return TRAJECTORY_STABLE_LINEAR

        # Compute metrics
        total_improvement = residuals[0] - residuals[-1]
        slope = total_improvement / max(len(residuals) - 1, 1)

        # Oscillation detection
        oscillation_count = 0
        for i in range(1, len(residuals) - 1):
            # Local extremum: goes up then down, or down then up
            if (residuals[i] > residuals[i-1] and residuals[i] > residuals[i+1]) or \
               (residuals[i] < residuals[i-1] and residuals[i] < residuals[i+1]):
                oscillation_count += 1
        oscillation_rate = oscillation_count / max(len(residuals) - 2, 1)

        # Stagnation: very low improvement rate
        if total_improvement < 1e-8:
            return TRAJECTORY_STALLED

        # Oscillatory
        if oscillation_rate > 0.3:
            return TRAJECTORY_OSCILLATORY

        # Fast converging: steep initial drop
        if len(residuals) >= 3:
            early_slope = (residuals[0] - residuals[2]) / 2.0
            if early_slope > slope * 2.0 and residuals[-1] < 0.01:
                return TRAJECTORY_FAST_CONVERGING

        # Default: stable linear
        return TRAJECTORY_STABLE_LINEAR

    # -- Internal helpers ----------------------------------------------------

    def _compute_difficulty_tier(
        self,
        confidence: ConfidenceEstimate,
        cost_estimate: ExecutionCostEstimate | None,
        retrieval_similarity: float,
        is_ood: bool,
        topology_family: str,
        node_count: int,
    ) -> str:
        """Compute difficulty tier from available signals. DETERMINISTIC."""
        # Start with difficulty from cost estimate
        if cost_estimate is not None:
            tier = cost_estimate.estimated_difficulty
        else:
            # Heuristic fallback
            if confidence.confidence_score >= 0.8 and not is_ood:
                tier = "easy"
            elif confidence.confidence_score >= 0.5 and not is_ood:
                tier = "moderate"
            elif is_ood:
                tier = "hard"
            else:
                tier = "moderate"

        # Promote difficulty if signals suggest it
        if is_ood and tier in ("trivial", "easy"):
            tier = "moderate"
        if confidence.confidence_score < 0.3 and tier in ("trivial", "easy", "moderate"):
            tier = "hard"

        # Retrieval can reduce difficulty (but not below easy)
        if retrieval_similarity >= 0.5 and tier in ("hard", "extreme"):
            tier = "moderate"

        # Family experience can adjust
        family_data = self._family_stats.get(topology_family, {})
        conv_rate = family_data.get("convergence_rate", 0.5)
        if conv_rate >= 0.9 and tier == "hard":
            tier = "moderate"
        elif conv_rate < 0.3 and tier in ("easy", "moderate"):
            tier = "hard"

        return tier

    def _tier_to_iterations(self, tier: str) -> int:
        """Map difficulty tier to max iterations. DETERMINISTIC."""
        mapping = {
            "trivial": self._BUDGET_TRIVIAL,
            "easy": self._BUDGET_EASY,
            "moderate": self._BUDGET_MODERATE,
            "hard": self._BUDGET_HARD,
            "extreme": self._BUDGET_EXTREME,
        }
        return mapping.get(tier, self._BUDGET_MODERATE)

    def _tier_to_patience(self, tier: str) -> int:
        """Map difficulty tier to stagnation patience. DETERMINISTIC."""
        mapping = {
            "trivial": self._PATIENCE_TRIVIAL,
            "easy": self._PATIENCE_EASY,
            "moderate": self._PATIENCE_MODERATE,
            "hard": self._PATIENCE_HARD,
            "extreme": self._PATIENCE_EXTREME,
        }
        return mapping.get(tier, self._PATIENCE_MODERATE)

    def _is_diverging(
        self,
        current_residual: float,
        previous_residuals: list[float],
    ) -> bool:
        """Detect divergence: residual increasing over last 3+ steps.

        Handles two calling conventions:
        1. should_stop(): current_residual is separate from previous_residuals
        2. classify_trajectory(): residuals[-1] passed as both current and in previous

        In case 2, current_residual == last3[2], so the strict > check fails.
        We check >= instead to cover both cases.
        """
        if len(previous_residuals) < 3:
            return False
        # Check if last 3 residuals are monotonically increasing
        last3 = previous_residuals[-3:]
        return last3[0] < last3[1] < last3[2] and current_residual >= last3[2]
