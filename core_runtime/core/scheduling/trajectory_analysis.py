"""CPT Runtime — Trajectory Analysis (v2.15).

Deterministic classification of projection convergence trajectories.
NO learning. NO randomness. NO hidden state.

Classifies trajectories as:
- fast_converging: rapid initial residual decrease
- stable_linear: steady linear decrease
- oscillatory: residual oscillates up/down
- stalled: minimal or no improvement
- divergence_risk: residual increasing
- retrieval_assisted: warmstart + good convergence

Uses:
- residual slope
- improvement rate
- oscillation detection
- stagnation detection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.runtime.projection_scheduler import (
    TRAJECTORY_FAST_CONVERGING,
    TRAJECTORY_STABLE_LINEAR,
    TRAJECTORY_OSCILLATORY,
    TRAJECTORY_STALLED,
    TRAJECTORY_DIVERGENCE_RISK,
    TRAJECTORY_RETRIEVAL_ASSISTED,
    VALID_TRAJECTORY_CLASSES,
)


# ---------------------------------------------------------------------------
# TrajectoryMetrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajectoryMetrics:
    """Quantitative metrics extracted from a residual history.

    Attributes:
        total_improvement: first_residual - last_residual (positive = improvement)
        average_slope: average residual decrease per iteration
        early_slope: slope from first 3 points (or fewer if short)
        late_slope: slope from last 3 points (or fewer if short)
        oscillation_count: number of local extrema
        oscillation_rate: oscillation_count / (len-2)
        max_residual: peak residual observed
        min_residual: lowest residual observed
        final_residual: last residual value
        iterations: total iterations in the history
        converged: whether final_residual < convergence_target
    """

    total_improvement: float
    average_slope: float
    early_slope: float
    late_slope: float
    oscillation_count: int
    oscillation_rate: float
    max_residual: float
    min_residual: float
    final_residual: float
    iterations: int
    converged: bool

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "total_improvement": round(self.total_improvement, 12),
            "average_slope": round(self.average_slope, 12),
            "early_slope": round(self.early_slope, 12),
            "late_slope": round(self.late_slope, 12),
            "oscillation_count": self.oscillation_count,
            "oscillation_rate": round(self.oscillation_rate, 6),
            "max_residual": round(self.max_residual, 12),
            "min_residual": round(self.min_residual, 12),
            "final_residual": round(self.final_residual, 12),
            "iterations": self.iterations,
            "converged": self.converged,
        }


# ---------------------------------------------------------------------------
# TrajectoryAnalysisResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajectoryAnalysisResult:
    """Complete trajectory analysis result.

    Attributes:
        trajectory_class: One of VALID_TRAJECTORY_CLASSES.
        metrics: Quantitative metrics.
        used_warmstart: Whether warmstart was used.
        stagnation_detected: Whether stagnation was found.
        divergence_detected: Whether divergence was found.
        oscillation_detected: Whether significant oscillation was found.
    """

    trajectory_class: str
    metrics: TrajectoryMetrics
    used_warmstart: bool
    stagnation_detected: bool
    divergence_detected: bool
    oscillation_detected: bool

    def __post_init__(self) -> None:
        if self.trajectory_class not in VALID_TRAJECTORY_CLASSES:
            raise ValueError(
                f"Invalid trajectory class: {self.trajectory_class}, "
                f"expected one of {VALID_TRAJECTORY_CLASSES}"
            )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "trajectory_class": self.trajectory_class,
            "metrics": self.metrics.to_json_dict(),
            "used_warmstart": self.used_warmstart,
            "stagnation_detected": self.stagnation_detected,
            "divergence_detected": self.divergence_detected,
            "oscillation_detected": self.oscillation_detected,
        }


# ---------------------------------------------------------------------------
# TrajectoryAnalyzer
# ---------------------------------------------------------------------------

class TrajectoryAnalyzer:
    """Deterministic projection trajectory analyzer.

    Extracts metrics from residual histories and classifies trajectories.
    Same residuals → same classification. Always.

    Thresholds are fixed constants — no tuning, no learning.
    """

    # Oscillation: fraction of points that are local extrema
    _OSCILLATION_THRESHOLD = 0.3

    # Stagnation: total improvement below this = stalled
    _STAGNATION_IMPROVEMENT = 1e-8

    # Fast converging: early slope > 2x average slope AND converged
    _FAST_CONVERGENCE_SLOPE_RATIO = 2.0
    _FAST_CONVERGENCE_FINAL_RESIDUAL = 0.01

    # Divergence: 3+ consecutive increasing residuals
    _DIVERGENCE_WINDOW = 3

    # Retrieval-assisted: warmstart + converged + good improvement
    _RETRIEVAL_ASSISTED_FINAL_RESIDUAL = 0.01

    # Convergence target for "converged" check
    _CONVERGENCE_TARGET = 1e-4

    def analyze(
        self,
        residuals: list[float],
        used_warmstart: bool = False,
        convergence_target: float | None = None,
    ) -> TrajectoryAnalysisResult:
        """Analyze a projection residual trajectory.

        Returns a TrajectoryAnalysisResult with classification and metrics.
        """
        target = convergence_target or self._CONVERGENCE_TARGET
        metrics = self._compute_metrics(residuals, target)

        # Detect features
        divergence_detected = self._detect_divergence(residuals)
        stagnation_detected = self._detect_stagnation(metrics)
        oscillation_detected = metrics.oscillation_rate > self._OSCILLATION_THRESHOLD

        # Classify
        trajectory_class = self._classify(
            residuals, metrics, used_warmstart,
            divergence_detected, stagnation_detected, oscillation_detected,
        )

        return TrajectoryAnalysisResult(
            trajectory_class=trajectory_class,
            metrics=metrics,
            used_warmstart=used_warmstart,
            stagnation_detected=stagnation_detected,
            divergence_detected=divergence_detected,
            oscillation_detected=oscillation_detected,
        )

    def _compute_metrics(
        self,
        residuals: list[float],
        convergence_target: float,
    ) -> TrajectoryMetrics:
        """Extract quantitative metrics from residual history."""
        n = len(residuals)
        if n == 0:
            return TrajectoryMetrics(
                total_improvement=0.0, average_slope=0.0,
                early_slope=0.0, late_slope=0.0,
                oscillation_count=0, oscillation_rate=0.0,
                max_residual=0.0, min_residual=0.0,
                final_residual=0.0, iterations=0,
                converged=True,
            )

        first = residuals[0]
        last = residuals[-1]
        total_improvement = first - last
        avg_slope = total_improvement / max(n - 1, 1)

        # Early slope: first 3 points
        if n >= 3:
            early_slope = (residuals[0] - residuals[2]) / 2.0
        elif n >= 2:
            early_slope = residuals[0] - residuals[1]
        else:
            early_slope = 0.0

        # Late slope: last 3 points
        if n >= 3:
            late_slope = (residuals[-3] - residuals[-1]) / 2.0
        elif n >= 2:
            late_slope = residuals[-2] - residuals[-1]
        else:
            late_slope = 0.0

        # Oscillation detection
        osc_count = 0
        for i in range(1, n - 1):
            is_local_max = residuals[i] > residuals[i-1] and residuals[i] > residuals[i+1]
            is_local_min = residuals[i] < residuals[i-1] and residuals[i] < residuals[i+1]
            if is_local_max or is_local_min:
                osc_count += 1
        osc_rate = osc_count / max(n - 2, 1)

        return TrajectoryMetrics(
            total_improvement=total_improvement,
            average_slope=avg_slope,
            early_slope=early_slope,
            late_slope=late_slope,
            oscillation_count=osc_count,
            oscillation_rate=osc_rate,
            max_residual=max(residuals),
            min_residual=min(residuals),
            final_residual=last,
            iterations=n,
            converged=(last <= convergence_target),
        )

    def _detect_divergence(self, residuals: list[float]) -> bool:
        """Detect 3+ consecutive increasing residuals."""
        if len(residuals) < self._DIVERGENCE_WINDOW:
            return False
        for i in range(len(residuals) - self._DIVERGENCE_WINDOW + 1):
            window = residuals[i:i + self._DIVERGENCE_WINDOW]
            if window[0] < window[1] < window[2]:
                return True
        return False

    def _detect_stagnation(self, metrics: TrajectoryMetrics) -> bool:
        """Detect stagnation: total improvement below threshold."""
        return metrics.total_improvement < self._STAGNATION_IMPROVEMENT

    def _classify(
        self,
        residuals: list[float],
        metrics: TrajectoryMetrics,
        used_warmstart: bool,
        divergence_detected: bool,
        stagnation_detected: bool,
        oscillation_detected: bool,
    ) -> str:
        """Classify trajectory. DETERMINISTIC priority order."""
        # Priority 1: Divergence risk
        if divergence_detected:
            return TRAJECTORY_DIVERGENCE_RISK

        # Priority 2: Stalled
        if stagnation_detected:
            return TRAJECTORY_STALLED

        # Priority 3: Retrieval-assisted
        if used_warmstart and metrics.total_improvement > 0 and \
           metrics.final_residual < self._RETRIEVAL_ASSISTED_FINAL_RESIDUAL:
            return TRAJECTORY_RETRIEVAL_ASSISTED

        # Priority 4: Oscillatory
        if oscillation_detected:
            return TRAJECTORY_OSCILLATORY

        # Priority 5: Fast converging
        if (metrics.early_slope > metrics.average_slope * self._FAST_CONVERGENCE_SLOPE_RATIO
                and metrics.final_residual < self._FAST_CONVERGENCE_FINAL_RESIDUAL):
            return TRAJECTORY_FAST_CONVERGING

        # Default: Stable linear
        return TRAJECTORY_STABLE_LINEAR
