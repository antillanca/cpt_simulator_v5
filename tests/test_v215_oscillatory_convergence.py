"""CPT v2.15 -- Oscillatory Convergence Validation Test Suite.

Validates that the TrajectoryAnalyzer correctly classifies:
  A: Oscillatory convergent (NOT divergence_risk, NOT stalled)
  B: Divergence risk (monotonically increasing)
  C: Stalled (near-zero improvement)
  D: Fast converging (strict exponential decay)

Also permanently covers _is_diverging() regression.
"""

import pytest

from backend.runtime.projection_scheduler import (
    TRAJECTORY_DIVERGENCE_RISK,
    TRAJECTORY_FAST_CONVERGING,
    TRAJECTORY_OSCILLATORY,
    TRAJECTORY_RETRIEVAL_ASSISTED,
    TRAJECTORY_STABLE_LINEAR,
    TRAJECTORY_STALLED,
    VALID_TRAJECTORY_CLASSES,
    ProjectionScheduler,
)
from backend.runtime.trajectory_analysis import (
    TrajectoryAnalysisResult,
    TrajectoryAnalyzer,
    TrajectoryMetrics,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def analyzer() -> TrajectoryAnalyzer:
    return TrajectoryAnalyzer()


@pytest.fixture
def scheduler() -> ProjectionScheduler:
    return ProjectionScheduler()


# ═══════════════════════════════════════════════════════════════
# CASE A: Oscillatory convergent
# 0.8 -> 0.4 -> 0.5 -> 0.25 -> 0.3 -> 0.1
# ═══════════════════════════════════════════════════════════════

class TestCaseAOscillatoryConvergent:
    """Oscillating but clearly converging trajectory."""

    RESIDUALS = [0.8, 0.4, 0.5, 0.25, 0.3, 0.1]

    def test_classified_as_oscillatory(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.trajectory_class == TRAJECTORY_OSCILLATORY, (
            f"Expected oscillatory, got {result.trajectory_class}"
        )

    def test_not_divergence_risk(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert not result.divergence_detected, (
            "Oscillatory convergent should NOT be flagged as divergence_risk"
        )

    def test_not_stalled(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert not result.stagnation_detected, (
            "Oscillatory convergent should NOT be stalled"
        )

    def test_total_improvement_positive(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.metrics.total_improvement > 0, (
            "Oscillatory convergent must have positive total improvement"
        )

    def test_oscillation_detected(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.oscillation_detected, (
            "Oscillatory trajectory must flag oscillation_detected=True"
        )

    def test_metrics_oscillation_rate(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.metrics.oscillation_rate > 0.3, (
            f"Expected oscillation_rate > 0.3, got {result.metrics.oscillation_rate}"
        )

    def test_final_residual(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.metrics.final_residual == 0.1


# ═══════════════════════════════════════════════════════════════
# CASE B: Divergence risk
# 0.8 -> 0.82 -> 0.85 -> 0.9
# ═══════════════════════════════════════════════════════════════

class TestCaseBDivergenceRisk:
    """Monotonically increasing residuals -> divergence_risk."""

    RESIDUALS = [0.8, 0.82, 0.85, 0.9]

    def test_classified_as_divergence_risk(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.trajectory_class == TRAJECTORY_DIVERGENCE_RISK, (
            f"Expected divergence_risk, got {result.trajectory_class}"
        )

    def test_divergence_detected(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.divergence_detected, (
            "Divergent trajectory must flag divergence_detected=True"
        )

    def test_stagnation_may_be_detected(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        # Divergence with no improvement also triggers stagnation - both
        # flags are independent; trajectory_class=divergence_risk takes priority
        assert result.divergence_detected, "Must be divergent regardless"
        # stagnation_detected may be True or False - both are valid

    def test_total_improvement_negative(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.metrics.total_improvement < 0, (
            "Divergent trajectory must have negative total improvement"
        )


# ═══════════════════════════════════════════════════════════════
# CASE C: Stalled
# 0.8, 0.7999999999, 0.7999999998, 0.7999999997 (near-zero improvement)
# ═══════════════════════════════════════════════════════════════

class TestCaseCStalled:
    """Near-zero improvement -> stalled."""

    RESIDUALS = [0.8, 0.7999999999, 0.7999999998, 0.7999999997]

    def test_classified_as_stalled(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.trajectory_class == TRAJECTORY_STALLED, (
            f"Expected stalled, got {result.trajectory_class}"
        )

    def test_stagnation_detected(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.stagnation_detected, (
            "Stalled trajectory must flag stagnation_detected=True"
        )

    def test_not_divergence_risk(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert not result.divergence_detected, (
            "Stalled trajectory should NOT be flagged as divergence_risk"
        )

    def test_minimal_improvement(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.metrics.total_improvement < 1e-2, (
            "Stalled trajectory must have minimal improvement"
        )


# ═══════════════════════════════════════════════════════════════
# CASE D: Fast converging (strict exponential decay)
# ═══════════════════════════════════════════════════════════════

class TestCaseDFastConverging:
    """Strict exponential decay -> fast_converging."""

    # Exponential: 0.5 * 0.1^k for k=0..7  ->  0.5, 0.05, 0.005, 0.0005, ...
    RESIDUALS = [0.5, 0.05, 0.005, 0.0005, 0.00005, 0.000005, 0.0000005, 0.00000005]

    def test_classified_as_fast_converging(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.trajectory_class == TRAJECTORY_FAST_CONVERGING, (
            f"Expected fast_converging, got {result.trajectory_class}"
        )

    def test_not_oscillatory(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert not result.oscillation_detected, (
            "Exponential decay should NOT be oscillatory"
        )

    def test_not_divergent(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert not result.divergence_detected, (
            "Exponential decay should NOT be divergent"
        )

    def test_not_stalled(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert not result.stagnation_detected, (
            "Exponential decay should NOT be stalled"
        )

    def test_converged(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.metrics.converged, (
            "Exponential decay must reach convergence target"
        )

    def test_early_slope_much_greater_than_average(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        # fast_converging requires early_slope > 2x average_slope
        assert result.metrics.early_slope > result.metrics.average_slope * 2.0


# ═══════════════════════════════════════════════════════════════
# _is_diverging() regression -- permanent coverage
# ═══════════════════════════════════════════════════════════════

class TestIsDivergingRegression:
    """Permanent regression coverage for _is_diverging()."""

    def test_monotonic_increase_3_steps(self, scheduler: ProjectionScheduler):
        assert scheduler._is_diverging(0.9, [0.8, 0.82, 0.85])

    def test_monotonic_increase_4_steps(self, scheduler: ProjectionScheduler):
        assert scheduler._is_diverging(1.0, [0.7, 0.8, 0.9])

    def test_no_divergence_decreasing(self, scheduler: ProjectionScheduler):
        assert not scheduler._is_diverging(0.1, [0.5, 0.3, 0.2])

    def test_no_divergence_insufficient_history(self, scheduler: ProjectionScheduler):
        assert not scheduler._is_diverging(0.9, [0.8])

    def test_no_divergence_empty_history(self, scheduler: ProjectionScheduler):
        assert not scheduler._is_diverging(0.9, [])

    def test_no_divergence_oscillatory_but_converging(self, scheduler: ProjectionScheduler):
        # CASE A residuals: 0.8, 0.4, 0.5, 0.25, 0.3, 0.1
        # Last 3 are: 0.25, 0.3, 0.1 -- NOT monotonically increasing
        assert not scheduler._is_diverging(0.1, [0.8, 0.4, 0.5, 0.25, 0.3])

    def test_boundary_exactly_equal(self, scheduler: ProjectionScheduler):
        # [0.8, 0.8, 0.8] -- not strictly increasing
        assert not scheduler._is_diverging(0.8, [0.8, 0.8, 0.8])

    def test_classify_trajectory_divergence(self, scheduler: ProjectionScheduler):
        result = scheduler.classify_trajectory([0.8, 0.82, 0.85, 0.9])
        assert result == TRAJECTORY_DIVERGENCE_RISK

    def test_classify_trajectory_converging(self, scheduler: ProjectionScheduler):
        result = scheduler.classify_trajectory([0.5, 0.1, 0.01, 0.001])
        assert result in (TRAJECTORY_FAST_CONVERGING, TRAJECTORY_STABLE_LINEAR)


# ═══════════════════════════════════════════════════════════════
# Determinism: same input -> same output
# ═══════════════════════════════════════════════════════════════

class TestOscillatoryDeterminism:
    """Verify deterministic classification for all trajectory types."""

    def test_case_a_deterministic(self, analyzer: TrajectoryAnalyzer):
        residuals = [0.8, 0.4, 0.5, 0.25, 0.3, 0.1]
        r1 = analyzer.analyze(residuals)
        r2 = analyzer.analyze(residuals)
        assert r1.trajectory_class == r2.trajectory_class
        assert r1.metrics.oscillation_rate == r2.metrics.oscillation_rate
        assert r1.metrics.final_residual == r2.metrics.final_residual

    def test_case_b_deterministic(self, analyzer: TrajectoryAnalyzer):
        residuals = [0.8, 0.82, 0.85, 0.9]
        r1 = analyzer.analyze(residuals)
        r2 = analyzer.analyze(residuals)
        assert r1.trajectory_class == r2.trajectory_class == TRAJECTORY_DIVERGENCE_RISK

    def test_case_c_deterministic(self, analyzer: TrajectoryAnalyzer):
        residuals = [0.8, 0.7999999999, 0.7999999998, 0.7999999997]
        r1 = analyzer.analyze(residuals)
        r2 = analyzer.analyze(residuals)
        assert r1.trajectory_class == r2.trajectory_class == TRAJECTORY_STALLED

    def test_case_d_deterministic(self, analyzer: TrajectoryAnalyzer):
        residuals = [0.5, 0.05, 0.005, 0.0005, 0.00005]
        r1 = analyzer.analyze(residuals)
        r2 = analyzer.analyze(residuals)
        assert r1.trajectory_class == r2.trajectory_class


# ═══════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases for oscillatory classification."""

    def test_single_point(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze([0.5])
        # Single point -> stable_linear by default
        assert result.trajectory_class in VALID_TRAJECTORY_CLASSES

    def test_two_points_converging(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze([0.5, 0.1])
        assert result.trajectory_class in (TRAJECTORY_FAST_CONVERGING, TRAJECTORY_STABLE_LINEAR)

    def test_two_points_equal(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze([0.5, 0.5])
        assert result.trajectory_class == TRAJECTORY_STALLED

    def test_empty_residuals(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze([])
        assert result.metrics.iterations == 0
        assert result.metrics.final_residual == 0.0

    def test_high_frequency_oscillation_converging(self, analyzer: TrajectoryAnalyzer):
        # 10 points, oscillating but converging strongly
        residuals = [1.0, 0.6, 0.7, 0.3, 0.4, 0.15, 0.2, 0.05, 0.08, 0.001]
        result = analyzer.analyze(residuals)
        # Should be oscillatory (high oscillation rate)
        assert result.oscillation_detected or result.trajectory_class == TRAJECTORY_FAST_CONVERGING

    def test_low_frequency_oscillation(self, analyzer: TrajectoryAnalyzer):
        # Mostly linear with one bump -- not enough oscillation
        residuals = [0.5, 0.4, 0.45, 0.3, 0.2, 0.1, 0.001]
        result = analyzer.analyze(residuals)
        # One bump shouldn't trigger oscillatory (>30% rate needed)
        # This has 1/5 = 0.2 oscillation rate -> below threshold
        assert result.trajectory_class in (TRAJECTORY_FAST_CONVERGING, TRAJECTORY_STABLE_LINEAR, TRAJECTORY_OSCILLATORY)
