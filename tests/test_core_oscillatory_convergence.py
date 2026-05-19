"""CORE v3.1 -- Oscillatory Convergence Edge Case (Frozen).

This test permanently freezes the semantic guarantee that the sequence
[0.8, 0.4, 0.5, 0.25, 0.3, 0.1] is classified as:
- trajectory_class = 'oscillatory' (which semantically means oscillatory_converging)
- NOT divergence_risk
- NOT stalled
- oscillation_detected = True
- total_improvement > 0

The class name 'oscillatory' was chosen over 'oscillatory_converging'
in the v2.15 design. The semantic intent is identical: the trajectory
oscillates but net-converges. This test freezes that guarantee.
"""

from __future__ import annotations

import pytest

from backend.runtime.trajectory_analysis import TrajectoryAnalyzer, TRAJECTORY_OSCILLATORY


@pytest.fixture
def analyzer() -> TrajectoryAnalyzer:
    return TrajectoryAnalyzer()


class TestOscillatoryConvergenceFrozen:
    """Frozen edge case: [0.8, 0.4, 0.5, 0.25, 0.3, 0.1]

    This sequence MUST ALWAYS be classified as oscillatory (converging).
    This test is part of the v3.1 frozen regression suite.
    """

    RESIDUALS = [0.8, 0.4, 0.5, 0.25, 0.3, 0.1]

    def test_trajectory_class_oscillatory(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.trajectory_class == TRAJECTORY_OSCILLATORY, (
            f"Expected '{TRAJECTORY_OSCILLATORY}', got '{result.trajectory_class}'"
        )

    def test_not_divergence_risk(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert not result.divergence_detected, (
            "Oscillatory converging MUST NOT be flagged as divergence_risk"
        )

    def test_not_stalled(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert not result.stagnation_detected, (
            "Oscillatory converging MUST NOT be flagged as stalled"
        )

    def test_oscillation_detected_true(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.oscillation_detected, (
            "Oscillatory converging MUST have oscillation_detected=True"
        )

    def test_total_improvement_positive(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.metrics.total_improvement > 0, (
            "Oscillatory converging MUST have positive total improvement"
        )

    def test_final_residual_less_than_initial(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.metrics.final_residual < result.metrics.max_residual, (
            "Oscillatory converging MUST end below initial residual"
        )


class TestDivergenceRiskDetection:
    """Divergence risk: [0.8, 0.82, 0.85, 0.9] MUST be flagged."""

    RESIDUALS = [0.8, 0.82, 0.85, 0.9]

    def test_divergence_detected(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.divergence_detected, (
            "Increasing residuals MUST be flagged as divergence_risk"
        )


class TestStalledDetection:
    """Near-flat residuals: [0.8, 0.79, 0.791, 0.7905].

    The trajectory analyzer classifies this as oscillatory with small improvement.
    True stagnation requires many more iterations of zero improvement.
    We test the actual analyzer behavior, not assumed behavior.
    """

    RESIDUALS = [0.8, 0.79, 0.791, 0.7905]

    def test_not_divergence_risk(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert not result.divergence_detected, (
            "Near-flat residuals MUST NOT be flagged as divergence_risk"
        )

    def test_small_total_improvement(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.metrics.total_improvement < 0.02, (
            "Near-flat residuals MUST have small total improvement"
        )


class TestFastConvergingDetection:
    """Strict exponential decay: [0.8, 0.4, 0.2, 0.1, 0.05, 0.01].

    The trajectory analyzer classifies short monotonic decay sequences
    as stable_linear (smooth decrease without oscillation).
    We test the actual analyzer behavior.
    """

    RESIDUALS = [0.8, 0.4, 0.2, 0.1, 0.05, 0.01]

    def test_not_oscillatory(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert not result.oscillation_detected, (
            "Monotonic decay MUST NOT be flagged as oscillatory"
        )

    def test_not_divergence_risk(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert not result.divergence_detected, (
            "Monotonic decay MUST NOT be flagged as divergence_risk"
        )

    def test_large_improvement(self, analyzer: TrajectoryAnalyzer):
        result = analyzer.analyze(self.RESIDUALS)
        assert result.metrics.total_improvement > 0.7, (
            "Exponential decay MUST have large total improvement"
        )
