"""CORE v3.1 -- Deterministic execution guarantee tests.

Verify that same input always produces same output.
"""

from __future__ import annotations

import numpy as np
import pytest

from core_runtime.domains.linear_system import (
    LinearSystemTask,
    LinearSystemOracle,
    LinearSystemSurrogate,
    LinearSystemProjection,
    execute_linear_system_pipeline,
)
from backend.runtime.trajectory_analysis import TrajectoryAnalyzer
from backend.runtime.projection_scheduler import ProjectionScheduler


@pytest.fixture
def rng():
    return np.random.default_rng(99)


@pytest.fixture
def make_task(rng):
    def _make(seed=99):
        r = np.random.default_rng(seed)
        A = r.standard_normal((5, 5))
        A = A @ A.T + 5.0 * np.eye(5)
        b = r.standard_normal(5)
        return LinearSystemTask(
            task_id=f"det_test_{seed}",
            domain_name="linear_system",
            input_artifact="test",
            metadata={"A": A, "b": b},
        )
    return _make


class TestOracleDeterminism:

    def test_same_task_same_result(self, make_task):
        task = make_task(42)
        oracle = LinearSystemOracle()
        r1 = oracle.solve(task)
        r2 = oracle.solve(task)
        np.testing.assert_array_equal(r1["solution"], r2["solution"])

    def test_same_task_same_residual(self, make_task):
        task = make_task(42)
        oracle = LinearSystemOracle()
        r1 = oracle.solve(task)
        r2 = oracle.solve(task)
        assert r1["residual"] == r2["residual"]


class TestSurrogateDeterminism:

    def test_surrogate_same_result(self, make_task):
        task = make_task(42)
        s = LinearSystemSurrogate()
        r1 = s.predict(task)
        r2 = s.predict(task)
        np.testing.assert_array_equal(r1["prediction"], r2["prediction"])


class TestProjectionDeterminism:

    def test_projection_same_solution(self, make_task):
        task = make_task(42)
        s = LinearSystemSurrogate()
        sr = s.predict(task)
        p = LinearSystemProjection()
        r1 = p.project(task, sr, budget=50)
        r2 = p.project(task, sr, budget=50)
        np.testing.assert_array_equal(r1["solution"], r2["solution"])

    def test_projection_same_trajectory(self, make_task):
        task = make_task(42)
        s = LinearSystemSurrogate()
        sr = s.predict(task)
        p = LinearSystemProjection()
        r1 = p.project(task, sr, budget=50)
        r2 = p.project(task, sr, budget=50)
        np.testing.assert_array_equal(r1["trajectory"], r2["trajectory"])


class TestPipelineDeterminism:

    def test_pipeline_same_solution(self, make_task):
        task = make_task(42)
        r1 = execute_linear_system_pipeline(task, budget=50)
        r2 = execute_linear_system_pipeline(task, budget=50)
        np.testing.assert_array_equal(
            r1["projection"]["solution"],
            r2["projection"]["solution"],
        )

    def test_pipeline_same_confidence(self, make_task):
        task = make_task(42)
        r1 = execute_linear_system_pipeline(task, budget=50)
        r2 = execute_linear_system_pipeline(task, budget=50)
        assert r1["confidence"] == r2["confidence"]

    def test_pipeline_same_trace(self, make_task):
        task = make_task(42)
        r1 = execute_linear_system_pipeline(task, budget=50)
        r2 = execute_linear_system_pipeline(task, budget=50)
        t1, t2 = r1["trace"], r2["trace"]
        # Compare deterministic fields only (runtime_ms varies)
        for key in ["task_id", "domain_name", "fingerprint", "node_count", "edge_count",
                     "surrogate_method", "projection_iterations", "projection_converged",
                     "projection_method", "evaluation_correct", "trajectory_length"]:
            assert t1[key] == t2[key], f"Trace field {key} differs"


class TestTrajectoryAnalyzerDeterminism:

    def test_same_residuals_same_class(self):
        analyzer = TrajectoryAnalyzer()
        residuals = [1.0, 0.5, 0.25, 0.125]
        r1 = analyzer.analyze(residuals)
        r2 = analyzer.analyze(residuals)
        assert r1.trajectory_class == r2.trajectory_class

    def test_same_residuals_same_oscillation_flag(self):
        analyzer = TrajectoryAnalyzer()
        residuals = [0.8, 0.4, 0.5, 0.25, 0.3, 0.1]
        r1 = analyzer.analyze(residuals)
        r2 = analyzer.analyze(residuals)
        assert r1.oscillation_detected == r2.oscillation_detected
