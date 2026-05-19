"""CORE v3.1 -- Domain SDK validation tests.

Verify that linear_system exercises the complete Domain SDK pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from core_runtime.domains.linear_system import (
    LinearSystemOracle,
    LinearSystemSurrogate,
    LinearSystemProjection,
    LinearSystemEvaluator,
    LinearSystemConfidence,
    LinearSystemTrace,
    LinearSystemTask,
    execute_linear_system_pipeline,
    __version__,
)


@pytest.fixture
def simple_system():
    """4x4 well-conditioned system."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((4, 4))
    A = A @ A.T + 4.0 * np.eye(4)
    b = rng.standard_normal(4)
    return A, b


@pytest.fixture
def simple_task(simple_system):
    A, b = simple_system
    return LinearSystemTask(
        task_id="sdk_test_42",
        domain_name="linear_system",
        input_artifact="test",
        metadata={"A": A, "b": b},
    )


class TestLinearSystemOracle:

    def test_oracle_solves_exactly(self, simple_task):
        oracle = LinearSystemOracle()
        result = oracle.solve(simple_task)
        A = simple_task.metadata["A"]
        b = simple_task.metadata["b"]
        assert np.allclose(A @ result["solution"], b, atol=1e-10)

    def test_oracle_returns_solution_and_residual(self, simple_task):
        oracle = LinearSystemOracle()
        result = oracle.solve(simple_task)
        assert "solution" in result
        assert "residual" in result

    def test_oracle_residual_near_zero(self, simple_task):
        oracle = LinearSystemOracle()
        result = oracle.solve(simple_task)
        assert result["residual"] < 1e-10


class TestLinearSystemSurrogate:

    def test_surrogate_returns_approximation(self, simple_task):
        surrogate = LinearSystemSurrogate()
        result = surrogate.predict(simple_task)
        assert "prediction" in result
        assert result["prediction"].shape == simple_task.metadata["b"].shape

    def test_surrogate_deterministic(self, simple_task):
        surrogate = LinearSystemSurrogate()
        r1 = surrogate.predict(simple_task)
        r2 = surrogate.predict(simple_task)
        np.testing.assert_array_equal(r1["prediction"], r2["prediction"])

    def test_surrogate_includes_method(self, simple_task):
        surrogate = LinearSystemSurrogate()
        result = surrogate.predict(simple_task)
        assert "method" in result


class TestLinearSystemProjection:

    def test_projection_reduces_residual(self, simple_task):
        surrogate = LinearSystemSurrogate()
        surrogate_result = surrogate.predict(simple_task)
        projection = LinearSystemProjection(tol=1e-6)
        result = projection.project(simple_task, surrogate_result, budget=50)
        assert result["residual"] < surrogate_result["residual"]

    def test_projection_returns_required_fields(self, simple_task):
        surrogate = LinearSystemSurrogate()
        surrogate_result = surrogate.predict(simple_task)
        projection = LinearSystemProjection(tol=1e-6)
        result = projection.project(simple_task, surrogate_result, budget=10)
        assert "solution" in result
        assert "residual" in result
        assert "iterations" in result
        assert "trajectory" in result


class TestLinearSystemEvaluator:

    def test_evaluator_returns_correct_field(self, simple_task):
        oracle = LinearSystemOracle()
        oracle_result = oracle.solve(simple_task)
        evaluator = LinearSystemEvaluator(tolerance=1e-6)
        # Evaluator expects projection-style dict
        proj_dict = {"solution": oracle_result["solution"], "residual": oracle_result["residual"]}
        result = evaluator.evaluate(simple_task, proj_dict)
        assert "correct" in result
        assert result["correct"] is True

    def test_evaluator_rejects_bad_solution(self, simple_task):
        evaluator = LinearSystemEvaluator(tolerance=1e-6)
        bad_dict = {"solution": np.zeros(4), "residual": 100.0}
        result = evaluator.evaluate(simple_task, bad_dict)
        assert result["correct"] is False


class TestLinearSystemConfidence:

    def test_confidence_converging_trajectory(self, simple_task):
        confidence = LinearSystemConfidence()
        surrogate = LinearSystemSurrogate()
        surrogate_result = surrogate.predict(simple_task)
        score = confidence.score(simple_task, surrogate_result)
        assert 0.0 <= score <= 1.0

    def test_confidence_returns_float(self, simple_task):
        confidence = LinearSystemConfidence()
        surrogate = LinearSystemSurrogate()
        surrogate_result = surrogate.predict(simple_task)
        score = confidence.score(simple_task, surrogate_result)
        assert isinstance(score, float)


class TestLinearSystemTrace:

    def test_trace_has_task_id(self, simple_task):
        trace = LinearSystemTrace.build_trace(
            simple_task,
            {"residual": 1.0, "method": "test", "runtime_ms": 0.1},
            {"residual": 0.01, "iterations": 10, "converged": True, "method": "test", "runtime_ms": 1.0, "trajectory": []},
            {"residual": 0.01, "correct": True},
        )
        assert trace["task_id"] == "sdk_test_42"

    def test_trace_has_domain_name(self, simple_task):
        trace = LinearSystemTrace.build_trace(
            simple_task,
            {"residual": 1.0, "method": "test", "runtime_ms": 0.1},
            {"residual": 0.01, "iterations": 10, "converged": True, "method": "test", "runtime_ms": 1.0, "trajectory": []},
            {"residual": 0.01, "correct": True},
        )
        assert trace["domain_name"] == "linear_system"

    def test_trace_deterministic(self, simple_task):
        sr = {"residual": 1.0, "method": "test", "runtime_ms": 0.1}
        pr = {"residual": 0.01, "iterations": 10, "converged": True, "method": "test", "runtime_ms": 1.0, "trajectory": []}
        er = {"residual": 0.01, "correct": True}
        t1 = LinearSystemTrace.build_trace(simple_task, sr, pr, er)
        t2 = LinearSystemTrace.build_trace(simple_task, sr, pr, er)
        assert t1 == t2


class TestFullPipeline:

    def test_pipeline_runs_end_to_end(self, simple_task):
        result = execute_linear_system_pipeline(simple_task, budget=100)
        assert result is not None
        assert "projection" in result
        assert "evaluation" in result
        assert "confidence" in result
        assert "trace" in result

    def test_pipeline_trace_present(self, simple_task):
        result = execute_linear_system_pipeline(simple_task, budget=50)
        trace = result["trace"]
        assert trace["task_id"] == "sdk_test_42"

    def test_pipeline_confidence_reasonable(self, simple_task):
        result = execute_linear_system_pipeline(simple_task, budget=100)
        confidence = result["confidence"]
        assert 0.0 <= confidence <= 1.0

    def test_pipeline_deterministic(self, simple_task):
        r1 = execute_linear_system_pipeline(simple_task, budget=50)
        r2 = execute_linear_system_pipeline(simple_task, budget=50)
        np.testing.assert_array_equal(
            r1["projection"]["solution"],
            r2["projection"]["solution"]
        )

    def test_pipeline_surrogate_included(self, simple_task):
        result = execute_linear_system_pipeline(simple_task, budget=50)
        assert "surrogate" in result
        assert "method" in result["surrogate"]


class TestLinearSystemVersion:

    def test_version_is_0_2_0(self):
        assert __version__ == "0.2.0"
