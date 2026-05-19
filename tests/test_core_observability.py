"""CORE v3.1 -- Observability and tracing tests.

Verify that CORE v3.1 observability features work correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from core_runtime.domains.linear_system import (
    LinearSystemTask,
    LinearSystemTrace,
    LinearSystemSurrogate,
    LinearSystemProjection,
    LinearSystemEvaluator,
    LinearSystemConfidence,
    execute_linear_system_pipeline,
)


@pytest.fixture
def task():
    rng = np.random.default_rng(55)
    A = rng.standard_normal((3, 3))
    A = A @ A.T + 3.0 * np.eye(3)
    b = rng.standard_normal(3)
    return LinearSystemTask(
        task_id="obs_test_55",
        domain_name="linear_system",
        input_artifact="test",
        metadata={"A": A, "b": b},
    )


class TestTraceObservability:

    def test_pipeline_produces_trace(self, task):
        result = execute_linear_system_pipeline(task, budget=50)
        assert "trace" in result
        trace = result["trace"]
        assert isinstance(trace, dict)

    def test_trace_has_task_id(self, task):
        result = execute_linear_system_pipeline(task, budget=50)
        assert result["trace"]["task_id"] == "obs_test_55"

    def test_trace_has_fingerprint(self, task):
        result = execute_linear_system_pipeline(task, budget=50)
        assert "fingerprint" in result["trace"]
        assert len(result["trace"]["fingerprint"]) > 0

    def test_trace_has_surrogate_method(self, task):
        result = execute_linear_system_pipeline(task, budget=50)
        assert "surrogate_method" in result["trace"]

    def test_trace_has_projection_iterations(self, task):
        result = execute_linear_system_pipeline(task, budget=50)
        assert "projection_iterations" in result["trace"]
        assert result["trace"]["projection_iterations"] > 0

    def test_trace_has_projection_converged(self, task):
        result = execute_linear_system_pipeline(task, budget=50)
        assert "projection_converged" in result["trace"]

    def test_trace_has_confidence_score(self, task):
        result = execute_linear_system_pipeline(task, budget=50)
        assert "confidence_score" in result["trace"]

    def test_trace_has_trajectory_length(self, task):
        result = execute_linear_system_pipeline(task, budget=50)
        assert "trajectory_length" in result["trace"]
        assert result["trace"]["trajectory_length"] > 0


class TestTraceBuildDirectly:

    def test_trace_build_with_all_fields(self, task):
        trace = LinearSystemTrace.build_trace(
            task,
            {"residual": 1.0, "method": "jacobi", "runtime_ms": 0.1},
            {"residual": 0.01, "iterations": 20, "converged": True, "method": "iterative", "runtime_ms": 1.0, "trajectory": [1.0, 0.5, 0.01]},
            {"residual": 0.01, "correct": True},
            confidence_score=0.85,
        )
        assert trace["task_id"] == "obs_test_55"
        assert trace["confidence_score"] == 0.85
        assert trace["projection_iterations"] == 20

    def test_trace_without_confidence(self, task):
        trace = LinearSystemTrace.build_trace(
            task,
            {"residual": 1.0, "method": "jacobi", "runtime_ms": 0.1},
            {"residual": 0.01, "iterations": 10, "converged": True, "method": "iterative", "runtime_ms": 1.0, "trajectory": []},
            {"residual": 0.01, "correct": True},
        )
        assert trace["confidence_score"] is None
