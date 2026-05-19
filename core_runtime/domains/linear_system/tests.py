"""Linear system domain tests -- prove the SDK works for a non-circuit domain."""

from __future__ import annotations

import numpy as np
import pytest

from core_runtime.domains.linear_system import (
    LinearSystemTask,
    LinearSystemOracle,
    LinearSystemSurrogate,
    LinearSystemProjection,
    LinearSystemEvaluator,
)


def _make_spd_matrix(n: int, seed: int = 42) -> np.ndarray:
    """Create a symmetric positive-definite matrix for testing."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A.T @ A + n * np.eye(n)  # guaranteed SPD


def _make_task(n: int = 10, seed: int = 42) -> LinearSystemTask:
    rng = np.random.default_rng(seed)
    A = _make_spd_matrix(n, seed)
    b = rng.standard_normal(n)
    return LinearSystemTask(
        task_id=f"ls_test_{n}_{seed}",
        domain_name="linear_system",
        input_artifact=f"ls_test_{n}_{seed}",
        metadata={"A": A, "b": b},
    )


class TestLinearSystemOracle:
    def test_oracle_solves_exact(self):
        task = _make_task(10)
        oracle = LinearSystemOracle()
        result = oracle.solve(task)
        assert result["residual"] < 1e-10
        assert result["runtime_ms"] >= 0
        assert "solution" in result

    def test_oracle_solution_is_correct(self):
        task = _make_task(5, seed=99)
        oracle = LinearSystemOracle()
        result = oracle.solve(task)
        A = task.metadata["A"]
        b = task.metadata["b"]
        x = result["solution"]
        np.testing.assert_allclose(A @ x, b, atol=1e-8)


class TestLinearSystemSurrogate:
    def test_surrogate_returns_prediction(self):
        task = _make_task(10)
        surrogate = LinearSystemSurrogate(use_jacobi=True)
        result = surrogate.predict(task)
        assert "prediction" in result
        assert result["residual"] >= 0

    def test_zero_init_surrogate(self):
        task = _make_task(5)
        surrogate = LinearSystemSurrogate(use_jacobi=False)
        result = surrogate.predict(task)
        np.testing.assert_array_equal(result["prediction"], np.zeros(5))


class TestLinearSystemProjection:
    def test_projection_reduces_residual(self):
        task = _make_task(10)
        surrogate = LinearSystemSurrogate(use_jacobi=True)
        pred = surrogate.predict(task)
        initial_residual = pred["residual"]

        projection = LinearSystemProjection(step_size=0.01, tol=1e-10)
        result = projection.project(task, pred, budget=200)
        assert result["residual"] <= initial_residual

    def test_projection_converges_for_small_systems(self):
        task = _make_task(3, seed=7)
        surrogate = LinearSystemSurrogate(use_jacobi=True)
        pred = surrogate.predict(task)

        projection = LinearSystemProjection(step_size=0.05, tol=1e-8)
        result = projection.project(task, pred, budget=500)
        assert result["residual"] < 1e-6


class TestLinearSystemEvaluator:
    def test_evaluator_correct_solution(self):
        task = _make_task(10)
        oracle = LinearSystemOracle()
        result = oracle.solve(task)
        evaluator = LinearSystemEvaluator(tolerance=1e-6)
        evaluation = evaluator.evaluate(task, result)
        assert evaluation["correct"] is True
        assert evaluation["residual"] < 1e-6

    def test_evaluator_bad_solution(self):
        task = _make_task(10)
        bad_solution = {"solution": np.zeros(10)}
        evaluator = LinearSystemEvaluator(tolerance=1e-6)
        evaluation = evaluator.evaluate(task, bad_solution)
        assert evaluation["correct"] is False


class TestLinearSystemEndToEnd:
    def test_full_pipeline(self):
        """Oracle -> surrogate -> projection -> evaluator pipeline."""
        task = _make_task(8, seed=123)
        oracle = LinearSystemOracle()
        surrogate = LinearSystemSurrogate(use_jacobi=True)
        projection = LinearSystemProjection(step_size=0.02, tol=1e-8)
        evaluator = LinearSystemEvaluator(tolerance=1e-6)

        # Surrogate predicts
        pred = surrogate.predict(task)
        assert pred["residual"] > 0  # not exact

        # Projection corrects
        proj = projection.project(task, pred, budget=300)
        assert proj["residual"] < pred["residual"]

        # Evaluator verifies
        eval_result = evaluator.evaluate(task, proj)
        assert eval_result["correct"] is True

        # Oracle is exact
        oracle_result = oracle.solve(task)
        oracle_eval = evaluator.evaluate(task, oracle_result)
        assert oracle_eval["correct"] is True
        assert oracle_eval["residual"] < proj["residual"] or abs(oracle_eval["residual"] - proj["residual"]) < 1e-8


class TestLinearSystemTask:
    def test_task_fingerprint(self):
        task = _make_task(5, seed=42)
        fp = task.fingerprint()
        assert fp.startswith("ls-")
        assert len(fp) > 10

    def test_task_fingerprint_deterministic(self):
        task1 = _make_task(5, seed=42)
        task2 = _make_task(5, seed=42)
        assert task1.fingerprint() == task2.fingerprint()

    def test_task_node_edge_count(self):
        task = _make_task(7)
        assert task.node_count() == 7
        assert task.edge_count() > 0  # SPD matrix has nonzero off-diags
