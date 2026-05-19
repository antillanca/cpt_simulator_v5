"""Linear system domain end-to-end tests."""

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


def _spd_matrix(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A.T @ A + n * np.eye(n)


def _task(n: int = 10, seed: int = 42) -> LinearSystemTask:
    rng = np.random.default_rng(seed)
    A = _spd_matrix(n, seed)
    b = rng.standard_normal(n)
    return LinearSystemTask(
        task_id=f"ls_{n}_{seed}",
        domain_name="linear_system",
        input_artifact=f"ls_{n}_{seed}",
        metadata={"A": A, "b": b},
    )


class TestLinearSystemE2E:
    def test_oracle_is_exact(self):
        task = _task(10)
        oracle = LinearSystemOracle()
        result = oracle.solve(task)
        assert result["residual"] < 1e-10

    def test_surrogate_then_projection(self):
        task = _task(8, seed=99)
        surr = LinearSystemSurrogate(use_jacobi=True)
        pred = surr.predict(task)

        proj = LinearSystemProjection()
        result = proj.project(task, pred, budget=500)
        assert result["residual"] < pred["residual"]
        assert result["converged"]

    def test_full_pipeline(self):
        task = _task(6, seed=7)
        oracle = LinearSystemOracle()
        surr = LinearSystemSurrogate()
        proj = LinearSystemProjection()
        ev = LinearSystemEvaluator(tolerance=1e-6)

        pred = surr.predict(task)
        projected = proj.project(task, pred, budget=500)
        evaluation = ev.evaluate(task, projected)
        assert evaluation["correct"]

    def test_fingerprint_deterministic(self):
        t1 = _task(5, seed=42)
        t2 = _task(5, seed=42)
        assert t1.fingerprint() == t2.fingerprint()

    def test_fingerprint_differs_for_different_input(self):
        t1 = _task(5, seed=42)
        t2 = _task(5, seed=99)
        assert t1.fingerprint() != t2.fingerprint()

    def test_node_edge_count(self):
        task = _task(7)
        assert task.node_count() == 7
        assert task.edge_count() > 0

    def test_evaluator_rejects_bad_solution(self):
        task = _task(10)
        ev = LinearSystemEvaluator(tolerance=1e-6)
        result = ev.evaluate(task, {"solution": np.zeros(10)})
        assert not result["correct"]
