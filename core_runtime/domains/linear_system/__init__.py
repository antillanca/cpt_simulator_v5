"""CORE Linear System Domain -- proof-of-concept for full SDK validation.

Tiny but real domain: solve Ax = b systems.
Oracle: numpy exact solve. Surrogate: Jacobi preconditioner.
Projection: iterative refinement (gradient descent on residual).
Evaluator: residual norm. Confidence: residual-based.

This domain proves the SDK is sufficient for a non-circuit domain
and exercises the COMPLETE SDK pipeline:
  task -> surrogate -> projection -> memory -> trace -> evaluator
"""

from __future__ import annotations

from core_runtime.core.domain_sdk import (
    DomainTaskBase,
    DomainOracle,
    DomainSurrogate,
    DomainProjection,
    DomainEvaluator,
    DomainConfidence,
    register_domain,
)
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import time
import hashlib
import json


# ============================================================
# Linear System Task
# ============================================================

@dataclass(frozen=True)
class LinearSystemTask(DomainTaskBase):
    """Task: solve Ax = b for a given matrix A and vector b."""

    def fingerprint(self) -> str:
        """Hash from matrix content."""
        a_b64 = _array_fingerprint(self.metadata.get("A"))
        b_b64 = _array_fingerprint(self.metadata.get("b"))
        return f"ls-{a_b64}-{b_b64}"

    def node_count(self) -> int:
        a = self.metadata.get("A")
        return a.shape[0] if a is not None else 0

    def edge_count(self) -> int:
        a = self.metadata.get("A")
        return int(np.count_nonzero(a)) if a is not None else 0


def _array_fingerprint(arr: np.ndarray | None) -> str:
    if arr is None:
        return "none"
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


# ============================================================
# Oracle: exact numpy solve
# ============================================================

class LinearSystemOracle:
    """Exact solver using numpy.linalg.solve."""

    def solve(self, task: DomainTaskBase) -> dict[str, Any]:
        A = task.metadata["A"]
        b = task.metadata["b"]
        t0 = time.perf_counter()
        x = np.linalg.solve(A, b)
        runtime_ms = (time.perf_counter() - t0) * 1000
        residual = float(np.linalg.norm(b - A @ x))
        return {
            "solution": x,
            "residual": residual,
            "runtime_ms": runtime_ms,
            "method": "numpy.linalg.solve",
        }


# ============================================================
# Surrogate: Jacobi preconditioner approximation
# ============================================================

class LinearSystemSurrogate:
    """Surrogate: Jacobi preconditioner as fast approximation."""

    def __init__(self, use_jacobi: bool = True):
        self._use_jacobi = use_jacobi

    def predict(self, task: DomainTaskBase) -> dict[str, Any]:
        A = task.metadata["A"]
        b = task.metadata["b"]
        t0 = time.perf_counter()
        n = A.shape[0]

        if self._use_jacobi:
            diag = np.diag(A).copy()
            diag[diag == 0] = 1.0
            x = b / diag
        else:
            x = np.zeros(n)

        runtime_ms = (time.perf_counter() - t0) * 1000
        residual = float(np.linalg.norm(b - A @ x))
        return {
            "prediction": x,
            "residual": residual,
            "runtime_ms": runtime_ms,
            "method": "jacobi_preconditioner" if self._use_jacobi else "zero_init",
        }


# ============================================================
# Projection: iterative refinement (gradient descent on residual)
# ============================================================

class LinearSystemProjection:
    """Projection via iterative refinement.

    Each iteration: x_{k+1} = x_k + alpha * A^T (b - A x_k)
    This is gradient descent on ||b - Ax||^2.
    """

    def __init__(self, step_size: float | None = None, tol: float = 1e-10):
        self._step_size = step_size
        self._tol = tol

    def project(
        self, task: DomainTaskBase, prediction: dict[str, Any], budget: int,
    ) -> dict[str, Any]:
        A = task.metadata["A"]
        b = task.metadata["b"]
        x = prediction["prediction"].copy()

        # Adaptive step size: 1 / (2 * spectral_norm^2) for stability
        if self._step_size is None:
            try:
                spectral_sq = float(np.linalg.norm(A.T @ A, ord=2))
                alpha = 0.9 / max(spectral_sq, 1e-6)
            except Exception:
                alpha = 0.001
        else:
            alpha = self._step_size

        t0 = time.perf_counter()
        At = A.T
        converged = False
        iterations = 0
        trajectory = []

        for i in range(budget):
            residual_vec = b - A @ x
            residual = float(np.linalg.norm(residual_vec))
            trajectory.append(residual)
            if residual < self._tol:
                converged = True
                iterations = i + 1
                break
            x = x + alpha * (At @ residual_vec)
            iterations = i + 1

        final_residual = float(np.linalg.norm(b - A @ x))
        if not converged:
            converged = final_residual < self._tol

        runtime_ms = (time.perf_counter() - t0) * 1000
        return {
            "solution": x,
            "residual": final_residual,
            "iterations": iterations,
            "converged": converged,
            "runtime_ms": runtime_ms,
            "method": "iterative_refinement",
            "trajectory": trajectory,
        }


# ============================================================
# Evaluator: residual norm + correctness check
# ============================================================

class LinearSystemEvaluator:
    """Evaluates linear system solutions by residual norm."""

    def __init__(self, tolerance: float = 1e-6):
        self._tol = tolerance

    def evaluate(
        self, task: DomainTaskBase, solution: dict[str, Any],
    ) -> dict[str, Any]:
        A = task.metadata["A"]
        b = task.metadata["b"]
        x = solution.get("solution", solution.get("prediction"))
        if x is None:
            return {"residual": float("inf"), "correct": False, "metrics": {}}

        residual = float(np.linalg.norm(b - A @ x))
        return {
            "residual": residual,
            "correct": residual < self._tol,
            "metrics": {
                "relative_residual": residual / max(float(np.linalg.norm(b)), 1e-12),
                "solution_norm": float(np.linalg.norm(x)),
            },
        }


# ============================================================
# Confidence: residual-based confidence scoring
# ============================================================

class LinearSystemConfidence:
    """Confidence scorer for linear system predictions.

    Scores based on inverse residual: lower residual = higher confidence.
    Maps residual to [0, 1] using exponential decay.
    """

    def __init__(self, scale: float = 1.0):
        self._scale = scale

    def score(self, task: DomainTaskBase, prediction: dict[str, Any]) -> float:
        residual = prediction.get("residual", float("inf"))
        # Exponential decay: confidence = exp(-residual / scale)
        return float(np.exp(-residual / self._scale))


# ============================================================
# Execution trace helper
# ============================================================

class LinearSystemTrace:
    """Trace emitter for linear system pipeline execution.

    Produces a deterministic execution trace that can be stored
    in the exact cache or retrieval memory.
    """

    @staticmethod
    def build_trace(
        task: LinearSystemTask,
        surrogate_result: dict[str, Any],
        projection_result: dict[str, Any],
        evaluation_result: dict[str, Any],
        confidence_score: float | None = None,
    ) -> dict[str, Any]:
        """Build a complete execution trace for a linear system task."""
        return {
            "task_id": task.task_id,
            "domain_name": task.domain_name,
            "fingerprint": task.fingerprint(),
            "node_count": task.node_count(),
            "edge_count": task.edge_count(),
            "surrogate_residual": surrogate_result.get("residual"),
            "surrogate_method": surrogate_result.get("method"),
            "surrogate_runtime_ms": surrogate_result.get("runtime_ms"),
            "projection_residual": projection_result.get("residual"),
            "projection_iterations": projection_result.get("iterations"),
            "projection_converged": projection_result.get("converged"),
            "projection_method": projection_result.get("method"),
            "projection_runtime_ms": projection_result.get("runtime_ms"),
            "evaluation_residual": evaluation_result.get("residual"),
            "evaluation_correct": evaluation_result.get("correct"),
            "confidence_score": confidence_score,
            "trajectory_length": len(projection_result.get("trajectory", [])),
        }


# ============================================================
# Full pipeline executor
# ============================================================

def execute_linear_system_pipeline(
    task: LinearSystemTask,
    surrogate: LinearSystemSurrogate | None = None,
    projection: LinearSystemProjection | None = None,
    evaluator: LinearSystemEvaluator | None = None,
    confidence: LinearSystemConfidence | None = None,
    budget: int = 500,
) -> dict[str, Any]:
    """Execute the full linear system pipeline through the SDK.

    task -> surrogate -> projection -> evaluator -> confidence -> trace

    This is the canonical pipeline that proves the SDK works end-to-end
    for a non-circuit domain.
    """
    if surrogate is None:
        surrogate = LinearSystemSurrogate()
    if projection is None:
        projection = LinearSystemProjection()
    if evaluator is None:
        evaluator = LinearSystemEvaluator()
    if confidence is None:
        confidence = LinearSystemConfidence()

    # 1. Surrogate prediction
    surrogate_result = surrogate.predict(task)

    # 2. Projection correction
    projection_result = projection.project(task, surrogate_result, budget)

    # 3. Evaluation
    evaluation_result = evaluator.evaluate(task, projection_result)

    # 4. Confidence scoring
    confidence_score = confidence.score(task, surrogate_result)

    # 5. Build execution trace
    trace = LinearSystemTrace.build_trace(
        task, surrogate_result, projection_result, evaluation_result, confidence_score,
    )

    return {
        "surrogate": surrogate_result,
        "projection": projection_result,
        "evaluation": evaluation_result,
        "confidence": confidence_score,
        "trace": trace,
    }


# ============================================================
# Domain registration -- FULL SDK
# ============================================================

register_domain(
    domain_name="linear_system",
    oracle=LinearSystemOracle,
    surrogate=LinearSystemSurrogate,
    projection=LinearSystemProjection,
    evaluator=LinearSystemEvaluator,
    confidence=LinearSystemConfidence,
)

__version__ = "0.2.0"
