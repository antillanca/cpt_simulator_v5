"""CPT Linear System Domain -- proof-of-concept for CORE SDK.

Tiny but real domain: solve Ax = b systems.
Oracle: numpy exact solve. Surrogate: zero-initial guess.
Projection: iterative refinement. Evaluator: residual norm.

This domain proves the SDK is sufficient for a non-circuit domain.
"""

from __future__ import annotations

from core_runtime.core.domain_sdk import (
    DomainTaskBase,
    DomainOracle,
    DomainSurrogate,
    DomainProjection,
    DomainEvaluator,
    register_domain,
)
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import time


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
    import hashlib
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
# Surrogate: trivial initial guess
# ============================================================

class LinearSystemSurrogate:
    """Trivial surrogate: zero vector initial guess."""

    def __init__(self, use_jacobi: bool = True):
        self._use_jacobi = use_jacobi

    def predict(self, task: DomainTaskBase) -> dict[str, Any]:
        A = task.metadata["A"]
        b = task.metadata["b"]
        t0 = time.perf_counter()
        n = A.shape[0]

        if self._use_jacobi:
            # Jacobi preconditioner: x_i = b_i / A_ii
            diag = np.diag(A).copy()
            diag[diag == 0] = 1.0  # avoid div-by-zero
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
# Projection: iterative refinement (Gauss-Seidel-like)
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

        for i in range(budget):
            residual_vec = b - A @ x
            residual = float(np.linalg.norm(residual_vec))
            if residual < self._tol:
                converged = True
                iterations = i + 1
                break
            # Gradient step: move toward reducing ||b - Ax||^2
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
# Domain registration
# ============================================================

register_domain(
    domain_name="linear_system",
    oracle=LinearSystemOracle,
    surrogate=LinearSystemSurrogate,
    projection=LinearSystemProjection,
    evaluator=LinearSystemEvaluator,
)

__version__ = "0.1.0"
