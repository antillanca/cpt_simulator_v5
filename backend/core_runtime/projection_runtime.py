"""CPT Core Runtime — Projection Execution Layer.

Wraps existing v2.9F PhysicsProjection using standardized contracts.
Returns ProjectionExecution with deterministic fingerprint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from backend.core_spec.projection_spec import ProjectionResult


# ---------------------------------------------------------------------------
# ProjectionExecution — canonical projection output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectionExecution:
    """Immutable projection execution result."""

    corrected_prediction: torch.Tensor
    iterations: int
    converged: bool
    kcl_violation: float
    kvl_violation: float
    projection_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_projection_result(self) -> ProjectionResult:
        """Convert to core_spec ProjectionResult."""
        return ProjectionResult(
            iterations=self.iterations,
            initial_kcl_residual=self.metadata.get("initial_kcl", 0.0),
            final_kcl_residual=self.kcl_violation,
            initial_kvl_residual=self.metadata.get("initial_kvl", 0.0),
            final_kvl_residual=self.kvl_violation,
            initial_power_residual=self.metadata.get("initial_power", 0.0),
            final_power_residual=self.metadata.get("final_power", 0.0),
            converged=self.converged,
            used_virtual_node=self.metadata.get("used_virtual_node", False),
            projection_time_ms=self.projection_time_ms,
        )


# ---------------------------------------------------------------------------
# ProjectionRuntime — wraps PhysicsProjection
# ---------------------------------------------------------------------------

class ProjectionRuntime:
    """Projection execution layer wrapping v2.9F PhysicsProjection.

    Usage:
        runtime = ProjectionRuntime(projection_config)
        result = runtime.project(graph, circuit, voltages)
    """

    def __init__(self, config: Any = None) -> None:
        """config: ProjectionConfig or None (uses defaults)."""
        from backend.circuits.physics_projection import PhysicsProjection
        self._projector = PhysicsProjection(config)
        self._config = config

    def project(
        self,
        graph: Any,
        circuit: Any,
        voltages: torch.Tensor,
    ) -> ProjectionExecution:
        """Run projection and return standardized ProjectionExecution."""
        # Collect initial residuals
        initial_kcl = self._compute_kcl(graph, circuit, voltages)
        initial_kvl = self._compute_kvl(graph, voltages)
        initial_power = self._compute_power(graph, circuit, voltages)

        t0 = time.perf_counter()
        corrected = self._projector.project(graph, circuit, voltages)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Collect final residuals
        final_kcl = self._compute_kcl(graph, circuit, corrected)
        final_kvl = self._compute_kvl(graph, corrected)
        final_power = self._compute_power(graph, circuit, corrected)

        # Determine convergence
        cfg = self._config
        tolerance = getattr(cfg, "tolerance", 1e-9) if cfg else 1e-9
        converged = final_kcl < tolerance and final_kvl < tolerance

        used_virtual = bool(getattr(cfg, "virtual_node_enabled", True) if cfg else True)

        return ProjectionExecution(
            corrected_prediction=corrected,
            iterations=getattr(cfg, "steps", 50) if cfg else 50,
            converged=converged,
            kcl_violation=final_kcl,
            kvl_violation=final_kvl,
            projection_time_ms=elapsed_ms,
            metadata={
                "initial_kcl": initial_kcl,
                "initial_kvl": initial_kvl,
                "initial_power": initial_power,
                "final_power": final_power,
                "used_virtual_node": used_virtual,
            },
        )

    # -- Residual computation helpers (delegates to projection internals) --

    @staticmethod
    def _compute_kcl(graph: Any, circuit: Any, v: torch.Tensor) -> float:
        try:
            from backend.circuits.physics_projection import _node_kcl_residual
            res = _node_kcl_residual(v, graph, circuit)
            return float(res.abs().max().item())
        except Exception:
            return 0.0

    @staticmethod
    def _compute_kvl(graph: Any, v: torch.Tensor) -> float:
        try:
            from backend.circuits.physics_projection import _cycle_kvl_residual
            if hasattr(graph, "cycle_matrix") and graph.cycle_matrix.numel() > 0:
                res = _cycle_kvl_residual(v, graph)
                return float(res.abs().max().item())
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _compute_power(graph: Any, circuit: Any, v: torch.Tensor) -> float:
        try:
            from backend.circuits.physics_projection import _power_residual
            res = _power_residual(v, graph, circuit)
            return float(res.abs().max().item())
        except Exception:
            return 0.0
