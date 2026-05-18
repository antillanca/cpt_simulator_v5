"""CPT Core Specification — Projection Result Contract.

Frozen dataclass for physics projection outcomes. Every projection call
must produce a ProjectionResult with deterministic fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProjectionResult:
    """Canonical outcome of a physics projection step.

    Captures all physical invariants before/after projection, convergence
    status, and timing. Deterministic fingerprint over all fields.
    """

    iterations: int
    initial_kcl_residual: float
    final_kcl_residual: float
    initial_kvl_residual: float
    final_kvl_residual: float
    initial_power_residual: float
    final_power_residual: float
    converged: bool
    used_virtual_node: bool
    projection_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- Serialization -------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "initial_kcl_residual": round(self.initial_kcl_residual, 12),
            "final_kcl_residual": round(self.final_kcl_residual, 12),
            "initial_kvl_residual": round(self.initial_kvl_residual, 12),
            "final_kvl_residual": round(self.final_kvl_residual, 12),
            "initial_power_residual": round(self.initial_power_residual, 12),
            "final_power_residual": round(self.final_power_residual, 12),
            "converged": self.converged,
            "used_virtual_node": self.used_virtual_node,
            "projection_time_ms": round(self.projection_time_ms, 3),
            "metadata": _sort_dict(self.metadata),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "ProjectionResult":
        return cls(
            iterations=data["iterations"],
            initial_kcl_residual=data["initial_kcl_residual"],
            final_kcl_residual=data["final_kcl_residual"],
            initial_kvl_residual=data["initial_kvl_residual"],
            final_kvl_residual=data["final_kvl_residual"],
            initial_power_residual=data["initial_power_residual"],
            final_power_residual=data["final_power_residual"],
            converged=data["converged"],
            used_virtual_node=data["used_virtual_node"],
            projection_time_ms=data["projection_time_ms"],
            metadata=data.get("metadata", {}),
        )

    # -- Fingerprint ---------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        """Deterministic SHA-256 over canonical JSON representation."""
        payload = json.dumps(self.to_json_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    # -- Validation ----------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.iterations < 0:
            errors.append("iterations must be non-negative")
        if self.final_kcl_residual < 0:
            errors.append("final_kcl_residual must be non-negative")
        if self.final_kvl_residual < 0:
            errors.append("final_kvl_residual must be non-negative")
        if self.projection_time_ms < 0:
            errors.append("projection_time_ms must be non-negative")
        return errors


# ---------------------------------------------------------------------------
# Conversion from existing projection_effort.ProjectionEffort
# ---------------------------------------------------------------------------

def from_projection_effort(effort: Any, config_used: dict[str, Any] | None = None) -> ProjectionResult:
    """Convert existing ProjectionEffort dataclass to canonical contract."""
    return ProjectionResult(
        iterations=effort.iterations_to_converge,
        initial_kcl_residual=effort.initial_residual,
        final_kcl_residual=effort.final_residual,
        initial_kvl_residual=0.0,  # not tracked in ProjectionEffort
        final_kvl_residual=0.0,
        initial_power_residual=0.0,
        final_power_residual=0.0,
        converged=effort.iterations_to_converge < effort.initial_residual * 1e6,  # heuristic
        used_virtual_node=config_used.get("virtual_node_enabled", True) if config_used else True,
        projection_time_ms=0.0,
        metadata={"correction_distance": effort.correction_distance, "residual_decay_rate": effort.residual_decay_rate},
    )


def _sort_dict(d: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k in sorted(d):
        v = d[k]
        result[k] = _sort_dict(v) if isinstance(v, dict) else v
    return result
