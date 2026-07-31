"""CORE v3.3 — Structured Failure Reports.

Deterministic, structured failure data replacing free-text output.
No LLM-generated text. No free-form narrative.
Only structured, deterministic reports.

SAFETY GUARANTEE:
  - Failure reports do not alter execution
  - They are observability records only
  - Same input -> same failure report
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Probable cause codes
# ---------------------------------------------------------------------------

PROBABLE_CAUSES = frozenset({
    "bridge_high_impedance",
    "radial_propagation_limit",
    "extreme_resistance_instability",
    "source_saturation",
    "ood_topology_gap",
    "projection_stagnation",
    "oracle_timeout",
    "surrogate_instability",
    "unknown",
})


# ---------------------------------------------------------------------------
# Recommended action codes
# ---------------------------------------------------------------------------

RECOMMENDED_ACTIONS = frozenset({
    "increase_projection_budget",
    "force_oracle_verification",
    "route_to_uncertainty_memory",
    "mark_for_audit",
    "fallback_to_safe_path",
})


# ---------------------------------------------------------------------------
# FailureReport — frozen, deterministic
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FailureReport:
    """Structured deterministic failure report."""
    task_hash: str
    error_type: str
    conditions: dict[str, Any]
    probable_cause: str
    recommended_action: str
    topology_signature: str
    confidence_score: float
    residual: float
    projection_iterations: int
    deterministic: bool = True

    def fingerprint(self) -> str:
        """SHA-256 fingerprint for traceability."""
        payload = json.dumps({
            "task_hash": self.task_hash,
            "error_type": self.error_type,
            "conditions": dict(sorted(self.conditions.items())),
            "probable_cause": self.probable_cause,
            "recommended_action": self.recommended_action,
            "topology_signature": self.topology_signature,
            "confidence_score": round(self.confidence_score, 12),
            "residual": round(self.residual, 12),
            "projection_iterations": self.projection_iterations,
            "deterministic": self.deterministic,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_hash": self.task_hash,
            "error_type": self.error_type,
            "conditions": self.conditions,
            "probable_cause": self.probable_cause,
            "recommended_action": self.recommended_action,
            "topology_signature": self.topology_signature,
            "confidence_score": self.confidence_score,
            "residual": self.residual,
            "projection_iterations": self.projection_iterations,
            "deterministic": self.deterministic,
            "fingerprint": self.fingerprint(),
        }


# ---------------------------------------------------------------------------
# Deterministic cause inference rules
# ---------------------------------------------------------------------------

def _infer_cause(
    error_type: str,
    residual: float,
    confidence_score: float,
    projection_iterations: int,
    ood: bool,
) -> str:
    """Infer probable cause from deterministic rules only."""
    if error_type == "oracle_timeout":
        return "oracle_timeout"
    if error_type == "surrogate_instability":
        return "surrogate_instability"
    if ood:
        return "ood_topology_gap"
    if residual > 1.0 and projection_iterations > 15:
        return "projection_stagnation"
    if residual > 0.5:
        return "extreme_resistance_instability"
    if confidence_score < 0.3:
        return "source_saturation"
    return "unknown"


def _infer_action(
    probable_cause: str,
    residual: float,
    confidence_score: float,
    ood: bool,
) -> str:
    """Infer recommended action from deterministic rules only."""
    if probable_cause == "oracle_timeout":
        return "force_oracle_verification"
    if probable_cause == "surrogate_instability":
        return "fallback_to_safe_path"
    if probable_cause == "ood_topology_gap":
        return "route_to_uncertainty_memory"
    if probable_cause == "projection_stagnation":
        return "increase_projection_budget"
    if confidence_score < 0.3:
        return "mark_for_audit"
    return "fallback_to_safe_path"


# ---------------------------------------------------------------------------
# generate_failure_report — deterministic factory
# ---------------------------------------------------------------------------

def generate_failure_report(
    task: Any,
    error_type: str,
    runtime_state: dict[str, Any],
) -> FailureReport:
    """Generate a deterministic structured failure report.

    Args:
        task: Any object with .metadata dict (for task_hash extraction)
              or a dict with 'task_hash' key.
        error_type: The error classification string.
        runtime_state: Dict with keys:
            residual, confidence_score, projection_iterations,
            topology_signature, ood, escalation_required, etc.

    Returns:
        Frozen FailureReport with deterministic cause and action.
    """
    # Extract task_hash
    if isinstance(task, dict):
        task_hash = task.get("task_hash", "unknown")
    elif hasattr(task, "metadata") and isinstance(task.metadata, dict):
        task_hash = task.metadata.get("task_hash", "unknown")
    elif hasattr(task, "task_hash"):
        task_hash = task.task_hash
    else:
        task_hash = "unknown"

    residual = float(runtime_state.get("residual", 0.0))
    confidence_score = float(runtime_state.get("confidence_score", 0.0))
    projection_iterations = int(runtime_state.get("projection_iterations", 0))
    topology_signature = str(runtime_state.get("topology_signature", "unknown"))
    ood = bool(runtime_state.get("ood", False))

    probable_cause = _infer_cause(
        error_type, residual, confidence_score, projection_iterations, ood,
    )
    recommended_action = _infer_action(probable_cause, residual, confidence_score, ood)

    conditions = {
        "residual": residual,
        "confidence_score": confidence_score,
        "projection_iterations": projection_iterations,
        "ood": ood,
        "escalation_required": bool(runtime_state.get("escalation_required", False)),
        "trajectory_class": runtime_state.get("trajectory_class", "unknown"),
    }

    return FailureReport(
        task_hash=task_hash,
        error_type=error_type,
        conditions=conditions,
        probable_cause=probable_cause,
        recommended_action=recommended_action,
        topology_signature=topology_signature,
        confidence_score=confidence_score,
        residual=residual,
        projection_iterations=projection_iterations,
        deterministic=True,
    )
