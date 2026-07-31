"""CORE v3.3 — Trustworthiness Runtime.

Deterministic operational trust classification.

SAFETY GUARANTEE:
  This module is an OBSERVABILITY layer only. It MUST NOT:
  - alter projection results
  - alter exact cache behavior
  - alter retrieval semantics
  - alter scheduler physics
  - interpret human values / morality / ideology
  - change execution outputs based on audit results

  Trust scores are observability signals, not execution modifiers.
  The router may CONSULT trust levels for budget allocation,
  but projection remains the final authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# TrustLevel — classification of operational trustworthiness
# ---------------------------------------------------------------------------

class TrustLevel(Enum):
    CERTAIN = "certain"
    TRANSITIONAL = "transitional"
    INDETERMINATE = "indeterminate"


# ---------------------------------------------------------------------------
# TrustFlag — individual trust indicators
# ---------------------------------------------------------------------------

class TrustFlag(Enum):
    HIGH_RESIDUAL_HEURISTIC = "high_residual_heuristic"
    UNSTABLE_CONVERGENCE = "unstable_convergence"
    ESCALATION_TRIGGERED = "escalation_triggered"
    LOW_CONFIDENCE = "low_confidence"
    OOD_EXECUTION = "ood_execution"
    HIGH_PROJECTION_EFFORT = "high_projection_effort"


# ---------------------------------------------------------------------------
# Thresholds — deterministic, auditable
# ---------------------------------------------------------------------------

_TRUST_THRESHOLDS = {
    "residual_high": 0.01,
    "confidence_low": 0.5,
    "unstable_trajectories": frozenset({"divergence_risk", "stalled"}),
    "projection_effort_ratio": 1.0,  # iterations / family_budget
}


# ---------------------------------------------------------------------------
# TrustworthinessAudit — frozen, fingerprintable
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrustworthinessAudit:
    """Deterministic trust audit for a runtime execution.

    SAFETY: This is a read-only record. It cannot influence execution.
    """
    task_hash: str
    trust_level: TrustLevel
    flags: tuple[TrustFlag, ...]
    confidence_score: float
    uncertainty_score: float
    projection_iterations: int
    final_residual: float
    escalation_required: bool
    deterministic: bool = True
    metadata: dict[str, Any] | None = None

    def fingerprint(self) -> str:
        """SHA-256 fingerprint of the audit — deterministic."""
        payload = json.dumps({
            "task_hash": self.task_hash,
            "trust_level": self.trust_level.value,
            "flags": sorted(f.value for f in self.flags),
            "confidence_score": round(self.confidence_score, 12),
            "uncertainty_score": round(self.uncertainty_score, 12),
            "projection_iterations": self.projection_iterations,
            "final_residual": round(self.final_residual, 12),
            "escalation_required": self.escalation_required,
            "deterministic": self.deterministic,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict representation."""
        return {
            "task_hash": self.task_hash,
            "trust_level": self.trust_level.value,
            "flags": [f.value for f in self.flags],
            "confidence_score": self.confidence_score,
            "uncertainty_score": self.uncertainty_score,
            "projection_iterations": self.projection_iterations,
            "final_residual": self.final_residual,
            "escalation_required": self.escalation_required,
            "deterministic": self.deterministic,
            "fingerprint": self.fingerprint(),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# audit_execution — deterministic trust classification
# ---------------------------------------------------------------------------

def audit_execution(
    task_hash: str,
    confidence_score: float,
    uncertainty_score: float,
    projection_iterations: int,
    final_residual: float,
    trajectory_class: str,
    escalation_required: bool,
    ood: bool,
    family_budget: int | None = None,
) -> TrustworthinessAudit:
    """Produce a deterministic trust audit for a runtime execution.

    Classification rules:
      - No flags -> CERTAIN
      - Exactly one flag, and it is HIGH_RESIDUAL_HEURISTIC -> TRANSITIONAL
      - Otherwise -> INDETERMINATE

    The audit is a pure function of its inputs. Same inputs -> same audit.
    """
    flags: list[TrustFlag] = []

    # Rule: residual > threshold
    if final_residual > _TRUST_THRESHOLDS["residual_high"]:
        flags.append(TrustFlag.HIGH_RESIDUAL_HEURISTIC)

    # Rule: low confidence
    if confidence_score < _TRUST_THRESHOLDS["confidence_low"]:
        flags.append(TrustFlag.LOW_CONFIDENCE)

    # Rule: unstable trajectory
    if trajectory_class in _TRUST_THRESHOLDS["unstable_trajectories"]:
        flags.append(TrustFlag.UNSTABLE_CONVERGENCE)

    # Rule: escalation triggered
    if escalation_required:
        flags.append(TrustFlag.ESCALATION_TRIGGERED)

    # Rule: OOD execution
    if ood:
        flags.append(TrustFlag.OOD_EXECUTION)

    # Rule: high projection effort
    if family_budget is not None and family_budget > 0:
        ratio = projection_iterations / family_budget
        if ratio > _TRUST_THRESHOLDS["projection_effort_ratio"]:
            flags.append(TrustFlag.HIGH_PROJECTION_EFFORT)
    elif projection_iterations > 15:
        # Fallback if no family budget: hard threshold
        flags.append(TrustFlag.HIGH_PROJECTION_EFFORT)

    # Classification
    flag_tuple = tuple(flags)
    if len(flags) == 0:
        trust_level = TrustLevel.CERTAIN
    elif len(flags) == 1 and flags[0] == TrustFlag.HIGH_RESIDUAL_HEURISTIC:
        trust_level = TrustLevel.TRANSITIONAL
    else:
        trust_level = TrustLevel.INDETERMINATE

    return TrustworthinessAudit(
        task_hash=task_hash,
        trust_level=trust_level,
        flags=flag_tuple,
        confidence_score=confidence_score,
        uncertainty_score=uncertainty_score,
        projection_iterations=projection_iterations,
        final_residual=final_residual,
        escalation_required=escalation_required,
        deterministic=True,
        metadata={
            "trajectory_class": trajectory_class,
            "ood": ood,
            "family_budget": family_budget,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_trust(audit: TrustworthinessAudit) -> str:
    """Human-readable trust classification string."""
    return audit.trust_level.value


def flags_summary(audit: TrustworthinessAudit) -> str:
    """Comma-separated flag names."""
    if not audit.flags:
        return "none"
    return ",".join(f.value for f in audit.flags)
