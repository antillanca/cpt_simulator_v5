"""CORE v3.3 — Explainability Runtime.

Deterministic post-hoc explanations for runtime results.
This is NOT a language model. This is deterministic rule-based explanation.

SAFETY GUARANTEE:
  - Explanations do not alter execution
  - They are observability records only
  - Same input -> same explanation
  - No free-form text generation
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from core_runtime.core.trustworthiness_runtime import (
    TrustLevel,
    TrustworthinessAudit,
)


# ---------------------------------------------------------------------------
# ExecutionExplanation — frozen, structured
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionExplanation:
    """Deterministic post-hoc explanation for a runtime execution."""
    task_hash: str
    summary: str
    key_factors: tuple[str, ...]
    topology_family: str
    confidence_score: float
    trust_level: str
    projected_iterations: int
    residual: float
    explanation_type: str
    metadata: dict[str, Any]

    def fingerprint(self) -> str:
        """SHA-256 fingerprint for traceability."""
        payload = json.dumps({
            "task_hash": self.task_hash,
            "summary": self.summary,
            "key_factors": list(self.key_factors),
            "topology_family": self.topology_family,
            "confidence_score": round(self.confidence_score, 12),
            "trust_level": self.trust_level,
            "projected_iterations": self.projected_iterations,
            "residual": round(self.residual, 12),
            "explanation_type": self.explanation_type,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_hash": self.task_hash,
            "summary": self.summary,
            "key_factors": list(self.key_factors),
            "topology_family": self.topology_family,
            "confidence_score": self.confidence_score,
            "trust_level": self.trust_level,
            "projected_iterations": self.projected_iterations,
            "residual": self.residual,
            "explanation_type": self.explanation_type,
            "fingerprint": self.fingerprint(),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Explanation types
# ---------------------------------------------------------------------------

EXPLANATION_TYPES = frozenset({
    "exact_cache_hit",
    "warmstart_assisted",
    "projection_escalated",
    "ood_conservative",
    "unstable_convergence",
    "certain_execution",
    "indeterminate_execution",
    "transitional_execution",
})


# ---------------------------------------------------------------------------
# Deterministic explanation templates
# ---------------------------------------------------------------------------

_SUMMARIES = {
    "exact_cache_hit": "Exact cache hit produced deterministic reuse without rerun.",
    "warmstart_assisted": "Warmstart reduced initial residual but remained below the acceptance threshold.",
    "projection_escalated": "High residual and oscillatory convergence required projection escalation.",
    "ood_conservative": "OOD topology triggered conservative budget allocation.",
    "unstable_convergence": "Unstable convergence detected with multiple trajectory corrections.",
    "certain_execution": "Standard execution completed with high confidence and low residual.",
    "indeterminate_execution": "Multiple trust flags raised — execution requires audit review.",
    "transitional_execution": "Marginal residual detected — execution acceptable but flagged for monitoring.",
}


# ---------------------------------------------------------------------------
# ExplainabilityRuntime — deterministic rule-based explainer
# ---------------------------------------------------------------------------

class ExplainabilityRuntime:
    """Generate deterministic post-hoc explanations for runtime results.

    Rules:
    - Cache hit -> exact_cache_hit
    - Warmstart used and accepted -> warmstart_assisted
    - Escalation triggered -> projection_escalated
    - OOD execution -> ood_conservative
    - Unstable trajectory -> unstable_convergence
    - CERTAIN trust -> certain_execution
    - TRANSITIONAL trust -> transitional_execution
    - INDETERMINATE trust -> indeterminate_execution
    """

    def explain(
        self,
        audit: TrustworthinessAudit,
        result: dict[str, Any] | None = None,
        trajectory: dict[str, Any] | None = None,
        routing: dict[str, Any] | None = None,
    ) -> ExecutionExplanation:
        """Generate a deterministic explanation from audit + runtime state."""
        result = result or {}
        trajectory = trajectory or {}
        routing = routing or {}

        # Determine explanation type from routing first
        routing_action = routing.get("action", "")
        cache_hit = routing_action == "exact_cache_hit"

        key_factors: list[str] = []
        explanation_type: str

        if cache_hit:
            explanation_type = "exact_cache_hit"
            key_factors.append("exact_cache_hit")
            key_factors.append(f"trust_level={audit.trust_level.value}")
        elif audit.escalation_required:
            explanation_type = "projection_escalated"
            key_factors.append("escalation_triggered")
            key_factors.append(f"residual={audit.final_residual:.4e}")
        elif trajectory.get("trajectory_class") in ("divergence_risk", "stalled"):
            explanation_type = "unstable_convergence"
            key_factors.append(f"trajectory={trajectory.get('trajectory_class')}")
        elif routing.get("ood", False) or any(
            f.value == "ood_execution" for f in audit.flags
        ):
            explanation_type = "ood_conservative"
            key_factors.append("ood_execution")
            key_factors.append(f"budget={routing.get('projection_budget', 'N/A')}")
        elif routing_action == "warmstart_projection":
            explanation_type = "warmstart_assisted"
            key_factors.append("warmstart_used")
            key_factors.append(f"similarity={routing.get('retrieval_similarity', 0):.3f}")
        elif audit.trust_level == TrustLevel.CERTAIN:
            explanation_type = "certain_execution"
            key_factors.append("high_confidence")
            key_factors.append(f"residual={audit.final_residual:.4e}")
        elif audit.trust_level == TrustLevel.TRANSITIONAL:
            explanation_type = "transitional_execution"
            key_factors.append("marginal_residual")
            key_factors.append(f"flags={[f.value for f in audit.flags]}")
        else:
            explanation_type = "indeterminate_execution"
            key_factors.append("multiple_flags")
            key_factors.append(f"flags={[f.value for f in audit.flags]}")

        # Add common factors
        key_factors.append(f"confidence={audit.confidence_score:.3f}")
        key_factors.append(f"iterations={audit.projection_iterations}")

        topology_family = "unknown"
        if audit.metadata and "topology_family" in audit.metadata:
            topology_family = audit.metadata.get("topology_family", trajectory.get("topology_family", "unknown"))
        elif trajectory:
            topology_family = trajectory.get("topology_family", "unknown")

        summary = _SUMMARIES.get(explanation_type, "Execution completed.")

        return ExecutionExplanation(
            task_hash=audit.task_hash,
            summary=summary,
            key_factors=tuple(key_factors),
            topology_family=topology_family,
            confidence_score=audit.confidence_score,
            trust_level=audit.trust_level.value,
            projected_iterations=audit.projection_iterations,
            residual=audit.final_residual,
            explanation_type=explanation_type,
            metadata={
                "routing_action": routing_action,
            },
        )
