"""CPT Core Runtime — Capability Router.

Deterministic rule-based routing. NO machine learning routing yet.

Rules:
  IF exact cache hit    → return cached result immediately
  ELIF confidence high  → small projection budget
  ELIF likely OOD       → large projection budget
  ELIF repeated failure → force oracle verification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.core_runtime.task_runtime import RuntimeTask
from backend.core_runtime.confidence_runtime import ConfidenceEstimate
from backend.core_runtime.execution_policy import ExecutionPolicy


# ---------------------------------------------------------------------------
# RoutingDecision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoutingDecision:
    """Deterministic routing decision for a task.

    action:    "cache_hit", "standard", "ood_escalation", "oracle_verification"
    projection_budget:  Max projection iterations.
    force_oracle:       Whether oracle verification is mandatory.
    reason:             Human-readable explanation.
    """
    action: str
    projection_budget: int
    force_oracle: bool
    reason: str

    def __post_init__(self) -> None:
        valid_actions = {"cache_hit", "standard", "ood_escalation", "oracle_verification"}
        if self.action not in valid_actions:
            raise ValueError(f"Invalid routing action: {self.action}, expected one of {valid_actions}")


# ---------------------------------------------------------------------------
# CapabilityRouter
# ---------------------------------------------------------------------------

class CapabilityRouter:
    """Deterministic rule-based capability router.

    Routes tasks based on exact cache, confidence estimate, and failure
    history. All decisions are deterministic and auditable.
    """

    # Failure repetition threshold for oracle verification
    _FAILURE_REPEAT_THRESHOLD = 3

    def __init__(
        self,
        policy: ExecutionPolicy | None = None,
        failure_counts: dict[str, int] | None = None,
    ) -> None:
        self._policy = policy or ExecutionPolicy()
        self._failure_counts: dict[str, int] = failure_counts or {}

    def route(
        self,
        task: RuntimeTask,
        confidence: ConfidenceEstimate,
        cache_hit: bool = False,
    ) -> RoutingDecision:
        """Make a deterministic routing decision.

        Priority:
        1. Exact cache hit → return immediately
        2. Repeated failure topology → force oracle verification
        3. Likely OOD → large projection budget
        4. High confidence → small projection budget
        5. Default → medium projection budget
        """
        # Rule 1: Cache hit
        if cache_hit:
            return RoutingDecision(
                action="cache_hit",
                projection_budget=0,
                force_oracle=False,
                reason="Exact cache hit — no execution needed",
            )

        # Extract topology family from task metadata
        topo = task.metadata.get("topology_family", "unknown")

        # Rule 2: Repeated failure topology
        failure_count = self._failure_counts.get(topo, 0)
        if failure_count >= self._FAILURE_REPEAT_THRESHOLD:
            return RoutingDecision(
                action="oracle_verification",
                projection_budget=self._policy.projection_budget_high,
                force_oracle=True,
                reason=f"Topology '{topo}' has {failure_count} past failures — forcing oracle verification",
            )

        # Rule 3: Likely OOD
        if confidence.likely_ood:
            return RoutingDecision(
                action="ood_escalation",
                projection_budget=self._policy.projection_budget_high,
                force_oracle=True,
                reason=f"OOD detected (confidence={confidence.confidence_score:.3f}) — large budget + oracle",
            )

        # Rule 4: High confidence
        if confidence.confidence_score >= 0.7:
            return RoutingDecision(
                action="standard",
                projection_budget=self._policy.projection_budget_low,
                force_oracle=False,
                reason=f"High confidence ({confidence.confidence_score:.3f}) — small budget",
            )

        # Rule 5: Default (medium confidence)
        return RoutingDecision(
            action="standard",
            projection_budget=self._policy.projection_budget_high,
            force_oracle=False,
            reason=f"Medium confidence ({confidence.confidence_score:.3f}) — standard budget",
        )

    def record_failure(self, topology_family: str) -> None:
        """Record a failure for a topology family (affects future routing)."""
        self._failure_counts[topology_family] = self._failure_counts.get(topology_family, 0) + 1

    def record_success(self, topology_family: str) -> None:
        """Record a success (reduces failure count by 1, min 0)."""
        current = self._failure_counts.get(topology_family, 0)
        if current > 0:
            self._failure_counts[topology_family] = current - 1

    @property
    def failure_counts(self) -> dict[str, int]:
        return dict(self._failure_counts)
