"""CPT Core Runtime — Capability Router (v2.14).

Deterministic rule-based routing. NO machine learning routing yet.

v2.14 adds:
- semantic_retrieval routing (FAISS hit, no exact cache)
- warmstart_projection routing (warmstart available)
- degraded_execution routing
- cost-aware budget allocation

Routing priority:
1. exact_cache_hit     → return cached result immediately
2. semantic_retrieval  → use warmstart from similar circuit
3. warmstart_projection → project with warmstart init
4. standard_projection → project with standard init
5. increased_budget    → more projection iterations
6. oracle_verification → force oracle check
7. degraded_execution  → runtime failure path

ALL routing decisions remain 100% deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.core_runtime.task_runtime import RuntimeTask
from backend.core_runtime.confidence_runtime import ConfidenceEstimate
from backend.core_runtime.execution_policy import ExecutionPolicy
from backend.runtime.cost_estimator import ExecutionCostEstimate


# ---------------------------------------------------------------------------
# RoutingDecision (v2.14)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoutingDecision:
    """Deterministic routing decision for a task.

    v2.14 actions:
      exact_cache_hit, semantic_retrieval, warmstart_projection,
      standard_projection, increased_budget, oracle_verification,
      degraded_execution
    """
    action: str
    projection_budget: int
    force_oracle: bool
    reason: str
    retrieval_similarity: float = 0.0  # Similarity score if retrieval-based
    estimated_cost: ExecutionCostEstimate | None = None

    def __post_init__(self) -> None:
        valid_actions = {
            "exact_cache_hit",
            "semantic_retrieval",
            "warmstart_projection",
            "standard_projection",
            "increased_budget",
            "oracle_verification",
            "degraded_execution",
        }
        if self.action not in valid_actions:
            raise ValueError(
                f"Invalid routing action: {self.action}, expected one of {valid_actions}"
            )


# ---------------------------------------------------------------------------
# CapabilityRouter (v2.14)
# ---------------------------------------------------------------------------

class CapabilityRouter:
    """Deterministic rule-based capability router.

    v2.14 adds: retrieval similarity, warmstart, cost estimation.
    All decisions remain deterministic and auditable.
    """

    _FAILURE_REPEAT_THRESHOLD = 3
    _MIN_SIMILARITY_WARMSTART = 0.5

    def __init__(
        self,
        policy: ExecutionPolicy | None = None,
        failure_counts: dict[str, int] | None = None,
        min_similarity_warmstart: float = 0.5,
    ) -> None:
        self._policy = policy or ExecutionPolicy()
        self._failure_counts: dict[str, int] = failure_counts or {}
        self._min_similarity = min_similarity_warmstart

    def route(
        self,
        task: RuntimeTask,
        confidence: ConfidenceEstimate,
        cache_hit: bool = False,
        retrieval_similarity: float = 0.0,
        cost_estimate: ExecutionCostEstimate | None = None,
        is_degraded: bool = False,
    ) -> RoutingDecision:
        """Make a deterministic routing decision.

        Priority:
        1. Exact cache hit → return immediately
        2. Degraded execution → failure path
        3. Repeated failure topology → force oracle verification
        4. Likely OOD → large projection budget
        5. High similarity retrieval → warmstart projection
        6. Moderate similarity → semantic retrieval path
        7. High confidence → small projection budget
        8. Default → standard projection budget
        """
        topo = task.metadata.get("topology_family", "unknown")

        # Rule 1: Exact cache hit — always first
        if cache_hit:
            return RoutingDecision(
                action="exact_cache_hit",
                projection_budget=0,
                force_oracle=False,
                reason="Exact cache hit — no execution needed",
                estimated_cost=cost_estimate,
            )

        # Rule 2: Degraded execution
        if is_degraded:
            return RoutingDecision(
                action="degraded_execution",
                projection_budget=self._policy.projection_budget_high,
                force_oracle=True,
                reason="Degraded execution detected — forcing oracle verification",
                estimated_cost=cost_estimate,
            )

        # Rule 3: Repeated failure topology → force oracle
        failure_count = self._failure_counts.get(topo, 0)
        if failure_count >= self._FAILURE_REPEAT_THRESHOLD:
            return RoutingDecision(
                action="oracle_verification",
                projection_budget=self._policy.projection_budget_high,
                force_oracle=True,
                reason=f"Topology '{topo}' has {failure_count} past failures — forcing oracle",
                estimated_cost=cost_estimate,
            )

        # Rule 4: Likely OOD → large budget + oracle
        if confidence.likely_ood:
            # Still try warmstart if similarity is very high
            if retrieval_similarity >= self._min_similarity:
                return RoutingDecision(
                    action="warmstart_projection",
                    projection_budget=self._policy.projection_budget_high,
                    force_oracle=True,
                    reason=f"OOD + warmstart available (sim={retrieval_similarity:.3f}) — high budget + oracle",
                    retrieval_similarity=retrieval_similarity,
                    estimated_cost=cost_estimate,
                )
            return RoutingDecision(
                action="increased_budget",
                projection_budget=self._policy.projection_budget_high,
                force_oracle=True,
                reason=f"OOD detected (confidence={confidence.confidence_score:.3f}) — large budget + oracle",
                estimated_cost=cost_estimate,
            )

        # Rule 5: High similarity retrieval → warmstart projection
        if retrieval_similarity >= self._min_similarity:
            budget = self._policy.projection_budget_low
            if cost_estimate and cost_estimate.estimated_projection_iterations > budget:
                budget = min(cost_estimate.estimated_projection_iterations, self._policy.projection_budget_high)
            return RoutingDecision(
                action="warmstart_projection",
                projection_budget=budget,
                force_oracle=False,
                reason=f"Warmstart available (sim={retrieval_similarity:.3f}) — reduced budget",
                retrieval_similarity=retrieval_similarity,
                estimated_cost=cost_estimate,
            )

        # Rule 6: Moderate similarity → semantic retrieval path
        if retrieval_similarity >= 0.3:
            return RoutingDecision(
                action="semantic_retrieval",
                projection_budget=self._policy.projection_budget_high,
                force_oracle=False,
                reason=f"Partial retrieval match (sim={retrieval_similarity:.3f}) — standard budget",
                retrieval_similarity=retrieval_similarity,
                estimated_cost=cost_estimate,
            )

        # Rule 7: High confidence → small budget
        if confidence.confidence_score >= 0.7:
            return RoutingDecision(
                action="standard_projection",
                projection_budget=self._policy.projection_budget_low,
                force_oracle=False,
                reason=f"High confidence ({confidence.confidence_score:.3f}) — small budget",
                estimated_cost=cost_estimate,
            )

        # Rule 8: Default — standard projection
        return RoutingDecision(
            action="standard_projection",
            projection_budget=self._policy.projection_budget_high,
            force_oracle=False,
            reason=f"Standard projection (confidence={confidence.confidence_score:.3f})",
            estimated_cost=cost_estimate,
        )

    def record_failure(self, topology_family: str) -> None:
        """Record a failure for a topology family."""
        self._failure_counts[topology_family] = self._failure_counts.get(topology_family, 0) + 1

    def record_success(self, topology_family: str) -> None:
        """Record a success (reduces failure count by 1, min 0)."""
        current = self._failure_counts.get(topology_family, 0)
        if current > 0:
            self._failure_counts[topology_family] = current - 1

    @property
    def failure_counts(self) -> dict[str, int]:
        return dict(self._failure_counts)
