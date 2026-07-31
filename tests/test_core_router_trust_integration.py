"""Tests for CORE v3.3 — Router Integration with Trust.

Verifies:
- trust-aware routing produces deterministic decisions
- TRANSITIONAL trust increases budget
- INDETERMINATE trust triggers safe routing
- CERTAIN trust does not alter baseline routing
- audit layer does not change routing for existing code paths
- exact cache hit still wins even with INDETERMINATE audit
"""

import pytest

from core_runtime.core.trustworthiness_runtime import (
    TrustLevel,
    TrustFlag,
    audit_execution,
)
from core_runtime.core.routing.capability_router import (
    CapabilityRouter,
    ConfidenceEstimate,
)
from backend.core_runtime.task_runtime import RuntimeTask


def _make_task(task_hash: str = "h1", confidence: float = 0.9):
    """Create a RuntimeTask for router testing."""
    return RuntimeTask(
        task_id=task_hash,
        domain="linear_system",
        input_artifact=task_hash,
        oracle_name="linear_system_oracle",
        surrogate_name="linear_system_surrogate",
        metadata={
            "task_hash": task_hash,
            "topology_family": "bridge",
            "confidence": confidence,
        },
    )


def _make_confidence(score: float = 0.9) -> ConfidenceEstimate:
    """Create a ConfidenceEstimate."""
    return ConfidenceEstimate(
        confidence_score=score,
        estimated_projection_iterations=10,
        likely_ood=False,
    )


class TestRouterTrustIntegration:

    def test_certain_audit_no_behavior_change(self):
        """CERTAIN trust does not change baseline routing."""
        router = CapabilityRouter()
        task = _make_task(confidence=0.95)
        conf = _make_confidence(0.95)
        audit = audit_execution("h1", 0.95, 0.05, 3, 0.0001, "fast_converging", False, False)
        assert audit.trust_level == TrustLevel.CERTAIN
        decision = router.route(task, conf, trust_audit=audit)
        # CERTAIN should not trigger any special trust routing
        assert decision.action in (
            "exact_cache_hit", "standard_projection", "warmstart_projection",
        )

    def test_transitional_audit_gets_conservative_budget(self):
        """TRANSITIONAL trust gets conservative budget (standard_projection)."""
        router = CapabilityRouter()
        task = _make_task(confidence=0.6)
        conf = _make_confidence(0.6)
        audit = audit_execution("h1", 0.6, 0.4, 6, 0.02, "fast_converging", False, False)
        assert audit.trust_level == TrustLevel.TRANSITIONAL
        decision = router.route(task, conf, trust_audit=audit)
        # TRANSITIONAL uses standard_projection with conservative budget
        assert decision.action == "standard_projection"
        assert decision.projection_budget < 20  # conservative

    def test_indeterminate_audit_gets_safe_routing(self):
        """INDETERMINATE trust gets oracle verification or degraded."""
        router = CapabilityRouter()
        task = _make_task(confidence=0.2)
        conf = _make_confidence(0.2)
        audit = audit_execution("h1", 0.2, 0.9, 25, 0.5, "divergence_risk", True, True)
        assert audit.trust_level == TrustLevel.INDETERMINATE
        decision = router.route(task, conf, trust_audit=audit)
        assert decision.action in (
            "oracle_verification", "degraded_execution", "increased_budget",
        )

    def test_no_audit_no_crash(self):
        """Router must work without trust audit (backward compat)."""
        router = CapabilityRouter()
        task = _make_task(confidence=0.9)
        conf = _make_confidence(0.9)
        decision = router.route(task, conf)
        assert decision.action is not None

    def test_router_deterministic_with_audit(self):
        """Same input + same audit -> same routing decision."""
        router = CapabilityRouter()
        task = _make_task(confidence=0.6)
        conf = _make_confidence(0.6)
        audit = audit_execution("h1", 0.6, 0.4, 6, 0.02, "fast_converging", False, False)
        d1 = router.route(task, conf, trust_audit=audit)
        d2 = router.route(task, conf, trust_audit=audit)
        assert d1.action == d2.action
        assert d1.projection_budget == d2.projection_budget

    def test_router_deterministic_100_times(self):
        """Routing decision is stable across 100 calls."""
        router = CapabilityRouter()
        task = _make_task(confidence=0.5)
        conf = _make_confidence(0.5)
        audit = audit_execution("h1", 0.5, 0.5, 12, 0.02, "oscillating", False, False)
        decisions = [
            router.route(task, conf, trust_audit=audit)
            for _ in range(100)
        ]
        actions = set(d.action for d in decisions)
        budgets = set(d.projection_budget for d in decisions)
        assert len(actions) == 1
        assert len(budgets) == 1

    def test_trust_audit_does_not_override_exact_cache(self):
        """Even with INDETERMINATE audit, exact cache hit still wins."""
        router = CapabilityRouter()
        task = _make_task(confidence=0.2)
        conf = _make_confidence(0.2)
        audit = audit_execution("h1", 0.2, 0.9, 25, 0.5, "divergence_risk", True, True)
        decision = router.route(task, conf, trust_audit=audit, cache_hit=True)
        assert decision.action == "exact_cache_hit"

    def test_certain_audit_zero_flags(self):
        """CERTAIN audit has zero flags."""
        audit = audit_execution("h1", 0.95, 0.05, 3, 0.0001, "fast_converging", False, False)
        assert audit.trust_level == TrustLevel.CERTAIN
        assert len(audit.flags) == 0
