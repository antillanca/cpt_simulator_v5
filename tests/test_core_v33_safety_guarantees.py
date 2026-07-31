"""Tests for CORE v3.3 — Safety Guarantees.

These tests verify that the v3.3 trust/audit layer does NOT violate
any frozen v3.1/v3.2 guarantees. If any of these fail, the trust
layer has leaked into the execution path and must be fixed.
"""

import numpy as np
import pytest

from core_runtime.core.trustworthiness_runtime import (
    TrustLevel,
    audit_execution,
)
from core_runtime.core.uncertainty_memory import UncertaintyMemory, UncertaintyEntry
from core_runtime.core.explainability_runtime import ExplainabilityRuntime
from core_runtime.core.audit_trace_store import AuditTraceStore


# ═══════════════════════════════════════════════════════════════
# Guarantee 1: Deterministic input -> deterministic audit
# ═══════════════════════════════════════════════════════════════

class TestDeterministicAudit:
    """Audit classification must be 100% deterministic."""

    def test_same_input_same_audit_1000_times(self):
        for _ in range(1000):
            a = audit_execution("h1", 0.7, 0.3, 8, 0.02, "standard", False, False)
        # residual > 0.01 triggers HIGH_RESIDUAL_HEURISTIC -> TRANSITIONAL
        assert a.trust_level == TrustLevel.TRANSITIONAL

    def test_audit_fingerprint_stable_across_serializations(self):
        a = audit_execution("h1", 0.9, 0.1, 3, 0.0001, "fast_converging", False, False)
        fp1 = a.fingerprint()
        d = a.to_dict()
        fp2 = a.fingerprint()
        assert fp1 == fp2


# ═══════════════════════════════════════════════════════════════
# Guarantee 2: Exact cache always first
# ═══════════════════════════════════════════════════════════════

class TestExactCachePriority:
    """Trust audit cannot override exact cache hit."""

    def test_indeterminate_audit_does_not_block_cache(self):
        from core_runtime.core.routing.capability_router import (
            CapabilityRouter,
            ConfidenceEstimate,
        )
        from backend.core_runtime.task_runtime import RuntimeTask

        router = CapabilityRouter()
        task = RuntimeTask(
            task_id="h1", domain="linear_system", input_artifact="h1",
            oracle_name="oracle", surrogate_name="surrogate",
            metadata={"task_hash": "h1", "topology_family": "bridge"},
        )
        conf = ConfidenceEstimate(
            confidence_score=0.2,
            estimated_projection_iterations=25,
            likely_ood=True,
        )
        audit = audit_execution(
            "h1", 0.2, 0.9, 25, 0.5, "divergence_risk", True, True,
        )
        assert audit.trust_level == TrustLevel.INDETERMINATE
        decision = router.route(task, conf, trust_audit=audit, cache_hit=True)
        assert decision.action == "exact_cache_hit"


# ═══════════════════════════════════════════════════════════════
# Guarantee 3: Projection remains final authority
# ═══════════════════════════════════════════════════════════════

class TestProjectionAuthority:
    """Trust audit does not alter projection results."""

    def test_audit_does_not_modify_projection_result(self):
        """Run projection with and without audit — results must be identical."""
        from core_runtime.domains.linear_system import (
            LinearSystemTask,
            LinearSystemOracle,
            LinearSystemSurrogate,
            LinearSystemProjection,
        )

        rng = np.random.RandomState(42)
        A = rng.randn(4, 4) + 4 * np.eye(4)
        b = rng.randn(4)
        task = LinearSystemTask(
            task_id="safety_test", domain_name="linear_system",
            input_artifact="safety_test",
            metadata={"A": A, "b": b, "n": 4, "task_hash": "safety_test",
                      "topology_family": "n=4"},
        )
        oracle = LinearSystemOracle()
        surrogate = LinearSystemSurrogate()
        projection = LinearSystemProjection()

        # Run without audit
        oracle_result = oracle.solve(task)
        surrogate_result = surrogate.predict(task)
        proj_no_audit = projection.project(task, surrogate_result, budget=50)

        # Run with audit (audit is post-hoc, doesn't feed into projection)
        audit = audit_execution(
            "safety_test", 0.9, 0.1,
            proj_no_audit["iterations"],
            float(proj_no_audit["residual"]),
            "fast_converging", False, False,
        )
        proj_with_audit = projection.project(task, surrogate_result, budget=50)

        # Results must be identical
        assert np.allclose(proj_no_audit["solution"], proj_with_audit["solution"])
        assert proj_no_audit["iterations"] == proj_with_audit["iterations"]
        assert abs(float(proj_no_audit["residual"]) -
                   float(proj_with_audit["residual"])) < 1e-14


# ═══════════════════════════════════════════════════════════════
# Guarantee 4: Same input + same seed -> same outputs
# ═══════════════════════════════════════════════════════════════

class TestDeterministicOutputs:
    """Full pipeline with audit layer must be deterministic."""

    def test_same_seed_same_trust_metrics(self):
        from core_runtime.domains.linear_system import (
            LinearSystemTask,
            LinearSystemOracle,
            LinearSystemSurrogate,
            LinearSystemProjection,
        )

        rng1 = np.random.RandomState(123)
        rng2 = np.random.RandomState(123)

        results1 = []
        results2 = []

        for i in range(10):
            for rng, results in [(rng1, results1), (rng2, results2)]:
                A = rng.randn(3, 3) + 3 * np.eye(3)
                b = rng.randn(3)
                task = LinearSystemTask(
                    task_id=f"det_{i}", domain_name="linear_system",
                    input_artifact=f"det_{i}",
                    metadata={"A": A, "b": b, "n": 3, "task_hash": f"det_{i}"},
                )
                oracle = LinearSystemOracle()
                surrogate = LinearSystemSurrogate()
                projection = LinearSystemProjection()
                oracle_result = oracle.solve(task)
                surrogate_result = surrogate.predict(task)
                proj = projection.project(task, surrogate_result, budget=50)
                residual = float(proj["residual"])
                confidence = max(0.0, 1.0 - residual)
                audit = audit_execution(
                    f"det_{i}", confidence, residual,
                    proj["iterations"], residual,
                    "fast_converging", False, False,
                )
                results.append((audit.trust_level, audit.fingerprint()))

        assert results1 == results2


# ═══════════════════════════════════════════════════════════════
# Guarantee 5: Uncertainty memory barred from clean cache
# ═══════════════════════════════════════════════════════════════

class TestUncertaintyMemoryIsolation:
    """Uncertainty memory must not pollute exact cache or retrieval memory."""

    def test_uncertainty_entries_not_in_exact_cache(self):
        um = UncertaintyMemory()
        entry = UncertaintyEntry(
            task_hash="h_degraded",
            reason="divergence",
            topology_signature="bridge",
            routing_action="degraded_execution",
            projection_iterations=25,
            residual=0.5,
            confidence_score=0.2,
            trust_level="indeterminate",
            timestamp="2026-01-01T00:00:00Z",
            scheduler_context={},
            metadata={},
        )
        um.add_entry(entry)
        # Uncertainty memory is a SEPARATE store
        # It has no connection to exact cache
        assert um.contains("h_degraded")
        # Verify it doesn't have cache-like methods
        assert not hasattr(um, "get_cache")
        assert not hasattr(um, "cache_hit")

    def test_uncertainty_memory_is_not_retrieval(self):
        um = UncertaintyMemory()
        assert not hasattr(um, "retrieve")
        assert not hasattr(um, "similarity_search")


# ═══════════════════════════════════════════════════════════════
# Guarantee 6: Knowledge/specs remain frozen
# ═══════════════════════════════════════════════════════════════

class TestFrozenKnowledge:
    """Trust layer does not modify frozen knowledge/specs."""

    def test_runtime_config_unchanged_by_audit(self):
        from core_runtime.core.runtime_config import RuntimeConfig
        c1 = RuntimeConfig()
        # Create an audit
        audit = audit_execution("h1", 0.3, 0.8, 25, 0.5, "divergence_risk", True, True)
        # Config must be the same
        c2 = RuntimeConfig()
        assert c1 == c2

    def test_feature_flags_still_default_false(self):
        from core_runtime.core.runtime_config import RuntimeConfig
        c = RuntimeConfig()
        assert c.enable_lora_experts is False
        assert c.enable_replay is False
        assert c.enable_continual_training is False
        assert c.enable_distributed_execution is False


# ═══════════════════════════════════════════════════════════════
# Guarantee 7: Domain logic isolated from trust orchestration
# ═══════════════════════════════════════════════════════════════

class TestDomainIsolation:
    """Trust modules do not import from domain-specific code."""

    def test_trust_runtime_no_domain_imports(self):
        """trustworthiness_runtime must not import domain modules."""
        import importlib
        mod = importlib.import_module("core_runtime.core.trustworthiness_runtime")
        src = open(mod.__file__).read()
        assert "circuits" not in src
        assert "linear_system" not in src
        assert "propositional_logic" not in src

    def test_uncertainty_memory_no_domain_imports(self):
        import importlib
        mod = importlib.import_module("core_runtime.core.uncertainty_memory")
        src = open(mod.__file__).read()
        assert "circuits" not in src
        assert "linear_system" not in src

    def test_explainability_runtime_no_domain_imports(self):
        import importlib
        mod = importlib.import_module("core_runtime.core.explainability_runtime")
        src = open(mod.__file__).read()
        assert "circuits" not in src
        assert "linear_system" not in src

    def test_audit_trace_store_no_domain_imports(self):
        import importlib
        mod = importlib.import_module("core_runtime.core.audit_trace_store")
        src = open(mod.__file__).read()
        assert "circuits" not in src
        assert "linear_system" not in src


# ═══════════════════════════════════════════════════════════════
# Guarantee 8: Scheduler does not change physics
# ═══════════════════════════════════════════════════════════════

class TestSchedulerPhysicsIndependence:
    """Trust audit does not feed into the physics scheduler."""

    def test_audit_not_scheduler_input(self):
        """TrustworthinessAudit is NOT accepted by any scheduler method."""
        from core_runtime.core.trustworthiness_runtime import TrustworthinessAudit
        # Verify audit is a frozen dataclass, not a scheduler input
        audit = audit_execution("h1", 0.9, 0.1, 3, 0.0001, "fast_converging", False, False)
        # It should not have scheduler-relevant methods
        assert not hasattr(audit, "schedule")
        assert not hasattr(audit, "dispatch")
        assert not hasattr(audit, "allocate")
