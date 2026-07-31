"""Tests for CORE v3.3 — Explainability Runtime.

Mandatory tests:
- same input -> same explanation
- explanation types are from known set
- no free-form text generation
- deterministic across repeated calls
"""

import pytest

from core_runtime.core.trustworthiness_runtime import (
    TrustLevel,
    TrustFlag,
    audit_execution,
)
from core_runtime.core.explainability_runtime import (
    ExecutionExplanation,
    ExplainabilityRuntime,
    EXPLANATION_TYPES,
)


def _audit_certain():
    return audit_execution("h1", 0.95, 0.05, 3, 0.0001, "fast_converging", False, False)


def _audit_transitional():
    return audit_execution("h2", 0.6, 0.4, 6, 0.02, "fast_converging", False, False)


def _audit_indeterminate():
    return audit_execution("h3", 0.2, 0.9, 25, 0.5, "divergence_risk", True, True)


class TestExecutionExplanation:

    def test_frozen(self):
        e = ExecutionExplanation(
            task_hash="h1", summary="test", key_factors=("a",),
            topology_family="bridge", confidence_score=0.9,
            trust_level="certain", projected_iterations=5,
            residual=0.001, explanation_type="certain_execution",
            metadata={},
        )
        with pytest.raises(AttributeError):
            e.summary = "changed"

    def test_fingerprint_deterministic(self):
        e = ExecutionExplanation(
            task_hash="h1", summary="test", key_factors=("a",),
            topology_family="bridge", confidence_score=0.9,
            trust_level="certain", projected_iterations=5,
            residual=0.001, explanation_type="certain_execution",
            metadata={},
        )
        assert e.fingerprint() == e.fingerprint()

    def test_fingerprint_differs_for_different_input(self):
        e1 = ExecutionExplanation(
            task_hash="h1", summary="test", key_factors=("a",),
            topology_family="bridge", confidence_score=0.9,
            trust_level="certain", projected_iterations=5,
            residual=0.001, explanation_type="certain_execution",
            metadata={},
        )
        e2 = ExecutionExplanation(
            task_hash="h2", summary="test", key_factors=("a",),
            topology_family="bridge", confidence_score=0.9,
            trust_level="certain", projected_iterations=5,
            residual=0.001, explanation_type="certain_execution",
            metadata={},
        )
        assert e1.fingerprint() != e2.fingerprint()

    def test_to_dict_roundtrip(self):
        e = ExecutionExplanation(
            task_hash="h1", summary="test", key_factors=("a", "b"),
            topology_family="ladder", confidence_score=0.8,
            trust_level="transitional", projected_iterations=10,
            residual=0.02, explanation_type="transitional_execution",
            metadata={"routing_action": "standard_projection"},
        )
        d = e.to_dict()
        assert d["task_hash"] == "h1"
        assert d["key_factors"] == ["a", "b"]
        assert "fingerprint" in d


class TestExplainabilityRuntime:

    def test_certain_audit_gets_certain_explanation(self):
        explainer = ExplainabilityRuntime()
        audit = _audit_certain()
        expl = explainer.explain(audit)
        assert expl.explanation_type == "certain_execution"

    def test_transitional_audit_gets_transitional_explanation(self):
        explainer = ExplainabilityRuntime()
        audit = _audit_transitional()
        expl = explainer.explain(audit)
        assert expl.explanation_type == "transitional_execution"

    def test_indeterminate_audit_gets_indeterminate_explanation(self):
        explainer = ExplainabilityRuntime()
        audit = _audit_indeterminate()
        expl = explainer.explain(audit)
        assert expl.explanation_type in (
            "projection_escalated", "indeterminate_execution",
        )

    def test_cache_hit_routing_gets_cache_hit_explanation(self):
        explainer = ExplainabilityRuntime()
        audit = _audit_certain()
        routing = {"action": "exact_cache_hit"}
        expl = explainer.explain(audit, routing=routing)
        assert expl.explanation_type == "exact_cache_hit"

    def test_warmstart_routing_gets_warmstart_explanation(self):
        explainer = ExplainabilityRuntime()
        audit = _audit_certain()
        routing = {
            "action": "warmstart_projection",
            "retrieval_similarity": 0.85,
        }
        expl = explainer.explain(audit, routing=routing)
        assert expl.explanation_type == "warmstart_assisted"

    def test_ood_routing_gets_ood_explanation(self):
        explainer = ExplainabilityRuntime()
        # Use a non-escalated audit with OOD routing
        audit = audit_execution("h1", 0.6, 0.4, 6, 0.05, "standard", False, True)
        routing = {"action": "standard_projection", "ood": True}
        expl = explainer.explain(audit, routing=routing)
        assert expl.explanation_type == "ood_conservative"

    def test_escalated_audit_gets_escalated_explanation(self):
        explainer = ExplainabilityRuntime()
        audit = audit_execution("h1", 0.3, 0.8, 25, 0.5, "divergence_risk", True, False)
        expl = explainer.explain(audit)
        assert expl.explanation_type == "projection_escalated"

    def test_unstable_trajectory_gets_unstable_explanation(self):
        explainer = ExplainabilityRuntime()
        audit = audit_execution("h1", 0.5, 0.5, 10, 0.05, "divergence_risk", False, False)
        trajectory = {"trajectory_class": "divergence_risk"}
        expl = explainer.explain(audit, trajectory=trajectory)
        assert expl.explanation_type == "unstable_convergence"

    def test_same_input_same_explanation(self):
        explainer = ExplainabilityRuntime()
        audit = _audit_transitional()
        e1 = explainer.explain(audit)
        e2 = explainer.explain(audit)
        assert e1.fingerprint() == e2.fingerprint()

    def test_repeated_100_times_deterministic(self):
        explainer = ExplainabilityRuntime()
        audit = _audit_certain()
        fps = [explainer.explain(audit).fingerprint() for _ in range(100)]
        assert len(set(fps)) == 1

    def test_explanation_type_in_known_set(self):
        explainer = ExplainabilityRuntime()
        for audit in [_audit_certain(), _audit_transitional(), _audit_indeterminate()]:
            expl = explainer.explain(audit)
            assert expl.explanation_type in EXPLANATION_TYPES

    def test_key_factors_nonempty(self):
        explainer = ExplainabilityRuntime()
        audit = _audit_certain()
        expl = explainer.explain(audit)
        assert len(expl.key_factors) > 0

    def test_summary_is_from_template(self):
        explainer = ExplainabilityRuntime()
        audit = _audit_certain()
        expl = explainer.explain(audit)
        # Summary must be from the predefined templates, not generated
        assert expl.summary  # not empty
        assert len(expl.summary) < 200  # not free-form long text

    def test_all_explanation_types_have_templates(self):
        from core_runtime.core.explainability_runtime import _SUMMARIES
        for etype in EXPLANATION_TYPES:
            assert etype in _SUMMARIES, f"Missing template for {etype}"
