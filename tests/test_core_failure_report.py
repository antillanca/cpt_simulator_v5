"""Tests for CORE v3.3 — Structured Failure Reports.

Mandatory tests:
- same input -> same failure report
- report is structured, not free text
- deterministic cause inference
- deterministic action inference
"""

import pytest

from core_runtime.core.failure_report import (
    FailureReport,
    generate_failure_report,
    PROBABLE_CAUSES,
    RECOMMENDED_ACTIONS,
)


def _make_task(task_hash: str = "h1"):
    return {"task_hash": task_hash}


def _make_state(**overrides):
    base = {
        "residual": 0.5,
        "confidence_score": 0.3,
        "projection_iterations": 10,
        "topology_signature": "bridge",
        "ood": False,
        "escalation_required": False,
        "trajectory_class": "standard",
    }
    base.update(overrides)
    return base


class TestFailureReport:

    def test_report_is_frozen(self):
        r = generate_failure_report(_make_task(), "timeout", _make_state())
        with pytest.raises(AttributeError):
            r.error_type = "other"

    def test_same_input_same_report(self):
        r1 = generate_failure_report(_make_task(), "timeout", _make_state())
        r2 = generate_failure_report(_make_task(), "timeout", _make_state())
        assert r1.fingerprint() == r2.fingerprint()

    def test_different_input_different_report(self):
        r1 = generate_failure_report(_make_task("h1"), "timeout", _make_state())
        r2 = generate_failure_report(_make_task("h2"), "timeout", _make_state())
        assert r1.fingerprint() != r2.fingerprint()

    def test_report_is_structured_not_free_text(self):
        r = generate_failure_report(_make_task(), "timeout", _make_state())
        # All fields are typed, not free text
        assert isinstance(r.task_hash, str)
        assert isinstance(r.error_type, str)
        assert isinstance(r.probable_cause, str)
        assert isinstance(r.recommended_action, str)
        assert isinstance(r.conditions, dict)
        assert r.deterministic is True

    def test_oracle_timeout_cause(self):
        r = generate_failure_report(
            _make_task(), "oracle_timeout", _make_state(),
        )
        assert r.probable_cause == "oracle_timeout"

    def test_surrogate_instability_cause(self):
        r = generate_failure_report(
            _make_task(), "surrogate_instability", _make_state(),
        )
        assert r.probable_cause == "surrogate_instability"

    def test_ood_cause(self):
        r = generate_failure_report(
            _make_task(), "projection_failure", _make_state(ood=True),
        )
        assert r.probable_cause == "ood_topology_gap"

    def test_projection_stagnation_cause(self):
        r = generate_failure_report(
            _make_task(), "projection_failure",
            _make_state(residual=2.0, projection_iterations=20),
        )
        assert r.probable_cause == "projection_stagnation"

    def test_extreme_resistance_cause(self):
        r = generate_failure_report(
            _make_task(), "projection_failure",
            _make_state(residual=0.7, projection_iterations=5),
        )
        assert r.probable_cause == "extreme_resistance_instability"

    def test_source_saturation_cause(self):
        r = generate_failure_report(
            _make_task(), "projection_failure",
            _make_state(residual=0.2, confidence_score=0.2, projection_iterations=5),
        )
        assert r.probable_cause == "source_saturation"

    def test_oracle_timeout_action(self):
        r = generate_failure_report(
            _make_task(), "oracle_timeout", _make_state(),
        )
        assert r.recommended_action == "force_oracle_verification"

    def test_surrogate_instability_action(self):
        r = generate_failure_report(
            _make_task(), "surrogate_instability", _make_state(),
        )
        assert r.recommended_action == "fallback_to_safe_path"

    def test_ood_action(self):
        r = generate_failure_report(
            _make_task(), "projection_failure", _make_state(ood=True),
        )
        assert r.recommended_action == "route_to_uncertainty_memory"

    def test_stagnation_action(self):
        r = generate_failure_report(
            _make_task(), "projection_failure",
            _make_state(residual=2.0, projection_iterations=20),
        )
        assert r.recommended_action == "increase_projection_budget"

    def test_known_causes_are_exhaustive(self):
        assert "projection_stagnation" in PROBABLE_CAUSES
        assert "ood_topology_gap" in PROBABLE_CAUSES
        assert "unknown" in PROBABLE_CAUSES

    def test_known_actions_are_exhaustive(self):
        assert "increase_projection_budget" in RECOMMENDED_ACTIONS
        assert "fallback_to_safe_path" in RECOMMENDED_ACTIONS

    def test_conditions_included(self):
        r = generate_failure_report(_make_task(), "timeout", _make_state())
        assert "residual" in r.conditions
        assert "confidence_score" in r.conditions
        assert "projection_iterations" in r.conditions

    def test_serializable(self):
        r = generate_failure_report(_make_task(), "timeout", _make_state())
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "fingerprint" in d
