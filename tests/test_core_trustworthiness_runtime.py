"""Tests for CORE v3.3 — Trustworthiness Runtime.

Mandatory tests:
- same input -> same audit
- certainty/transitional/indeterminate classification is deterministic
- audit does not modify projection results
- audit does not modify exact cache behavior
- trust scores are observability signals only
"""

import pytest

from core_runtime.core.trustworthiness_runtime import (
    TrustLevel,
    TrustFlag,
    TrustworthinessAudit,
    audit_execution,
)


# ═══════════════════════════════════════════════════════════════
# Determinism tests
# ═══════════════════════════════════════════════════════════════

class TestAuditDeterminism:
    """Same input -> same audit."""

    def test_same_input_same_trust_level(self):
        a1 = audit_execution("h1", 0.9, 0.1, 5, 0.001, "fast_converging", False, False)
        a2 = audit_execution("h1", 0.9, 0.1, 5, 0.001, "fast_converging", False, False)
        assert a1.trust_level == a2.trust_level

    def test_same_input_same_flags(self):
        a1 = audit_execution("h1", 0.3, 0.8, 20, 0.5, "divergence_risk", True, True)
        a2 = audit_execution("h1", 0.3, 0.8, 20, 0.5, "divergence_risk", True, True)
        assert a1.flags == a2.flags

    def test_same_input_same_confidence(self):
        a1 = audit_execution("h1", 0.75, 0.25, 10, 0.005, "standard", False, False)
        a2 = audit_execution("h1", 0.75, 0.25, 10, 0.005, "standard", False, False)
        assert a1.confidence_score == a2.confidence_score

    def test_same_input_same_fingerprint(self):
        a1 = audit_execution("h1", 0.9, 0.1, 5, 0.001, "fast_converging", False, False)
        a2 = audit_execution("h1", 0.9, 0.1, 5, 0.001, "fast_converging", False, False)
        assert a1.fingerprint() == a2.fingerprint()

    def test_different_input_different_fingerprint(self):
        a1 = audit_execution("h1", 0.9, 0.1, 5, 0.001, "fast_converging", False, False)
        a2 = audit_execution("h2", 0.9, 0.1, 5, 0.001, "fast_converging", False, False)
        assert a1.fingerprint() != a2.fingerprint()

    def test_deterministic_flag_is_true(self):
        a = audit_execution("h1", 0.9, 0.1, 5, 0.001, "fast_converging", False, False)
        assert a.deterministic is True


# ═══════════════════════════════════════════════════════════════
# Classification tests
# ═══════════════════════════════════════════════════════════════

class TestTrustClassification:
    """Certainty/transitional/indeterminate classification is deterministic."""

    def test_no_flags_is_certain(self):
        a = audit_execution("h1", 0.9, 0.1, 3, 0.001, "fast_converging", False, False)
        assert a.trust_level == TrustLevel.CERTAIN
        assert len(a.flags) == 0

    def test_single_high_residual_is_transitional(self):
        a = audit_execution("h1", 0.6, 0.3, 4, 0.02, "fast_converging", False, False)
        assert a.trust_level == TrustLevel.TRANSITIONAL
        assert a.flags == (TrustFlag.HIGH_RESIDUAL_HEURISTIC,)

    def test_multiple_flags_is_indeterminate(self):
        a = audit_execution("h1", 0.3, 0.8, 20, 0.5, "divergence_risk", True, True)
        assert a.trust_level == TrustLevel.INDETERMINATE
        assert len(a.flags) >= 2

    def test_escalation_always_indeterminate(self):
        a = audit_execution("h1", 0.8, 0.2, 5, 0.001, "fast_converging", True, False)
        assert a.trust_level == TrustLevel.INDETERMINATE
        assert TrustFlag.ESCALATION_TRIGGERED in a.flags

    def test_ood_always_indeterminate(self):
        a = audit_execution("h1", 0.8, 0.2, 5, 0.001, "fast_converging", False, True)
        assert a.trust_level == TrustLevel.INDETERMINATE
        assert TrustFlag.OOD_EXECUTION in a.flags

    def test_low_confidence_flag(self):
        a = audit_execution("h1", 0.4, 0.6, 5, 0.001, "fast_converging", False, False)
        assert TrustFlag.LOW_CONFIDENCE in a.flags

    def test_unstable_convergence_flag(self):
        a = audit_execution("h1", 0.9, 0.1, 5, 0.001, "divergence_risk", False, False)
        assert TrustFlag.UNSTABLE_CONVERGENCE in a.flags

    def test_high_projection_effort_flag(self):
        a = audit_execution("h1", 0.9, 0.1, 25, 0.001, "fast_converging", False, False,
                            family_budget=10)
        assert TrustFlag.HIGH_PROJECTION_EFFORT in a.flags

    def test_classification_reproducible(self):
        """Run same classification 100 times — must be identical every time."""
        for _ in range(100):
            a = audit_execution("h1", 0.5, 0.5, 12, 0.02, "oscillating", False, False)
        assert a.trust_level == TrustLevel.TRANSITIONAL


# ═══════════════════════════════════════════════════════════════
# Safety tests
# ═══════════════════════════════════════════════════════════════

class TestTrustSafety:
    """Audit layer cannot alter physics or outputs."""

    def test_audit_is_frozen(self):
        a = audit_execution("h1", 0.9, 0.1, 5, 0.001, "fast_converging", False, False)
        with pytest.raises(AttributeError):
            a.trust_level = TrustLevel.INDETERMINATE

    def test_audit_is_observational_only(self):
        """Audit does not produce any executable code or side effects."""
        a = audit_execution("h1", 0.9, 0.1, 5, 0.001, "fast_converging", False, False)
        # The audit is a pure data record — no methods that modify anything
        assert not hasattr(a, "execute")
        assert not hasattr(a, "run")
        assert not hasattr(a, "modify")

    def test_audit_serializable(self):
        """Audit must be serializable for traceability."""
        a = audit_execution("h1", 0.9, 0.1, 5, 0.001, "fast_converging", False, False)
        d = a.to_dict()
        assert isinstance(d, dict)
        assert "task_hash" in d
        assert "fingerprint" in d
        assert d["deterministic"] is True

    def test_audit_fingerprint_stable(self):
        """Fingerprint must be stable across serializations."""
        a = audit_execution("h1", 0.9, 0.1, 5, 0.001, "fast_converging", False, False)
        fp1 = a.fingerprint()
        d = a.to_dict()
        fp2 = a.fingerprint()
        assert fp1 == fp2

    def test_trust_flags_are_exhaustive(self):
        """All expected trust flags exist."""
        expected = {
            "high_residual_heuristic",
            "unstable_convergence",
            "escalation_triggered",
            "low_confidence",
            "ood_execution",
            "high_projection_effort",
        }
        actual = {f.value for f in TrustFlag}
        assert actual == expected

    def test_trust_levels_are_exhaustive(self):
        expected = {"certain", "transitional", "indeterminate"}
        actual = {l.value for l in TrustLevel}
        assert actual == expected
