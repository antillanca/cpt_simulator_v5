"""Tests for CORE v3.3 — Audit Trace Store.

Mandatory tests:
- deterministic ordering
- atomic persistence
- same input -> same trace
- no influence on runtime outputs
- export/import roundtrip
"""

import json
import tempfile
from pathlib import Path

import pytest

from core_runtime.core.trustworthiness_runtime import (
    TrustLevel,
    TrustFlag,
    audit_execution,
)
from core_runtime.core.explainability_runtime import ExplainabilityRuntime
from core_runtime.core.audit_trace_store import (
    AuditTraceRecord,
    AuditTraceStore,
)


def _make_audit(task_hash: str = "h1", confidence: float = 0.9):
    return audit_execution(
        task_hash, confidence, 1.0 - confidence, 5, 0.001,
        "fast_converging", False, False,
    )


def _make_store_with_n_records(n: int = 3):
    store = AuditTraceStore()
    explainer = ExplainabilityRuntime()
    for i in range(n):
        audit = _make_audit(f"h{i:03d}", confidence=0.95 - i * 0.1)
        expl = explainer.explain(audit)
        store.append(audit, expl)
    return store


class TestAuditTraceRecord:

    def test_record_is_frozen(self):
        audit = _make_audit()
        explainer = ExplainabilityRuntime()
        expl = explainer.explain(audit)
        rec = AuditTraceRecord(audit=audit, explanation=expl)
        with pytest.raises(AttributeError):
            rec.report = None

    def test_record_to_dict(self):
        audit = _make_audit()
        explainer = ExplainabilityRuntime()
        expl = explainer.explain(audit)
        rec = AuditTraceRecord(audit=audit, explanation=expl)
        d = rec.to_dict()
        assert "audit" in d
        assert "explanation" in d
        assert d["report"] is None

    def test_record_with_failure_report(self):
        from core_runtime.core.failure_report import generate_failure_report
        audit = _make_audit()
        explainer = ExplainabilityRuntime()
        expl = explainer.explain(audit)
        report = generate_failure_report({"task_hash": "h1"}, "timeout", {
            "residual": 0.5, "confidence_score": 0.3,
            "projection_iterations": 10,
            "topology_signature": "bridge",
            "ood": False, "escalation_required": False,
            "trajectory_class": "standard",
        })
        rec = AuditTraceRecord(audit=audit, explanation=expl, report=report)
        d = rec.to_dict()
        assert d["report"] is not None
        assert d["report"]["error_type"] == "timeout"


class TestAuditTraceStore:

    def test_append_and_contains(self):
        store = _make_store_with_n_records(3)
        assert store.contains("h000")
        assert store.contains("h001")
        assert store.contains("h002")
        assert not store.contains("h999")

    def test_deterministic_ordering(self):
        store = _make_store_with_n_records(5)
        hashes = [r.audit.task_hash for r in store.all_records()]
        assert hashes == ["h000", "h001", "h002", "h003", "h004"]

    def test_get_existing(self):
        store = _make_store_with_n_records(3)
        rec = store.get("h000")
        assert rec is not None
        assert rec.audit.task_hash == "h000"

    def test_get_nonexistent_returns_none(self):
        store = _make_store_with_n_records(3)
        assert store.get("h999") is None

    def test_overwrite_preserves_order(self):
        store = AuditTraceStore()
        explainer = ExplainabilityRuntime()
        a1 = _make_audit("h001")
        a2 = _make_audit("h002")
        store.append(a1, explainer.explain(a1))
        store.append(a2, explainer.explain(a2))
        # Overwrite h001
        a1_new = _make_audit("h001")
        store.append(a1_new, explainer.explain(a1_new))
        assert len(store) == 2
        hashes = [r.audit.task_hash for r in store.all_records()]
        assert hashes == ["h001", "h002"]

    def test_len(self):
        store = _make_store_with_n_records(5)
        assert len(store) == 5

    def test_trust_level_distribution(self):
        store = AuditTraceStore()
        explainer = ExplainabilityRuntime()
        # Add a CERTAIN audit
        a1 = _make_audit("h1", confidence=0.95)
        store.append(a1, explainer.explain(a1))
        # Add a TRANSITIONAL audit
        a2 = audit_execution("h2", 0.6, 0.4, 6, 0.02, "fast_converging", False, False)
        store.append(a2, explainer.explain(a2))
        dist = store.trust_level_distribution()
        assert "certain" in dist
        assert "transitional" in dist

    def test_flag_distribution(self):
        store = AuditTraceStore()
        explainer = ExplainabilityRuntime()
        a = audit_execution("h1", 0.3, 0.8, 25, 0.5, "divergence_risk", True, True)
        store.append(a, explainer.explain(a))
        dist = store.flag_distribution()
        assert len(dist) > 0

    def test_escalation_rate(self):
        store = AuditTraceStore()
        explainer = ExplainabilityRuntime()
        a1 = _make_audit("h1", confidence=0.95)  # no escalation
        store.append(a1, explainer.explain(a1))
        a2 = audit_execution("h2", 0.3, 0.8, 25, 0.5, "divergence_risk", True, True)
        store.append(a2, explainer.explain(a2))
        rate = store.escalation_rate()
        assert 0.0 < rate <= 1.0

    def test_avg_confidence(self):
        store = _make_store_with_n_records(3)
        avg = store.avg_confidence()
        assert 0.0 <= avg <= 1.0

    def test_avg_residual(self):
        store = _make_store_with_n_records(3)
        avg = store.avg_residual()
        assert avg >= 0.0

    def test_empty_store_stats(self):
        store = AuditTraceStore()
        assert store.escalation_rate() == 0.0
        assert store.avg_confidence() == 0.0
        assert store.avg_residual() == 0.0
        assert store.trust_level_distribution() == {}
        assert store.flag_distribution() == {}

    def test_export_import_roundtrip(self):
        store = _make_store_with_n_records(5)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "audit.jsonl"
            store.export_jsonl(path)
            store2 = AuditTraceStore()
            store2.load_jsonl(path)
            assert len(store2) == 5
            # Verify ordering preserved
            h1 = [r.audit.task_hash for r in store.all_records()]
            h2 = [r.audit.task_hash for r in store2.all_records()]
            assert h1 == h2

    def test_atomic_export(self):
        store = _make_store_with_n_records(3)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "audit.jsonl"
            store.export_jsonl(path)
            content = path.read_text()
            lines = [l for l in content.strip().split("\n") if l.strip()]
            assert len(lines) == 3
            for line in lines:
                data = json.loads(line)
                assert "audit" in data
                assert "explanation" in data

    def test_same_input_same_trace(self):
        store1 = AuditTraceStore()
        store2 = AuditTraceStore()
        explainer = ExplainabilityRuntime()
        audit = _make_audit("h1")
        store1.append(audit, explainer.explain(audit))
        store2.append(audit, explainer.explain(audit))
        r1 = store1.get("h1")
        r2 = store2.get("h1")
        assert r1.audit.fingerprint() == r2.audit.fingerprint()
        assert r1.explanation.fingerprint() == r2.explanation.fingerprint()
