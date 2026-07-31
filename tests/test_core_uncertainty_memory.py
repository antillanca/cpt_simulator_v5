"""Tests for CORE v3.3 — Uncertainty Memory.

Mandatory tests:
- uncertainty memory stores degraded/transitional/indeterminate cases
- same input -> same record
- deterministic ordering
- atomic persistence
- entries are SHA-256 anchored
"""

import json
import tempfile
from pathlib import Path

import pytest

from core_runtime.core.uncertainty_memory import (
    UncertaintyEntry,
    UncertaintyMemory,
    UNCERTAINTY_REASONS,
)


def _make_entry(task_hash: str = "h1", reason: str = "divergence") -> UncertaintyEntry:
    return UncertaintyEntry(
        task_hash=task_hash,
        reason=reason,
        topology_signature="bridge",
        routing_action="increased_budget",
        projection_iterations=20,
        residual=0.5,
        confidence_score=0.3,
        trust_level="indeterminate",
        timestamp="2026-01-01T00:00:00Z",
        scheduler_context={},
        metadata={},
    )


class TestUncertaintyEntry:

    def test_entry_is_frozen(self):
        e = _make_entry()
        with pytest.raises(AttributeError):
            e.reason = "other"

    def test_anchor_deterministic(self):
        e1 = _make_entry()
        e2 = _make_entry()
        assert e1.anchor() == e2.anchor()

    def test_anchor_differs_for_different_input(self):
        e1 = _make_entry(task_hash="h1")
        e2 = _make_entry(task_hash="h2")
        assert e1.anchor() != e2.anchor()

    def test_to_dict_roundtrip(self):
        e = _make_entry()
        d = e.to_dict()
        e2 = UncertaintyEntry.from_dict(d)
        assert e.task_hash == e2.task_hash
        assert e.reason == e2.reason
        assert e.anchor() == e2.anchor()

    def test_known_reasons_include_divergence(self):
        assert "divergence" in UNCERTAINTY_REASONS
        assert "ood_execution" in UNCERTAINTY_REASONS


class TestUncertaintyMemory:

    def test_add_and_contains(self):
        um = UncertaintyMemory()
        um.add_entry(_make_entry("h1"))
        assert um.contains("h1")
        assert not um.contains("h2")

    def test_get_existing(self):
        um = UncertaintyMemory()
        um.add_entry(_make_entry("h1"))
        entry = um.get("h1")
        assert entry is not None
        assert entry.task_hash == "h1"

    def test_get_nonexistent_returns_none(self):
        um = UncertaintyMemory()
        assert um.get("h999") is None

    def test_search_by_reason(self):
        um = UncertaintyMemory()
        um.add_entry(_make_entry("h1", "divergence"))
        um.add_entry(_make_entry("h2", "low_confidence"))
        um.add_entry(_make_entry("h3", "divergence"))
        results = um.search_by_reason("divergence")
        assert len(results) == 2

    def test_deterministic_ordering(self):
        um = UncertaintyMemory()
        um.add_entry(_make_entry("h1"))
        um.add_entry(_make_entry("h2"))
        um.add_entry(_make_entry("h3"))
        all_entries = um.all_entries()
        assert [e.task_hash for e in all_entries] == ["h1", "h2", "h3"]

    def test_overwrite_preserves_order(self):
        um = UncertaintyMemory()
        um.add_entry(_make_entry("h1", "divergence"))
        um.add_entry(_make_entry("h2", "low_confidence"))
        um.add_entry(_make_entry("h1", "instability"))  # overwrite
        assert len(um) == 2
        assert um.get("h1").reason == "instability"
        assert [e.task_hash for e in um.all_entries()] == ["h1", "h2"]

    def test_same_input_same_record(self):
        um1 = UncertaintyMemory()
        um1.add_entry(_make_entry("h1"))
        um2 = UncertaintyMemory()
        um2.add_entry(_make_entry("h1"))
        assert um1.get("h1").anchor() == um2.get("h1").anchor()

    def test_export_import_jsonl(self):
        um = UncertaintyMemory()
        um.add_entry(_make_entry("h1", "divergence"))
        um.add_entry(_make_entry("h2", "low_confidence"))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "uncertainty.jsonl"
            um.export_jsonl(path)
            um2 = UncertaintyMemory()
            um2.load_jsonl(path)
            assert len(um2) == 2
            assert um2.contains("h1")
            assert um2.contains("h2")
            assert um2.get("h1").anchor() == um.get("h1").anchor()

    def test_atomic_export(self):
        um = UncertaintyMemory()
        um.add_entry(_make_entry("h1"))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.jsonl"
            um.export_jsonl(path)
            content = path.read_text()
            lines = [l for l in content.strip().split("\n") if l.strip()]
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["task_hash"] == "h1"
            assert "anchor" in data

    def test_reason_distribution(self):
        um = UncertaintyMemory()
        um.add_entry(_make_entry("h1", "divergence"))
        um.add_entry(_make_entry("h2", "divergence"))
        um.add_entry(_make_entry("h3", "low_confidence"))
        dist = um.reason_distribution()
        assert dist["divergence"] == 2
        assert dist["low_confidence"] == 1

    def test_trust_level_distribution(self):
        um = UncertaintyMemory()
        um.add_entry(_make_entry("h1"))
        assert um.get("h1").trust_level == "indeterminate"
        dist = um.trust_level_distribution()
        assert dist.get("indeterminate", 0) == 1
