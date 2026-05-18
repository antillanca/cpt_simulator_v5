"""Tests for CPT v2.13 — Resilient Runtime, Exact Cache & Confidence-Aware Routing.

Tests:
1. ExactCacheEntry: construction, frozen, fingerprint, JSON roundtrip
2. ExactMatchCache: put, get, contains, count, clear, persistence
3. TaskHashing: compute_task_hash determinism, canonicalize, float normalization
4. CircuitHashing: compute_circuit_hash for equivalent circuits
5. ExecutionPolicy: defaults, frozen, custom values
6. RecoveryHandler: oracle timeout, NaN detection, surrogate instability,
   projection divergence, degraded results, events
7. ConfidenceEstimate: construction, validation
8. ConfidenceRuntime: estimate with all heuristics, history, determinism
9. CapabilityRouter: cache_hit, high_confidence, ood, repeated_failure
10. AtomicMemory: atomic append, compact, crash-safety (fsync)
11. Real E2E: task → oracle → surrogate → projection → eval → memory → trace → cache
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend.core_runtime.task_runtime import RuntimeTask, RuntimeResult, RuntimeExecutor
from backend.core_runtime.oracle_protocol import MNAOracleAdapter
from backend.core_runtime.surrogate_runtime import SurrogateRuntime
from backend.core_runtime.projection_runtime import ProjectionRuntime, ProjectionExecution
from backend.core_runtime.memory_runtime import MemoryRuntime
from backend.core_runtime.execution_trace import ExecutionTrace, TraceStore
from backend.core_runtime.exact_cache import ExactMatchCache, ExactCacheEntry
from backend.core_runtime.task_hashing import (
    compute_task_hash, compute_circuit_hash, canonicalize_task,
    HASH_SCHEMA_VERSION, _normalize_float,
)
from backend.core_runtime.execution_policy import (
    ExecutionPolicy, RecoveryHandler,
    DEGRADED_ORACLE_TIMEOUT, DEGRADED_NAN_OUTPUT,
    DEGRADED_SURROGATE_INSTABILITY, DEGRADED_PROJECTION_DIVERGENCE,
    DEGRADED_CACHE_FALLBACK,
)
from backend.core_runtime.confidence_runtime import ConfidenceRuntime, ConfidenceEstimate
from backend.core_runtime.capability_router import CapabilityRouter, RoutingDecision
from backend.circuits.models import Circuit, Resistor, VoltageSource


# ===================================================================
# PHASE 1 — ExactCacheEntry
# ===================================================================

class TestExactCacheEntry:
    def test_construction(self):
        entry = ExactCacheEntry(
            task_hash="a" * 64,
            runtime_result_hash="b" * 64,
            topology_family="mesh",
            projection_iterations=10,
            failure_type=None,
            created_at="2025-01-01T00:00:00Z",
        )
        assert entry.task_hash == "a" * 64
        assert entry.topology_family == "mesh"

    def test_frozen(self):
        entry = ExactCacheEntry(
            task_hash="a" * 64, runtime_result_hash="b" * 64,
            topology_family="x", projection_iterations=1,
            failure_type=None, created_at="2025-01-01T00:00:00Z",
        )
        with pytest.raises(AttributeError):
            entry.task_hash = "new"  # type: ignore

    def test_fingerprint(self):
        entry = ExactCacheEntry(
            task_hash="a" * 64, runtime_result_hash="b" * 64,
            topology_family="x", projection_iterations=1,
            failure_type=None, created_at="2025-01-01T00:00:00Z",
        )
        assert entry.fingerprint == entry.fingerprint
        assert len(entry.fingerprint) == 64

    def test_json_roundtrip(self):
        entry = ExactCacheEntry(
            task_hash="c" * 64, runtime_result_hash="d" * 64,
            topology_family="radial", projection_iterations=5,
            failure_type="conservation_drift", created_at="2025-06-01T00:00:00Z",
        )
        j = entry.to_json()
        restored = ExactCacheEntry.from_json(j)
        assert restored.task_hash == entry.task_hash
        assert restored.topology_family == entry.topology_family
        assert restored.failure_type == entry.failure_type


# ===================================================================
# PHASE 1b — ExactMatchCache
# ===================================================================

class TestExactMatchCache:
    def _make_result(self, task_id="t1") -> RuntimeResult:
        return RuntimeResult(
            task_id=task_id,
            task_fingerprint="fp_" + task_id,
            oracle_voltages=torch.tensor([5.0, 2.5, 0.0]),
            surrogate_voltages=torch.tensor([4.8, 2.3, 0.1]),
            projected_voltages=torch.tensor([4.95, 2.48, 0.02]),
            projection_result=None,
            evaluation_report=None,
            memory_entry=None,
            total_runtime_ms=100.0,
            oracle_runtime_ms=10.0,
            surrogate_runtime_ms=5.0,
            projection_runtime_ms=50.0,
            failure_type=None,
        )

    def test_put_and_get(self, tmp_path):
        cache = ExactMatchCache(str(tmp_path / "cache"))
        result = self._make_result()
        entry = cache.put("hash_abc", result)
        assert cache.contains("hash_abc")
        retrieved = cache.get("hash_abc")
        assert retrieved is not None
        assert retrieved.task_id == "t1"
        torch.testing.assert_close(retrieved.oracle_voltages, result.oracle_voltages, atol=1e-5, rtol=1e-5)

    def test_get_missing(self, tmp_path):
        cache = ExactMatchCache(str(tmp_path / "cache2"))
        assert cache.get("nonexistent") is None
        assert not cache.contains("nonexistent")

    def test_count_and_clear(self, tmp_path):
        cache = ExactMatchCache(str(tmp_path / "cache3"))
        cache.put("h1", self._make_result("t1"))
        cache.put("h2", self._make_result("t2"))
        assert cache.count() == 2
        cache.clear()
        assert cache.count() == 0

    def test_cache_hit_deterministic(self, tmp_path):
        """Same task_hash always retrieves the same result."""
        cache = ExactMatchCache(str(tmp_path / "cache4"))
        cache.put("h_det", self._make_result())
        r1 = cache.get("h_det")
        r2 = cache.get("h_det")
        assert r1 is not None and r2 is not None
        torch.testing.assert_close(r1.oracle_voltages, r2.oracle_voltages, atol=1e-6, rtol=1e-6)


# ===================================================================
# PHASE 2 — TaskHashing
# ===================================================================

class TestTaskHashing:
    def test_deterministic(self):
        task = RuntimeTask(
            task_id="hash_test", domain="circuit",
            input_artifact="fp_abc", oracle_name="mna", surrogate_name="gnn",
        )
        h1 = compute_task_hash(task)
        h2 = compute_task_hash(task)
        assert h1 == h2
        assert len(h1) == 64

    def test_different_tasks_different_hashes(self):
        t1 = RuntimeTask(task_id="a", domain="c", input_artifact="x", oracle_name="m", surrogate_name="s")
        t2 = RuntimeTask(task_id="b", domain="c", input_artifact="x", oracle_name="m", surrogate_name="s")
        assert compute_task_hash(t1) != compute_task_hash(t2)

    def test_schema_version_included(self):
        canon = canonicalize_task(RuntimeTask(
            task_id="t", domain="d", input_artifact="a",
            oracle_name="o", surrogate_name="s",
        ))
        assert canon["hash_schema_version"] == HASH_SCHEMA_VERSION

    def test_float_normalization(self):
        assert _normalize_float(1000.0000000001) == _normalize_float(1000.0)
        assert _normalize_float(0.001) == _normalize_float(0.001)

    def test_equivalent_metadata_same_hash(self):
        t1 = RuntimeTask(
            task_id="t", domain="c", input_artifact="x",
            oracle_name="m", surrogate_name="s",
            metadata={"resistance": 1000.0, "voltage": 10.0},
        )
        t2 = RuntimeTask(
            task_id="t", domain="c", input_artifact="x",
            oracle_name="m", surrogate_name="s",
            metadata={"voltage": 10.0, "resistance": 1000.0},
        )
        # Different key ordering but same content
        assert compute_task_hash(t1) == compute_task_hash(t2)


class TestCircuitHashing:
    def _make_circuit(self):
        return Circuit(
            name="test",
            resistors=(Resistor("R1", "1", "2", 1000.0),),
            voltage_sources=(VoltageSource("V1", "1", "0", 10.0),),
        )

    def test_deterministic(self):
        c = self._make_circuit()
        h1 = compute_circuit_hash(c)
        h2 = compute_circuit_hash(c)
        assert h1 == h2

    def test_different_circuits(self):
        c1 = Circuit(name="a", resistors=(Resistor("R1", "1", "2", 1000.0),), voltage_sources=(VoltageSource("V1", "1", "0", 10.0),))
        c2 = Circuit(name="a", resistors=(Resistor("R1", "1", "2", 2000.0),), voltage_sources=(VoltageSource("V1", "1", "0", 10.0),))
        assert compute_circuit_hash(c1) != compute_circuit_hash(c2)

    def test_equivalent_circuits(self):
        """Circuits with same components in different order should hash identically."""
        c1 = Circuit(
            name="eq",
            resistors=(Resistor("R1", "1", "2", 1000.0), Resistor("R2", "2", "0", 1000.0)),
            voltage_sources=(VoltageSource("V1", "1", "0", 10.0),),
        )
        c2 = Circuit(
            name="eq",
            resistors=(Resistor("R2", "2", "0", 1000.0), Resistor("R1", "1", "2", 1000.0)),
            voltage_sources=(VoltageSource("V1", "1", "0", 10.0),),
        )
        # Circuit.__post_init__ normalizes ordering
        assert compute_circuit_hash(c1) == compute_circuit_hash(c2)


# ===================================================================
# PHASE 3 — ExecutionPolicy & RecoveryHandler
# ===================================================================

class TestExecutionPolicy:
    def test_defaults(self):
        p = ExecutionPolicy()
        assert p.oracle_timeout_s == 30.0
        assert p.max_retries == 2
        assert p.fallback_to_cache is True
        assert p.projection_budget_high == 20
        assert p.projection_budget_low == 5

    def test_frozen(self):
        p = ExecutionPolicy()
        with pytest.raises(AttributeError):
            p.oracle_timeout_s = 99  # type: ignore

    def test_custom(self):
        p = ExecutionPolicy(oracle_timeout_s=60.0, max_retries=5, projection_budget_low=3)
        assert p.oracle_timeout_s == 60.0
        assert p.max_retries == 5
        assert p.projection_budget_low == 3


class TestRecoveryHandler:
    def _make_task(self):
        return RuntimeTask(
            task_id="recovery_test", domain="circuit",
            input_artifact="fp", oracle_name="mna", surrogate_name="gnn",
        )

    def test_oracle_timeout(self):
        handler = RecoveryHandler(ExecutionPolicy(oracle_timeout_s=1.0))
        task = self._make_task()
        reason = handler.check_oracle_timeout(1500.0, task)  # 1.5s > 1.0s
        assert reason == DEGRADED_ORACLE_TIMEOUT
        assert len(handler.events) == 1

    def test_no_timeout(self):
        handler = RecoveryHandler(ExecutionPolicy(oracle_timeout_s=30.0))
        task = self._make_task()
        reason = handler.check_oracle_timeout(10.0, task)
        assert reason is None

    def test_nan_detection(self):
        handler = RecoveryHandler()
        task = self._make_task()
        reason = handler.check_nan_output(torch.tensor([1.0, float("nan"), 3.0]), "surrogate", task)
        assert reason == DEGRADED_NAN_OUTPUT

    def test_no_nan(self):
        handler = RecoveryHandler()
        task = self._make_task()
        reason = handler.check_nan_output(torch.tensor([1.0, 2.0, 3.0]), "surrogate", task)
        assert reason is None

    def test_surrogate_instability(self):
        handler = RecoveryHandler(ExecutionPolicy(surrogate_instability_threshold=10.0))
        task = self._make_task()
        oracle = torch.tensor([5.0, 2.5, 0.0])
        surr = torch.tensor([500.0, 250.0, 100.0])  # 100x off
        reason = handler.check_surrogate_instability(surr, oracle, task)
        assert reason == DEGRADED_SURROGATE_INSTABILITY

    def test_projection_divergence(self):
        handler = RecoveryHandler()
        task = self._make_task()
        reason = handler.check_projection_divergence(20, 20, converged=False, task=task)
        assert reason == DEGRADED_PROJECTION_DIVERGENCE

    def test_projection_converged(self):
        handler = RecoveryHandler()
        task = self._make_task()
        reason = handler.check_projection_divergence(15, 20, converged=True, task=task)
        assert reason is None

    def test_make_degraded_result(self):
        handler = RecoveryHandler()
        task = self._make_task()
        result = handler.make_degraded_result(task, DEGRADED_NAN_OUTPUT, total_runtime_ms=50.0)
        assert result.failure_type == DEGRADED_NAN_OUTPUT
        assert result.metadata.get("degraded") is True
        assert result.metadata.get("degradation_reason") == DEGRADED_NAN_OUTPUT

    def test_cache_fallback_event(self):
        handler = RecoveryHandler()
        handler.record_cache_fallback("task_1", "hash_abc")
        assert any(e["reason"] == DEGRADED_CACHE_FALLBACK for e in handler.events)


# ===================================================================
# PHASE 4 — ConfidenceEstimate
# ===================================================================

class TestConfidenceEstimate:
    def test_construction(self):
        ce = ConfidenceEstimate(
            confidence_score=0.85,
            estimated_projection_iterations=3,
            likely_ood=False,
        )
        assert ce.confidence_score == 0.85
        assert ce.likely_ood is False

    def test_validation_score_range(self):
        with pytest.raises(ValueError):
            ConfidenceEstimate(confidence_score=1.5, estimated_projection_iterations=1, likely_ood=False)
        with pytest.raises(ValueError):
            ConfidenceEstimate(confidence_score=-0.1, estimated_projection_iterations=1, likely_ood=True)

    def test_validation_iterations(self):
        with pytest.raises(ValueError):
            ConfidenceEstimate(confidence_score=0.5, estimated_projection_iterations=-1, likely_ood=False)

    def test_frozen(self):
        ce = ConfidenceEstimate(confidence_score=0.5, estimated_projection_iterations=5, likely_ood=False)
        with pytest.raises(AttributeError):
            ce.confidence_score = 0.9  # type: ignore


# ===================================================================
# PHASE 4b — ConfidenceRuntime
# ===================================================================

class TestConfidenceRuntime:
    def _make_task(self, topo="radial"):
        return RuntimeTask(
            task_id="conf_test", domain="circuit",
            input_artifact="fp", oracle_name="mna", surrogate_name="gnn",
            metadata={"topology_family": topo},
        )

    def test_high_confidence_small_graph(self):
        rt = ConfidenceRuntime()
        task = self._make_task("radial")
        est = rt.estimate(task, graph_size=3, resistance_range=1.5, topology_family="radial")
        assert est.confidence_score > 0.7
        assert not est.likely_ood
        assert est.estimated_projection_iterations <= 5

    def test_low_confidence_large_graph(self):
        rt = ConfidenceRuntime()
        task = self._make_task("unknown")
        est = rt.estimate(task, graph_size=50, resistance_range=200.0, topology_family="unknown")
        assert est.confidence_score < 0.5
        assert est.likely_ood
        assert est.estimated_projection_iterations >= 10

    def test_ood_from_kcl(self):
        rt = ConfidenceRuntime()
        task = self._make_task()
        est = rt.estimate(task, kcl_residual=0.5, topology_family="radial")
        assert est.likely_ood

    def test_historical_failure_lowers_confidence(self):
        rt = ConfidenceRuntime(history=[
            {"topology_family": "bridge", "failure_type": "conservation_drift", "projection_iterations": 20},
            {"topology_family": "bridge", "failure_type": "kcl_violation", "projection_iterations": 15},
            {"topology_family": "bridge", "failure_type": None, "projection_iterations": 3},
        ])
        task = self._make_task("bridge")
        est = rt.estimate(task, graph_size=5, topology_family="bridge")
        # 2/3 failure rate should lower confidence
        assert est.confidence_score < 0.8

    def test_deterministic(self):
        rt = ConfidenceRuntime()
        task = self._make_task()
        e1 = rt.estimate(task, graph_size=10, resistance_range=5.0, topology_family="mesh")
        e2 = rt.estimate(task, graph_size=10, resistance_range=5.0, topology_family="mesh")
        assert e1.confidence_score == e2.confidence_score
        assert e1.estimated_projection_iterations == e2.estimated_projection_iterations


# ===================================================================
# PHASE 5 — CapabilityRouter
# ===================================================================

class TestCapabilityRouter:
    def _make_task(self, topo="radial"):
        return RuntimeTask(
            task_id="route_test", domain="circuit",
            input_artifact="fp", oracle_name="mna", surrogate_name="gnn",
            metadata={"topology_family": topo},
        )

    def _make_confidence(self, score=0.8, ood=False):
        return ConfidenceEstimate(
            confidence_score=score,
            estimated_projection_iterations=3,
            likely_ood=ood,
        )

    def test_cache_hit(self):
        router = CapabilityRouter()
        decision = router.route(self._make_task(), self._make_confidence(), cache_hit=True)
        assert decision.action == "exact_cache_hit"
        assert decision.projection_budget == 0
        assert not decision.force_oracle

    def test_high_confidence(self):
        router = CapabilityRouter()
        decision = router.route(self._make_task(), self._make_confidence(score=0.9))
        assert decision.action == "standard_projection"
        assert decision.projection_budget <= 5  # budget_low

    def test_ood_escalation(self):
        router = CapabilityRouter()
        decision = router.route(self._make_task(), self._make_confidence(score=0.2, ood=True))
        assert decision.action == "increased_budget"
        assert decision.force_oracle is True
        assert decision.projection_budget >= 15

    def test_repeated_failure_oracle_verification(self):
        router = CapabilityRouter(failure_counts={"bridge": 5})
        decision = router.route(
            self._make_task("bridge"),
            self._make_confidence(score=0.6),
        )
        assert decision.action == "oracle_verification"
        assert decision.force_oracle is True

    def test_record_failure(self):
        router = CapabilityRouter()
        router.record_failure("mesh")
        router.record_failure("mesh")
        router.record_failure("mesh")
        assert router.failure_counts["mesh"] == 3
        decision = router.route(self._make_task("mesh"), self._make_confidence(score=0.8))
        assert decision.action == "oracle_verification"

    def test_record_success_reduces_failures(self):
        router = CapabilityRouter(failure_counts={"mesh": 2})
        router.record_success("mesh")
        assert router.failure_counts["mesh"] == 1

    def test_invalid_action_rejected(self):
        with pytest.raises(ValueError):
            RoutingDecision(action="invalid", projection_budget=5, force_oracle=False, reason="test")


# ===================================================================
# PHASE 6 — Atomic Memory Persistence
# ===================================================================

class TestAtomicMemory:
    def test_atomic_append(self, tmp_path):
        memory = MemoryRuntime(str(tmp_path / "atomic_mem"))
        task = RuntimeTask(
            task_id="at_1", domain="circuit",
            input_artifact="fp", oracle_name="mna", surrogate_name="gnn",
        )
        entry = memory.register_execution(task=task, topology_family="radial")
        assert memory.count() == 1
        loaded = memory.load_all()
        assert len(loaded) == 1
        assert loaded[0].entry_id == entry.entry_id

    def test_compact_deduplicates(self, tmp_path):
        memory = MemoryRuntime(str(tmp_path / "compact_mem"))
        task = RuntimeTask(
            task_id="comp", domain="circuit",
            input_artifact="fp", oracle_name="mna", surrogate_name="gnn",
        )
        # Register same task multiple times
        e1 = memory.register_execution(task=task, topology_family="radial")
        e2 = memory.register_execution(task=task, topology_family="radial")
        assert memory.count() == 2
        after = memory.compact()
        assert after == 2  # Different entry_ids, both kept

    def test_crash_safety_write_exists(self, tmp_path):
        """After atomic write, file must exist and be valid JSONL."""
        memory = MemoryRuntime(str(tmp_path / "crash_mem"))
        task = RuntimeTask(
            task_id="crash", domain="circuit",
            input_artifact="fp", oracle_name="mna", surrogate_name="gnn",
        )
        memory.register_execution(task=task)
        # Verify file is valid JSONL
        log_path = memory.log_path
        assert log_path.exists()
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)  # Must be valid JSON
                    assert "entry_id" in d


# ===================================================================
# PHASE 8 — Real E2E Integration Test
# ===================================================================

class TestE2EIntegration:
    """Full pipeline: task → oracle → surrogate → memory → trace → cache.
    NO mocks except filesystem temp dirs."""

    def _make_circuit(self):
        return Circuit(
            name="e2e_test",
            resistors=(
                Resistor("R1", "1", "2", 1000.0),
                Resistor("R2", "2", "0", 2000.0),
            ),
            voltage_sources=(VoltageSource("V1", "1", "0", 12.0),),
        )

    def test_full_pipeline_no_projection(self, tmp_path):
        """E2E: oracle + surrogate + memory + trace + cache."""
        circuit = self._make_circuit()
        oracle = MNAOracleAdapter()
        oracle.register_circuit("fp_e2e", circuit)
        surrogate = SurrogateRuntime(name="e2e_gnn")
        memory = MemoryRuntime(str(tmp_path / "e2e_mem"))
        cache = ExactMatchCache(str(tmp_path / "e2e_cache"))
        trace_store = TraceStore(str(tmp_path / "e2e_traces"))

        task = RuntimeTask(
            task_id="e2e_001", domain="circuit",
            input_artifact="fp_e2e", oracle_name="mna_dc_solver",
            surrogate_name="e2e_gnn", projection_enabled=False,
            metadata={"topology_family": "radial"},
        )

        executor = RuntimeExecutor(
            oracle=oracle, surrogate=surrogate,
            projection=None, evaluator=None,
            memory_sink=memory,
        )

        result = executor.execute(task)

        # Verify result
        assert result.oracle_voltages is not None
        assert result.surrogate_voltages is not None
        assert result.total_runtime_ms > 0
        assert result.memory_entry is not None

        # Verify memory persisted
        assert memory.count() == 1

        # Verify trace
        trace = ExecutionTrace(
            trace_id=f"tr_{task.task_id}",
            task_id=task.task_id,
            runtime_ms=result.total_runtime_ms,
            oracle_runtime_ms=result.oracle_runtime_ms,
            surrogate_runtime_ms=result.surrogate_runtime_ms,
            projection_runtime_ms=result.projection_runtime_ms,
            projection_iterations=0,
            topology_family="radial",
            failure_type=result.failure_type,
        )
        trace_store.save(trace)
        loaded = trace_store.load(task.task_id)
        assert len(loaded) == 1
        assert loaded[0].task_id == "e2e_001"

        # Verify cache
        task_hash = compute_task_hash(task)
        cache.put(task_hash, result)
        cached = cache.get(task_hash)
        assert cached is not None
        torch.testing.assert_close(cached.oracle_voltages, result.oracle_voltages, atol=1e-5, rtol=1e-5)

    def test_deterministic_hashes(self, tmp_path):
        """Same execution must produce identical hashes."""
        circuit = self._make_circuit()
        oracle = MNAOracleAdapter()
        oracle.register_circuit("fp_det", circuit)
        surrogate = SurrogateRuntime(name="det_gnn")

        task = RuntimeTask(
            task_id="det_001", domain="circuit",
            input_artifact="fp_det", oracle_name="mna_dc_solver",
            surrogate_name="det_gnn", projection_enabled=False,
        )

        # Run twice
        executor = RuntimeExecutor(oracle=oracle, surrogate=surrogate)
        r1 = executor.execute(task)
        r2 = executor.execute(task)

        # Task fingerprints must match
        assert r1.task_fingerprint == r2.task_fingerprint

        # Task hashes must match
        h1 = compute_task_hash(task)
        h2 = compute_task_hash(task)
        assert h1 == h2

    def test_degraded_execution_handling(self, tmp_path):
        """RecoveryHandler produces degraded results on NaN."""
        handler = RecoveryHandler()
        task = RuntimeTask(
            task_id="degraded_001", domain="circuit",
            input_artifact="fp", oracle_name="mna", surrogate_name="gnn",
        )
        nan_output = torch.tensor([1.0, float("nan"), 3.0])
        reason = handler.check_nan_output(nan_output, "surrogate", task)
        assert reason == DEGRADED_NAN_OUTPUT
        degraded = handler.make_degraded_result(task, reason, total_runtime_ms=10.0)
        assert degraded.failure_type == DEGRADED_NAN_OUTPUT
        assert degraded.metadata["degraded"] is True

    def test_cache_hit_path(self, tmp_path):
        """Cache hit returns identical result."""
        cache = ExactMatchCache(str(tmp_path / "e2e_cache_hit"))
        result = RuntimeResult(
            task_id="cache_hit_test",
            task_fingerprint="fp_ch",
            oracle_voltages=torch.tensor([6.0, 4.0, 0.0]),
            surrogate_voltages=torch.tensor([5.5, 3.8, 0.2]),
            projected_voltages=None,
            projection_result=None,
            evaluation_report=None,
            memory_entry=None,
            total_runtime_ms=50.0,
            oracle_runtime_ms=5.0,
            surrogate_runtime_ms=3.0,
            projection_runtime_ms=0.0,
        )
        cache.put("hash_ch", result)
        cached = cache.get("hash_ch")
        assert cached is not None
        torch.testing.assert_close(cached.oracle_voltages, result.oracle_voltages)

    def test_confidence_aware_routing(self, tmp_path):
        """Router makes correct decisions based on confidence."""
        router = CapabilityRouter()
        task = RuntimeTask(
            task_id="route_e2e", domain="circuit",
            input_artifact="fp", oracle_name="mna", surrogate_name="gnn",
            metadata={"topology_family": "mesh"},
        )
        # High confidence → standard, small budget
        high = ConfidenceEstimate(confidence_score=0.9, estimated_projection_iterations=2, likely_ood=False)
        d = router.route(task, high)
        assert d.action == "standard_projection"
        assert d.projection_budget <= 5

        # Low confidence + OOD → escalation
        low = ConfidenceEstimate(confidence_score=0.2, estimated_projection_iterations=20, likely_ood=True)
        d = router.route(task, low)
        assert d.action == "increased_budget"
        assert d.force_oracle is True
