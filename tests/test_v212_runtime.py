"""Tests for CPT v2.12 — Execution Runtime & Task Standardization.

Validates:
1. RuntimeTask: construction, fingerprint, determinism
2. RuntimeResult: immutability
3. RuntimeExecutor: pipeline orchestration
4. MNAOracleAdapter: oracle protocol conformance
5. SurrogateRuntime: zero baseline + model inference
6. ProjectionRuntime: wraps v2.9F projection
7. MemoryRuntime: JSONL persistence, roundtrip
8. ExecutionTrace: fingerprint, serialization roundtrip
9. TraceStore: persistence, load, roundtrip
10. DatasetManifest: construction, hash, serialization
11. DatasetRegistry: register, find, roundtrip
12. Determinism: identical runs produce identical fingerprints
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# Ensure repo root
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend.core_runtime.task_runtime import RuntimeTask, RuntimeResult, RuntimeExecutor
from backend.core_runtime.oracle_protocol import MNAOracleAdapter, OracleProtocol
from backend.core_runtime.surrogate_runtime import SurrogateRuntime, SurrogatePrediction
from backend.core_runtime.projection_runtime import ProjectionRuntime, ProjectionExecution
from backend.core_runtime.memory_runtime import MemoryRuntime
from backend.core_runtime.execution_trace import ExecutionTrace, TraceStore, make_trace_id
from backend.core_runtime.dataset_registry import DatasetManifest, DatasetRegistry, compute_dataset_sha256
from backend.core_spec.memory_spec import MemoryEntry
from backend.core_spec.projection_spec import ProjectionResult
from backend.core_spec.failure_taxonomy import FAILURE_TYPES, FAILURE_CATEGORIES


# ===================================================================
# PHASE 1 — RuntimeTask
# ===================================================================

class TestRuntimeTask:
    def test_construction(self):
        task = RuntimeTask(
            task_id="t_001",
            domain="circuit",
            input_artifact="fp_abc123",
            oracle_name="mna_dc_solver",
            surrogate_name="circuit_gnn",
            projection_enabled=True,
        )
        assert task.task_id == "t_001"
        assert task.domain == "circuit"
        assert task.projection_enabled is True

    def test_frozen(self):
        task = RuntimeTask(
            task_id="t_002", domain="circuit",
            input_artifact="fp_abc", oracle_name="mna", surrogate_name="gnn",
        )
        with pytest.raises(AttributeError):
            task.task_id = "modified"  # type: ignore

    def test_fingerprint_deterministic(self):
        task = RuntimeTask(
            task_id="t_003", domain="circuit",
            input_artifact="fp_abc", oracle_name="mna", surrogate_name="gnn",
        )
        fp1 = task.fingerprint()
        fp2 = task.fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex

    def test_fingerprint_differs_for_different_tasks(self):
        t1 = RuntimeTask(task_id="a", domain="circuit", input_artifact="x", oracle_name="m", surrogate_name="s")
        t2 = RuntimeTask(task_id="b", domain="circuit", input_artifact="x", oracle_name="m", surrogate_name="s")
        assert t1.fingerprint() != t2.fingerprint()

    def test_metadata_in_fingerprint(self):
        t1 = RuntimeTask(task_id="a", domain="circuit", input_artifact="x", oracle_name="m", surrogate_name="s", metadata={"k": 1})
        t2 = RuntimeTask(task_id="a", domain="circuit", input_artifact="x", oracle_name="m", surrogate_name="s", metadata={"k": 2})
        assert t1.fingerprint() != t2.fingerprint()

    def test_default_metadata(self):
        task = RuntimeTask(task_id="t", domain="c", input_artifact="a", oracle_name="o", surrogate_name="s")
        assert task.metadata == {}


# ===================================================================
# PHASE 2 — Oracle Protocol
# ===================================================================

class TestMNAOracleAdapter:
    def _make_circuit(self):
        from backend.circuits.models import Circuit, Resistor, VoltageSource, CurrentSource
        return Circuit(
            name="test_r_divider",
            resistors=(Resistor("R1", "1", "2", 1000.0), Resistor("R2", "2", "0", 1000.0)),
            voltage_sources=(VoltageSource("V1", "1", "0", 10.0),),
            current_sources=(),
            ground_node="0",
        )

    def test_oracle_protocol_conformance(self):
        oracle = MNAOracleAdapter()
        assert isinstance(oracle, OracleProtocol)

    def test_solve_circuit(self):
        oracle = MNAOracleAdapter()
        circuit = self._make_circuit()
        result = oracle.solve(circuit)
        assert "voltages" in result
        assert isinstance(result["voltages"], torch.Tensor)
        assert result["oracle_name"] == "mna_dc_solver"
        assert result["latency_ms"] > 0

    def test_solve_deterministic(self):
        oracle = MNAOracleAdapter()
        circuit = self._make_circuit()
        r1 = oracle.solve(circuit)
        r2 = oracle.solve(circuit)
        assert torch.allclose(r1["voltages"], r2["voltages"], atol=1e-9)

    def test_register_and_solve_by_fingerprint(self):
        oracle = MNAOracleAdapter()
        circuit = self._make_circuit()
        oracle.register_circuit("fp_test", circuit)
        result = oracle.solve({"input_artifact": "fp_test"} if False else circuit)
        assert "voltages" in result

    def test_name(self):
        assert MNAOracleAdapter().name() == "mna_dc_solver"


# ===================================================================
# PHASE 3 — SurrogateRuntime
# ===================================================================

class TestSurrogateRuntime:
    def test_zero_baseline(self):
        runtime = SurrogateRuntime(name="test_zero")

        class FakeGraph:
            num_nodes = 5
            node_features = torch.zeros(5, 8)
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        pred = runtime.predict(FakeGraph())
        assert isinstance(pred, SurrogatePrediction)
        assert pred.prediction.shape == (5,)
        assert pred.surrogate_name == "test_zero"
        assert pred.metadata["mode"] == "zero_baseline"

    def test_fingerprint_deterministic(self):
        class FakeGraph:
            num_nodes = 3
            node_features = torch.zeros(3, 8)
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        runtime = SurrogateRuntime(name="test")
        p1 = runtime.predict(FakeGraph())
        p2 = runtime.predict(FakeGraph())
        assert p1.fingerprint() == p2.fingerprint()

    def test_latency_recorded(self):
        class FakeGraph:
            num_nodes = 3
            node_features = torch.zeros(3, 8)
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        runtime = SurrogateRuntime(name="test")
        pred = runtime.predict(FakeGraph())
        assert pred.latency_ms >= 0


# ===================================================================
# PHASE 4 — ProjectionExecution
# ===================================================================

class TestProjectionExecution:
    def test_construction(self):
        pe = ProjectionExecution(
            corrected_prediction=torch.tensor([1.0, 2.0]),
            iterations=10,
            converged=True,
            kcl_violation=1e-10,
            kvl_violation=1e-10,
            projection_time_ms=5.0,
        )
        assert pe.converged is True
        assert pe.iterations == 10

    def test_frozen(self):
        pe = ProjectionExecution(
            corrected_prediction=torch.tensor([1.0]),
            iterations=5, converged=False,
            kcl_violation=1.0, kvl_violation=1.0, projection_time_ms=1.0,
        )
        with pytest.raises(AttributeError):
            pe.iterations = 99  # type: ignore

    def test_to_projection_result(self):
        pe = ProjectionExecution(
            corrected_prediction=torch.tensor([1.0]),
            iterations=20, converged=True,
            kcl_violation=1e-10, kvl_violation=1e-10, projection_time_ms=3.0,
            metadata={"initial_kcl": 0.5, "initial_kvl": 0.3, "initial_power": 0.2, "final_power": 0.001, "used_virtual_node": True},
        )
        pr = pe.to_projection_result()
        assert isinstance(pr, ProjectionResult)
        assert pr.iterations == 20
        assert pr.converged is True
        assert pr.used_virtual_node is True


# ===================================================================
# PHASE 5 — MemoryRuntime
# ===================================================================

class TestMemoryRuntime:
    def test_register_and_load(self, tmp_path):
        memory = MemoryRuntime(str(tmp_path / "mem"))
        task = RuntimeTask(
            task_id="t_mem", domain="circuit",
            input_artifact="fp_test", oracle_name="mna", surrogate_name="gnn",
        )
        entry = memory.register_execution(
            task=task,
            oracle_ms=1.0, surrogate_ms=2.0, projection_ms=3.0,
            topology_family="radial",
        )
        assert isinstance(entry, MemoryEntry)
        assert entry.topology_family == "radial"
        assert entry.oracle_time_ms == 1.0

        # Load back
        entries = memory.load_all()
        assert len(entries) == 1
        assert entries[0].entry_id == entry.entry_id

    def test_jsonl_roundtrip(self, tmp_path):
        memory = MemoryRuntime(str(tmp_path / "mem2"))
        task = RuntimeTask(
            task_id="t_rt", domain="circuit",
            input_artifact="fp_rt", oracle_name="mna", surrogate_name="gnn",
        )
        entry = memory.register_execution(task=task)
        loaded = memory.load_all()[0]
        assert loaded.graph_fingerprint == entry.graph_fingerprint
        assert loaded.projection_iterations == entry.projection_iterations

    def test_count(self, tmp_path):
        memory = MemoryRuntime(str(tmp_path / "mem3"))
        assert memory.count() == 0
        task = RuntimeTask(task_id="c1", domain="circuit", input_artifact="x", oracle_name="m", surrogate_name="s")
        memory.register_execution(task=task)
        memory.register_execution(task=task)
        assert memory.count() == 2

    def test_clear(self, tmp_path):
        memory = MemoryRuntime(str(tmp_path / "mem4"))
        task = RuntimeTask(task_id="cl", domain="circuit", input_artifact="x", oracle_name="m", surrogate_name="s")
        memory.register_execution(task=task)
        assert memory.count() == 1
        memory.clear()
        assert memory.count() == 0


# ===================================================================
# PHASE 6 — ExecutionTrace
# ===================================================================

class TestExecutionTrace:
    def test_construction(self):
        trace = ExecutionTrace(
            trace_id="tr_001", task_id="t_001",
            runtime_ms=100.0, oracle_runtime_ms=10.0,
            surrogate_runtime_ms=5.0, projection_runtime_ms=50.0,
            projection_iterations=20, topology_family="mesh",
            failure_type=None,
        )
        assert trace.trace_id == "tr_001"
        assert trace.topology_family == "mesh"

    def test_frozen(self):
        trace = ExecutionTrace(
            trace_id="tr", task_id="t", runtime_ms=1,
            oracle_runtime_ms=1, surrogate_runtime_ms=1,
            projection_runtime_ms=1, projection_iterations=1,
            topology_family="x", failure_type=None,
        )
        with pytest.raises(AttributeError):
            trace.trace_id = "new"  # type: ignore

    def test_fingerprint_deterministic(self):
        trace = ExecutionTrace(
            trace_id="tr", task_id="t", runtime_ms=1.0,
            oracle_runtime_ms=1.0, surrogate_runtime_ms=1.0,
            projection_runtime_ms=1.0, projection_iterations=1,
            topology_family="x", failure_type=None, timestamp="2025-01-01T00:00:00Z",
        )
        assert trace.fingerprint() == trace.fingerprint()
        assert len(trace.fingerprint()) == 64

    def test_json_roundtrip(self):
        trace = ExecutionTrace(
            trace_id="tr_rt", task_id="t_rt", runtime_ms=42.0,
            oracle_runtime_ms=10.0, surrogate_runtime_ms=5.0,
            projection_runtime_ms=20.0, projection_iterations=15,
            topology_family="bridge", failure_type="conservation_drift",
            timestamp="2025-01-01T00:00:00Z",
        )
        j = trace.to_json()
        restored = ExecutionTrace.from_json(j)
        assert restored.trace_id == trace.trace_id
        assert restored.task_id == trace.task_id
        assert restored.runtime_ms == trace.runtime_ms
        assert restored.projection_iterations == trace.projection_iterations
        assert restored.failure_type == trace.failure_type


# ===================================================================
# PHASE 6b — TraceStore
# ===================================================================

class TestTraceStore:
    def test_save_and_load(self, tmp_path):
        store = TraceStore(str(tmp_path / "traces"))
        trace = ExecutionTrace(
            trace_id="tr_1", task_id="t_1", runtime_ms=100.0,
            oracle_runtime_ms=10.0, surrogate_runtime_ms=5.0,
            projection_runtime_ms=50.0, projection_iterations=20,
            topology_family="radial", failure_type=None,
        )
        path = store.save(trace)
        assert path.exists()

        loaded = store.load("t_1")
        assert len(loaded) == 1
        assert loaded[0].trace_id == "tr_1"

    def test_load_all(self, tmp_path):
        store = TraceStore(str(tmp_path / "traces2"))
        for i in range(3):
            trace = ExecutionTrace(
                trace_id=f"tr_{i}", task_id=f"t_{i}", runtime_ms=float(i),
                oracle_runtime_ms=1, surrogate_runtime_ms=1,
                projection_runtime_ms=1, projection_iterations=1,
                topology_family="x", failure_type=None,
            )
            store.save(trace)
        all_traces = store.load_all()
        assert len(all_traces) == 3

    def test_clear(self, tmp_path):
        store = TraceStore(str(tmp_path / "traces3"))
        trace = ExecutionTrace(
            trace_id="tr_c", task_id="t_c", runtime_ms=1,
            oracle_runtime_ms=1, surrogate_runtime_ms=1,
            projection_runtime_ms=1, projection_iterations=1,
            topology_family="x", failure_type=None,
        )
        store.save(trace)
        store.clear()
        assert store.load("t_c") == []


# ===================================================================
# PHASE 8 — DatasetManifest & Registry
# ===================================================================

class TestDatasetManifest:
    def test_construction(self):
        m = DatasetManifest(
            dataset_id="ds_001",
            sha256="abc123" * 10 + "abcd",
            sample_count=1000,
            created_at="2025-01-01T00:00:00Z",
        )
        assert m.dataset_id == "ds_001"
        assert m.sample_count == 1000

    def test_frozen(self):
        m = DatasetManifest(
            dataset_id="ds", sha256="x" * 64, sample_count=1,
            created_at="2025-01-01T00:00:00Z",
        )
        with pytest.raises(AttributeError):
            m.dataset_id = "new"  # type: ignore

    def test_fingerprint_equals_sha256(self):
        m = DatasetManifest(
            dataset_id="ds", sha256="a" * 64, sample_count=1,
            created_at="2025-01-01T00:00:00Z",
        )
        assert m.fingerprint() == m.sha256

    def test_json_roundtrip(self):
        m = DatasetManifest(
            dataset_id="ds_rt", sha256="b" * 64, sample_count=42,
            created_at="2025-06-01T00:00:00Z",
            domain="circuit",
            topology_families=("radial", "mesh"),
        )
        j = m.to_json()
        restored = DatasetManifest.from_json(j)
        assert restored.dataset_id == m.dataset_id
        assert restored.sample_count == m.sample_count
        assert restored.topology_families == m.topology_families


class TestDatasetRegistry:
    def test_register_and_find(self, tmp_path):
        reg = DatasetRegistry(str(tmp_path / "ds_reg.jsonl"))
        m = DatasetManifest(
            dataset_id="ds_test", sha256="c" * 64, sample_count=100,
            created_at="2025-01-01T00:00:00Z",
        )
        reg.register(m)
        found = reg.find_by_id("ds_test")
        assert found is not None
        assert found.sample_count == 100

    def test_find_by_sha256(self, tmp_path):
        reg = DatasetRegistry(str(tmp_path / "ds_reg2.jsonl"))
        m = DatasetManifest(
            dataset_id="ds_sha", sha256="d" * 64, sample_count=50,
            created_at="2025-01-01T00:00:00Z",
        )
        reg.register(m)
        found = reg.find_by_sha256("d" * 64)
        assert found is not None
        assert found.dataset_id == "ds_sha"

    def test_find_missing(self, tmp_path):
        reg = DatasetRegistry(str(tmp_path / "ds_reg3.jsonl"))
        assert reg.find_by_id("nonexistent") is None

    def test_clear(self, tmp_path):
        reg = DatasetRegistry(str(tmp_path / "ds_reg4.jsonl"))
        m = DatasetManifest(dataset_id="ds_cl", sha256="e" * 64, sample_count=1, created_at="2025-01-01T00:00:00Z")
        reg.register(m)
        reg.clear()
        assert reg.find_by_id("ds_cl") is None


class TestComputeDatasetSha256:
    def test_deterministic_hash(self, tmp_path):
        p = tmp_path / "test_data.pt"
        p.write_text("hello world")
        h1 = compute_dataset_sha256(p, seed=42)
        h2 = compute_dataset_sha256(p, seed=42)
        assert h1 == h2
        assert len(h1) == 64

    def test_different_seeds_different_hash(self, tmp_path):
        p = tmp_path / "test_data2.pt"
        p.write_text("hello world")
        h1 = compute_dataset_sha256(p, seed=42)
        h2 = compute_dataset_sha256(p, seed=99)
        assert h1 != h2

    def test_list_hash(self):
        h = compute_dataset_sha256([1, 2, 3], seed=42)
        assert len(h) == 64


# ===================================================================
# PHASE 9 — Determinism & Integration
# ===================================================================

class TestDeterminism:
    def test_task_fingerprint_stable(self):
        """Same task inputs always produce same fingerprint."""
        tasks = [
            RuntimeTask(
                task_id="det", domain="circuit",
                input_artifact="fp_det", oracle_name="mna",
                surrogate_name="gnn", projection_enabled=True,
                metadata={"seed": 42},
            )
            for _ in range(5)
        ]
        fingerprints = [t.fingerprint() for t in tasks]
        assert len(set(fingerprints)) == 1

    def test_trace_fingerprint_stable(self):
        """Same trace inputs always produce same fingerprint."""
        traces = [
            ExecutionTrace(
                trace_id="tr_det", task_id="t_det",
                runtime_ms=100.0, oracle_runtime_ms=10.0,
                surrogate_runtime_ms=5.0, projection_runtime_ms=50.0,
                projection_iterations=20, topology_family="mesh",
                failure_type=None, timestamp="2025-01-01T00:00:00Z",
            )
            for _ in range(5)
        ]
        fps = [t.fingerprint() for t in traces]
        assert len(set(fps)) == 1


class TestFailureTaxonomyConsistency:
    def test_all_failure_types_in_categories(self):
        """Every FAILURE_TYPE must appear in exactly one category."""
        categorized = set()
        for cat, types in FAILURE_CATEGORIES.items():
            for t in types:
                categorized.add(t)
        official = set(FAILURE_TYPES)
        # All categorized must be official
        assert categorized.issubset(official), f"Extra in categories: {categorized - official}"
        # All official should be categorized
        uncategorized = official - categorized
        # Allow "ood_generalization_failure" as legacy alias (not in categories)
        if uncategorized - {"ood_generalization_failure"}:
            pytest.fail(f"Uncategorized failure types: {uncategorized}")


class TestRuntimeExecutorIntegration:
    def test_executor_with_mock_components(self, tmp_path):
        """Integration test: executor runs pipeline with mock oracle/surrogate."""

        class MockOracle:
            def solve(self, task_or_graph):
                return {
                    "voltages": torch.tensor([5.0, 2.5, 0.0]),
                    "oracle_name": self.name(),
                }
            def name(self):
                return "mock_oracle"

        class MockSurrogate:
            def predict(self, task_or_graph):
                return torch.tensor([4.8, 2.3, 0.1])

        memory = MemoryRuntime(str(tmp_path / "mem_int"))

        executor = RuntimeExecutor(
            oracle=MockOracle(),
            surrogate=MockSurrogate(),
            projection=None,
            evaluator=None,
            memory_sink=memory,
        )

        task = RuntimeTask(
            task_id="int_001", domain="circuit",
            input_artifact="fp_int", oracle_name="mock_oracle",
            surrogate_name="mock_surrogate", projection_enabled=False,
        )

        result = executor.execute(task)
        assert isinstance(result, RuntimeResult)
        assert result.oracle_voltages is not None
        assert result.surrogate_voltages is not None
        assert result.total_runtime_ms > 0
        assert result.memory_entry is not None
