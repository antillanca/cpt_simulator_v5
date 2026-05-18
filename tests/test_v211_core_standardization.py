"""Tests for CPT v2.11 — Core Standardization Freeze.

Validates:
1. CanonicalCircuitGraph: construction, fingerprint, validation, serialization roundtrip
2. ProjectionResult: fingerprint determinism, serialization, validation
3. ModelMetadata: fingerprint determinism, serialization
4. CPTModel protocol compliance (CircuitGNNAdapter)
5. ExperimentSpec: fingerprint, validation, serialization
6. EvaluationReport: fingerprint, validation, serialization
7. Failure taxonomy: consistency, categories, validation
8. MemoryEntry: validation, serialization, failure reference
9. Cross-contract: from_circuit_graph conversion
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from backend.core_spec.graph_spec import (
    CanonicalCircuitGraph,
    TopologyFamily,
    compute_graph_fingerprint,
    compute_graph_fingerprint_from_dict,
    from_circuit_graph,
    validate_graph,
)
from backend.core_spec.projection_spec import ProjectionResult, from_projection_effort
from backend.core_spec.model_spec import (
    CPTModel,
    CircuitGNNAdapter,
    ModelMetadata,
)
from backend.core_spec.experiment_spec import ExperimentSpec
from backend.core_spec.report_spec import EvaluationReport
from backend.core_spec.failure_taxonomy import (
    FAILURE_TYPES,
    FailureCategory,
    category_of,
    is_valid_failure_type,
    validate_failure_type,
    validate_taxonomy_consistency,
)
from backend.core_spec.memory_spec import MemoryEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_node_features() -> torch.Tensor:
    return torch.tensor([[1.0, 0.5], [2.0, 0.3], [0.0, 1.0]], dtype=torch.float32)

@pytest.fixture
def sample_edge_index() -> torch.Tensor:
    return torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

@pytest.fixture
def sample_edge_features() -> torch.Tensor:
    return torch.tensor([[0.1, 0.9], [0.2, 0.8]], dtype=torch.float32)

@pytest.fixture
def canonical_graph(sample_node_features, sample_edge_index, sample_edge_features) -> CanonicalCircuitGraph:
    g = CanonicalCircuitGraph(
        graph_id="test-graph-001",
        fingerprint="placeholder",
        num_nodes=3,
        num_edges=2,
        node_features=sample_node_features,
        edge_index=sample_edge_index,
        edge_features=sample_edge_features,
        topology_family=TopologyFamily.RADIAL,
        cycle_count=0,
        connected_components=1,
        source_nodes=[0],
        ground_node=0,
        metadata={"source": "test"},
    )
    object.__setattr__(g, "fingerprint", compute_graph_fingerprint(g))
    return g


# ===========================================================================
# 1. CanonicalCircuitGraph
# ===========================================================================

class TestCanonicalCircuitGraph:

    def test_fingerprint_deterministic(self, canonical_graph):
        fp1 = canonical_graph.fingerprint
        fp2 = compute_graph_fingerprint(canonical_graph)
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex

    def test_fingerprint_changes_with_structure(self, canonical_graph, sample_node_features, sample_edge_features):
        different_index = torch.tensor([[0, 2], [2, 1]], dtype=torch.long)
        g2 = CanonicalCircuitGraph(
            graph_id="test-graph-002",
            fingerprint="placeholder",
            num_nodes=3,
            num_edges=2,
            node_features=sample_node_features,
            edge_index=different_index,
            edge_features=sample_edge_features,
            topology_family=TopologyFamily.RADIAL,
            cycle_count=0,
            connected_components=1,
            source_nodes=[0],
            ground_node=0,
        )
        object.__setattr__(g2, "fingerprint", compute_graph_fingerprint(g2))
        assert canonical_graph.fingerprint != g2.fingerprint

    def test_serialization_roundtrip(self, canonical_graph):
        d = canonical_graph.to_json_dict()
        assert isinstance(d, dict)
        assert d["topology_family"] == "radial"
        assert d["num_nodes"] == 3

        restored = CanonicalCircuitGraph.from_json_dict(d)
        assert restored.graph_id == canonical_graph.graph_id
        assert restored.num_nodes == canonical_graph.num_nodes
        assert torch.allclose(restored.node_features, canonical_graph.node_features)
        assert torch.equal(restored.edge_index, canonical_graph.edge_index)
        assert restored.fingerprint == canonical_graph.fingerprint

    def test_validation_valid_graph(self, canonical_graph):
        errors = canonical_graph.validate()
        assert errors == []

    def test_validation_invalid_num_nodes(self, sample_edge_index, sample_edge_features):
        g = CanonicalCircuitGraph(
            graph_id="bad", fingerprint="x", num_nodes=0, num_edges=2,
            node_features=torch.zeros(0, 2), edge_index=sample_edge_index,
            edge_features=sample_edge_features, topology_family=TopologyFamily.UNKNOWN,
            cycle_count=0, connected_components=1, source_nodes=[], ground_node=0,
        )
        errors = g.validate()
        assert any("num_nodes must be positive" in e for e in errors)

    def test_frozen(self, canonical_graph):
        with pytest.raises(AttributeError):
            canonical_graph.num_nodes = 99  # type: ignore


class TestTopologyFamily:

    def test_classify_radial(self):
        assert TopologyFamily.classify(0, 1, []) == TopologyFamily.RADIAL

    def test_classify_bridge(self):
        assert TopologyFamily.classify(1, 1, []) == TopologyFamily.BRIDGE

    def test_classify_mesh(self):
        assert TopologyFamily.classify(3, 1, []) == TopologyFamily.MESH

    def test_classify_current_source(self):
        assert TopologyFamily.classify(2, 1, [0]) == TopologyFamily.CURRENT_SOURCE

    def test_classify_disconnected(self):
        assert TopologyFamily.classify(0, 2, []) == TopologyFamily.UNKNOWN


# ===========================================================================
# 2. ProjectionResult
# ===========================================================================

class TestProjectionResult:

    @pytest.fixture
    def result(self) -> ProjectionResult:
        return ProjectionResult(
            iterations=5,
            initial_kcl_residual=0.1,
            final_kcl_residual=1e-8,
            initial_kvl_residual=0.05,
            final_kvl_residual=1e-9,
            initial_power_residual=0.01,
            final_power_residual=1e-7,
            converged=True,
            used_virtual_node=True,
            projection_time_ms=12.5,
        )

    def test_fingerprint_deterministic(self, result):
        fp1 = result.fingerprint
        fp2 = result.fingerprint
        assert fp1 == fp2
        assert len(fp1) == 64

    def test_serialization_roundtrip(self, result):
        d = result.to_json_dict()
        restored = ProjectionResult.from_json_dict(d)
        assert restored.iterations == result.iterations
        assert restored.converged == result.converged
        assert restored.fingerprint == result.fingerprint

    def test_validation(self, result):
        assert result.validate() == []

    def test_validation_negative_iterations(self):
        bad = ProjectionResult(
            iterations=-1, initial_kcl_residual=0.1, final_kcl_residual=0.0,
            initial_kvl_residual=0.0, final_kvl_residual=0.0,
            initial_power_residual=0.0, final_power_residual=0.0,
            converged=False, used_virtual_node=False, projection_time_ms=0.0,
        )
        errors = bad.validate()
        assert any("iterations" in e for e in errors)

    def test_frozen(self, result):
        with pytest.raises(AttributeError):
            result.iterations = 99  # type: ignore


# ===========================================================================
# 3. ModelMetadata + CPTModel Protocol
# ===========================================================================

class TestModelMetadata:

    @pytest.fixture
    def meta(self) -> ModelMetadata:
        return ModelMetadata(
            model_name="CircuitGNN",
            version="2.10",
            parameter_count=12345,
            topology_specialization=None,
            training_dataset_fingerprint="abc123",
            projection_aware=True,
        )

    def test_fingerprint_deterministic(self, meta):
        assert meta.fingerprint == meta.fingerprint
        assert len(meta.fingerprint) == 64

    def test_fingerprint_changes_with_version(self, meta):
        meta2 = ModelMetadata(
            model_name="CircuitGNN", version="2.11",
            parameter_count=12345, topology_specialization=None,
            training_dataset_fingerprint="abc123", projection_aware=True,
        )
        assert meta.fingerprint != meta2.fingerprint

    def test_serialization_roundtrip(self, meta):
        d = meta.to_json_dict()
        restored = ModelMetadata.from_json_dict(d)
        assert restored.model_name == meta.model_name
        assert restored.fingerprint == meta.fingerprint


class TestCircuitGNNAdapter:

    def test_adapter_satisfies_protocol(self):
        """CircuitGNNAdapter should satisfy CPTModel protocol."""
        # We can't easily instantiate a real GNN here, so we check the protocol
        assert issubclass(CircuitGNNAdapter, CPTModel)


# ===========================================================================
# 4. ExperimentSpec
# ===========================================================================

class TestExperimentSpec:

    @pytest.fixture
    def spec(self) -> ExperimentSpec:
        return ExperimentSpec(
            experiment_id="exp-001",
            seed=42,
            dataset_fingerprint="dataset-abc",
            checkpoint_fingerprint="ckpt-def",
            target_mode="blended_projection",
            topology_curriculum=True,
            projection_enabled=True,
            projection_config={"steps": 50, "alpha_kcl": 0.1},
            training_config={"lr": 1e-3, "epochs": 100},
            evaluation_config={"ood_ratio": 0.2},
        )

    def test_fingerprint_deterministic(self, spec):
        assert spec.fingerprint == spec.fingerprint
        assert len(spec.fingerprint) == 64

    def test_serialization_roundtrip(self, spec):
        d = spec.to_json_dict()
        restored = ExperimentSpec.from_json_dict(d)
        assert restored.experiment_id == spec.experiment_id
        assert restored.seed == spec.seed
        assert restored.fingerprint == spec.fingerprint

    def test_validation_valid(self, spec):
        assert spec.validate() == []

    def test_validation_invalid_target_mode(self):
        bad = ExperimentSpec(
            experiment_id="exp-bad", seed=42, dataset_fingerprint="x",
            checkpoint_fingerprint=None, target_mode="invalid_mode",
            topology_curriculum=False, projection_enabled=False,
            projection_config={}, training_config={}, evaluation_config={},
        )
        errors = bad.validate()
        assert any("target_mode" in e for e in errors)

    def test_yaml_output(self, spec):
        lines = spec.to_yaml_lines()
        assert any("experiment_id" in l for l in lines)
        assert any("seed" in l for l in lines)


# ===========================================================================
# 5. EvaluationReport
# ===========================================================================

class TestEvaluationReport:

    @pytest.fixture
    def report(self) -> EvaluationReport:
        return EvaluationReport(
            report_id="rpt-001",
            model_fingerprint="model-abc",
            dataset_fingerprint="data-def",
            iid_mae=0.05,
            ood_mae=0.12,
            iid_kcl_max=1e-4,
            ood_kcl_max=5e-4,
            iid_kvl_max=1e-5,
            ood_kvl_max=2e-5,
            projection_iterations_mean=4.2,
            speedup_factor=150.0,
            topology_metrics={"radial_mae": 0.03},
            failure_summary={"conservation_drift": 5},
        )

    def test_fingerprint_deterministic(self, report):
        assert report.fingerprint == report.fingerprint

    def test_serialization_roundtrip(self, report):
        d = report.to_json_dict()
        restored = EvaluationReport.from_json_dict(d)
        assert restored.report_id == report.report_id
        assert restored.fingerprint == report.fingerprint

    def test_validation_valid(self, report):
        assert report.validate() == []

    def test_validation_negative_mae(self):
        bad = EvaluationReport(
            report_id="rpt-bad", model_fingerprint="x", dataset_fingerprint="y",
            iid_mae=-1.0, ood_mae=0.1, iid_kcl_max=0.0, ood_kcl_max=0.0,
            iid_kvl_max=0.0, ood_kvl_max=0.0, projection_iterations_mean=0.0,
            speedup_factor=1.0,
        )
        errors = bad.validate()
        assert any("MAE" in e for e in errors)


# ===========================================================================
# 6. Failure Taxonomy
# ===========================================================================

class TestFailureTaxonomy:

    def test_types_sorted_alphabetically(self):
        assert FAILURE_TYPES == sorted(FAILURE_TYPES)

    def test_all_types_in_categories(self):
        errors = validate_taxonomy_consistency()
        assert errors == [], f"Taxonomy inconsistencies: {errors}"

    def test_is_valid_failure_type(self):
        assert is_valid_failure_type("conservation_drift") is True
        assert is_valid_failure_type("nonexistent_failure") is False

    def test_validate_failure_type_raises(self):
        with pytest.raises(ValueError, match="Invalid failure type"):
            validate_failure_type("not_a_real_failure")

    def test_category_of(self):
        assert category_of("conservation_drift") == FailureCategory.PHYSICS
        assert category_of("topology_collapse") == FailureCategory.TOPOLOGY
        assert category_of("projection_overshoot") == FailureCategory.PROJECTION
        assert category_of("ood_voltage_explosion") == FailureCategory.OOD

    def test_legacy_type_in_taxonomy(self):
        assert "ood_generalization_failure" in FAILURE_TYPES

    def test_no_duplicate_types(self):
        assert len(FAILURE_TYPES) == len(set(FAILURE_TYPES))


# ===========================================================================
# 7. MemoryEntry
# ===========================================================================

class TestMemoryEntry:

    @pytest.fixture
    def entry(self) -> MemoryEntry:
        return MemoryEntry(
            entry_id="mem-001",
            graph_fingerprint="graph-abc",
            topology_family="radial",
            projection_iterations=5,
            initial_residual=0.1,
            final_residual=1e-8,
            dominant_failure=None,
            oracle_time_ms=50.0,
            projection_time_ms=12.5,
            used_lora_expert=None,
        )

    def test_fingerprint_deterministic(self, entry):
        assert entry.fingerprint == entry.fingerprint

    def test_serialization_roundtrip(self, entry):
        d = entry.to_json_dict()
        restored = MemoryEntry.from_json_dict(d)
        assert restored.entry_id == entry.entry_id
        assert restored.fingerprint == entry.fingerprint

    def test_validation_valid(self, entry):
        assert entry.validate() == []

    def test_validation_invalid_failure(self):
        bad = MemoryEntry(
            entry_id="mem-bad", graph_fingerprint="x", topology_family="radial",
            projection_iterations=1, initial_residual=0.0, final_residual=0.0,
            dominant_failure="not_a_real_failure", oracle_time_ms=0.0,
            projection_time_ms=0.0, used_lora_expert=None,
        )
        errors = bad.validate()
        assert any("dominant_failure" in e for e in errors)

    def test_frozen(self, entry):
        with pytest.raises(AttributeError):
            entry.entry_id = "changed"  # type: ignore


# ===========================================================================
# 8. Cross-contract: from_circuit_graph
# ===========================================================================

class TestFromCircuitGraph:

    def test_conversion_basic(self):
        """Convert a mock CircuitGraph to CanonicalCircuitGraph."""
        class MockCircuitGraph:
            node_features = torch.tensor([[1.0, 0.5]], dtype=torch.float32)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_features = torch.tensor([[0.1]], dtype=torch.float32)
            fingerprint = "mock-fp-1234567890"

        mock = MockCircuitGraph()
        canonical = from_circuit_graph(mock, cycle_count=0, connected_components=1)
        assert canonical.num_nodes == 1
        assert canonical.num_edges == 1
        assert canonical.topology_family == TopologyFamily.RADIAL
        assert len(canonical.fingerprint) == 64

    def test_validate_graph_raises_on_invalid(self):
        with pytest.raises(ValueError):
            g = CanonicalCircuitGraph(
                graph_id="bad", fingerprint="x", num_nodes=0, num_edges=0,
                node_features=torch.zeros(0, 1), edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_features=torch.zeros(0, 1), topology_family=TopologyFamily.UNKNOWN,
                cycle_count=0, connected_components=1, source_nodes=[], ground_node=0,
            )
            validate_graph(g)


# ===========================================================================
# 9. Determinism across sessions
# ===========================================================================

class TestDeterminism:

    def test_graph_fingerprint_from_dict_deterministic(self):
        data = {
            "num_nodes": 3, "num_edges": 2,
            "edge_index": [[0, 1], [1, 2]],
            "topology_family": "radial",
            "cycle_count": 0, "connected_components": 1,
            "source_nodes": [0], "ground_node": 0,
        }
        fp1 = compute_graph_fingerprint_from_dict(data)
        fp2 = compute_graph_fingerprint_from_dict(data)
        assert fp1 == fp2

    def test_report_fingerprint_json_sorted(self):
        """Report fingerprint is deterministic regardless of dict insertion order."""
        r1 = EvaluationReport(
            report_id="r1", model_fingerprint="m", dataset_fingerprint="d",
            iid_mae=0.1, ood_mae=0.2, iid_kcl_max=0.0, ood_kcl_max=0.0,
            iid_kvl_max=0.0, ood_kvl_max=0.0, projection_iterations_mean=3.0,
            speedup_factor=100.0, metadata={"z_key": 1, "a_key": 2},
        )
        r2 = EvaluationReport(
            report_id="r1", model_fingerprint="m", dataset_fingerprint="d",
            iid_mae=0.1, ood_mae=0.2, iid_kcl_max=0.0, ood_kcl_max=0.0,
            iid_kvl_max=0.0, ood_kvl_max=0.0, projection_iterations_mean=3.0,
            speedup_factor=100.0, metadata={"a_key": 2, "z_key": 1},
        )
        assert r1.fingerprint == r2.fingerprint

    def test_experiment_spec_fingerprint_json_sorted(self):
        s1 = ExperimentSpec(
            experiment_id="e1", seed=42, dataset_fingerprint="d",
            checkpoint_fingerprint=None, target_mode="oracle",
            topology_curriculum=False, projection_enabled=False,
            projection_config={}, training_config={"z": 1, "a": 2},
            evaluation_config={},
        )
        s2 = ExperimentSpec(
            experiment_id="e1", seed=42, dataset_fingerprint="d",
            checkpoint_fingerprint=None, target_mode="oracle",
            topology_curriculum=False, projection_enabled=False,
            projection_config={}, training_config={"a": 2, "z": 1},
            evaluation_config={},
        )
        assert s1.fingerprint == s2.fingerprint
