"""Tests for CPT v2.9A — Surrogate Feasibility (Graph Neural Surrogate).

Covers:
- Graph conversion determinism
- Model forward pass determinism
- Training reproducibility
- Evaluation metrics correctness
- Invariant computation on surrogate predictions
- OOD generation reproducibility
- Baseline comparisons
- Speed benchmark basics
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.graph_dataset import CircuitGraph, circuit_to_graph
from backend.circuits.invariants import validate_invariants
from backend.circuits.models import Circuit, CurrentSource, Resistor, VoltageSource
from backend.circuits.ood_generator import generate_ood_circuits
from backend.circuits.parser import parse_netlist
from backend.circuits.surrogate_eval import (
    evaluate_linear_baseline,
    evaluate_mean_baseline,
    evaluate_surrogate,
    target_to_voltage,
    voltage_to_target,
)
from backend.circuits.traces import generate_oracle_trace
from backend.neural.models.circuit_gnn import (
    CircuitGNN,
    EdgeAwareCircuitGNN,
    EdgeConditionedConv,
    ManualGCNConv,
)


# ---- Fixtures ----

@pytest.fixture
def voltage_divider() -> Circuit:
    netlist = "V1 VIN 0 5\nR1 VIN N1 1000\nR2 N1 0 2000\n"
    return parse_netlist(netlist, name="vdivider")


@pytest.fixture
def voltage_divider_solution(voltage_divider):
    return solve_dc_circuit(voltage_divider)


@pytest.fixture
def voltage_divider_graph(voltage_divider, voltage_divider_solution):
    return circuit_to_graph(voltage_divider, voltage_divider_solution)


@pytest.fixture
def wheatstone() -> Circuit:
    netlist = (
        "# Wheatstone bridge\n"
        "V1 N1 0 10\n"
        "R1 N1 N2 100\n"
        "R2 N1 N3 200\n"
        "R3 N2 N4 200\n"
        "R4 N3 N4 100\n"
        "R5 N4 0 500\n"
    )
    return parse_netlist(netlist, name="wheatstone")


@pytest.fixture
def wheatstone_solution(wheatstone):
    return solve_dc_circuit(wheatstone)


@pytest.fixture
def wheatstone_graph(wheatstone, wheatstone_solution):
    return circuit_to_graph(wheatstone, wheatstone_solution)


# ---- PHASE 1: Graph Conversion ----


class TestGraphConversion:
    """Test circuit-to-graph conversion."""

    def test_graph_has_correct_node_count(self, voltage_divider_graph):
        """Voltage divider: N1, VIN (ground excluded)."""
        assert voltage_divider_graph.node_features.size(0) == 2

    def test_graph_has_correct_edge_count(self, voltage_divider_graph):
        """3 components × 2 directions = 6 edges."""
        assert voltage_divider_graph.edge_index.size(1) == 6

    def test_graph_node_features_dim(self, voltage_divider_graph):
        """Node features must be 8-dimensional."""
        assert voltage_divider_graph.node_features.size(1) == 8

    def test_graph_edge_features_dim(self, voltage_divider_graph):
        """Edge features must be 4-dimensional."""
        assert voltage_divider_graph.edge_features.size(1) == 4

    def test_target_voltages_match_oracle(self, voltage_divider_graph, voltage_divider_solution):
        """Target voltages in graph must match oracle solution (sorted nodes)."""
        oracle_v = [voltage_divider_solution.node_voltages[n] for n in voltage_divider_graph.node_names]
        torch.testing.assert_close(
            voltage_divider_graph.target_voltages,
            torch.tensor(oracle_v, dtype=torch.float32),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_graph_determinism(self, voltage_divider, voltage_divider_solution):
        """Same circuit → same graph fingerprint."""
        g1 = circuit_to_graph(voltage_divider, voltage_divider_solution)
        g2 = circuit_to_graph(voltage_divider, voltage_divider_solution)
        assert g1.fingerprint == g2.fingerprint

    def test_graph_features_are_finite(self, voltage_divider_graph):
        """No NaN or Inf in features."""
        assert torch.isfinite(voltage_divider_graph.node_features).all()
        assert torch.isfinite(voltage_divider_graph.edge_features).all()
        assert torch.isfinite(voltage_divider_graph.target_voltages).all()

    def test_node_names_sorted(self, voltage_divider_graph):
        """Node names must be alphabetically sorted."""
        assert voltage_divider_graph.node_names == tuple(sorted(voltage_divider_graph.node_names))

    def test_wheatstone_node_count(self, wheatstone_graph):
        """Wheatstone: N1, N2, N3, N4 (4 non-ground nodes)."""
        assert wheatstone_graph.node_features.size(0) == 4

    def test_ground_node_voltage_is_zero(self, voltage_divider_solution):
        """Ground node voltage must be 0."""
        assert voltage_divider_solution.node_voltages.get("0", 0.0) == 0.0


# ---- PHASE 2: GNN Model ----


class TestGNNModel:
    """Test GNN model architecture and forward pass."""

    def test_basic_gnn_forward(self):
        """Basic CircuitGNN produces correct output shape."""
        model = CircuitGNN(node_dim=8, edge_dim=4, hidden_dim=32)
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        out = model(x, edge_index)
        assert out.shape == (5,)

    def test_edge_aware_gnn_forward(self):
        """EdgeAwareCircuitGNN produces correct output shape."""
        model = EdgeAwareCircuitGNN(node_dim=8, edge_dim=4, hidden_dim=32)
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_feat = torch.randn(4, 4)
        out = model(x, edge_index, edge_feat)
        assert out.shape == (5,)

    def test_model_parameter_count_under_250k(self):
        """EdgeAwareCircuitGNN with hidden_dim=96 must be < 250k params."""
        model = EdgeAwareCircuitGNN(node_dim=8, edge_dim=4, hidden_dim=96)
        assert model.count_parameters() < 250_000

    def test_forward_determinism(self):
        """Same input → same output (deterministic inference)."""
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model = EdgeAwareCircuitGNN(node_dim=8, edge_dim=4, hidden_dim=32)
        model.eval()
        x = torch.randn(4, 8)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_feat = torch.randn(3, 4)
        with torch.no_grad():
            out1 = model(x, edge_index, edge_feat)
            out2 = model(x, edge_index, edge_feat)
        torch.testing.assert_close(out1, out2, atol=0, rtol=0)

    def test_manual_gcn_conv(self):
        """ManualGCNConv basic forward pass."""
        conv = ManualGCNConv(8, 16)
        x = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out = conv(x, edge_index)
        assert out.shape == (3, 16)

    def test_edge_conditioned_conv(self):
        """EdgeConditionedConv basic forward pass."""
        conv = EdgeConditionedConv(16, 4, 16)
        x = torch.randn(3, 16)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_feat = torch.randn(2, 4)
        out = conv(x, edge_index, edge_feat)
        assert out.shape == (3, 16)


# ---- PHASE 3: Voltage Transform ----


class TestVoltageTransform:
    """Test signed log1p voltage transform."""

    def test_roundtrip_zero(self):
        """Zero voltage roundtrips."""
        v = torch.tensor([0.0])
        t = voltage_to_target(v)
        v2 = target_to_voltage(t)
        torch.testing.assert_close(v2, v, atol=1e-6, rtol=1e-6)

    def test_roundtrip_positive(self):
        """Positive voltage roundtrips."""
        v = torch.tensor([5.0, 12.0, 100.0])
        t = voltage_to_target(v)
        v2 = target_to_voltage(t)
        torch.testing.assert_close(v2, v, atol=1e-5, rtol=1e-5)

    def test_roundtrip_negative(self):
        """Negative voltage roundtrips."""
        v = torch.tensor([-5.0, -12.0])
        t = voltage_to_target(v)
        v2 = target_to_voltage(t)
        torch.testing.assert_close(v2, v, atol=1e-5, rtol=1e-5)

    def test_roundtrip_small(self):
        """Small voltage roundtrips."""
        v = torch.tensor([0.001, 0.0001])
        t = voltage_to_target(v)
        v2 = target_to_voltage(t)
        torch.testing.assert_close(v2, v, atol=1e-6, rtol=1e-6)

    def test_compresses_range(self):
        """Transform compresses large voltage range."""
        v = torch.tensor([0.0, 1.0, 10.0, 100.0, 1000.0])
        t = voltage_to_target(v)
        # log1p(1000) = ~6.9, much smaller than 1000
        assert t.max().item() < 10.0
        assert t.max().item() < v.max().item()


# ---- PHASE 4: Surrogate Evaluation ----


class TestSurrogateEvaluation:
    """Test evaluation pipeline."""

    def test_mean_baseline_runs(self, voltage_divider_graph):
        """Mean baseline produces valid metrics."""
        result = evaluate_mean_baseline([voltage_divider_graph], mean_voltage=5.0)
        assert "mae" in result
        assert result["mae"] > 0  # non-zero because not all nodes are 5V

    def test_linear_baseline_runs(self, voltage_divider_graph):
        """Linear baseline produces valid metrics."""
        result = evaluate_linear_baseline([voltage_divider_graph])
        assert "mae" in result
        assert result["mae"] >= 0

    def test_evaluate_surrogate_runs(self, voltage_divider, voltage_divider_graph):
        """evaluate_surrogate runs without error."""
        random.seed(42)
        torch.manual_seed(42)
        model = EdgeAwareCircuitGNN(node_dim=8, edge_dim=4, hidden_dim=32)
        model.eval()
        result = evaluate_surrogate(
            model, [voltage_divider_graph], [voltage_divider], use_edge_features=True
        )
        assert result.count == 1
        assert result.mae >= 0
        assert result.replay_consistency >= 0

    def test_replay_consistency_is_zero(self, voltage_divider, voltage_divider_graph):
        """Replay consistency must be exactly 0 for deterministic model."""
        random.seed(42)
        torch.manual_seed(42)
        model = EdgeAwareCircuitGNN(node_dim=8, edge_dim=4, hidden_dim=32)
        model.eval()
        result = evaluate_surrogate(
            model, [voltage_divider_graph], [voltage_divider], use_edge_features=True
        )
        assert result.replay_consistency == 0.0


# ---- PHASE 5: OOD Generation ----


class TestOODGeneration:
    """Test OOD circuit generation."""

    def test_ood_reproducibility(self):
        """Same seed → same OOD circuits."""
        rows1 = generate_ood_circuits(seed=123, num_circuits=10)
        rows2 = generate_ood_circuits(seed=123, num_circuits=10)
        ids1 = [r["id"] for r in rows1]
        ids2 = [r["id"] for r in rows2]
        assert ids1 == ids2

    def test_ood_different_seeds(self):
        """Different seeds → different OOD circuits."""
        rows1 = generate_ood_circuits(seed=123, num_circuits=10)
        rows2 = generate_ood_circuits(seed=456, num_circuits=10)
        ids1 = [r["id"] for r in rows1]
        ids2 = [r["id"] for r in rows2]
        assert ids1 != ids2

    def test_ood_has_ood_type(self):
        """OOD rows have ood_type field."""
        rows = generate_ood_circuits(seed=123, num_circuits=10)
        for row in rows:
            assert "ood_type" in row

    def test_ood_circuits_solve(self):
        """All OOD circuits must be solvable."""
        rows = generate_ood_circuits(seed=123, num_circuits=20)
        for row in rows:
            circuit = parse_netlist(row["netlist"], name=row["circuit_name"])
            solution = solve_dc_circuit(circuit)
            assert len(solution.node_voltages) > 0

    def test_ood_jsonl_output(self):
        """OOD JSONL file is valid JSON."""
        from backend.circuits.ood_generator import generate_ood_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_ood_jsonl(seed=123, num_circuits=5, output_path=f"{tmpdir}/ood.jsonl")
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 5
            for line in lines:
                row = json.loads(line)
                assert "id" in row
                assert "netlist" in row


# ---- PHASE 6: End-to-End Graph Pipeline ----


class TestEndToEndPipeline:
    """Test full pipeline: netlist → graph → model → prediction."""

    def test_pipeline_produces_finite_predictions(self, voltage_divider_graph):
        """Full pipeline produces finite predictions."""
        random.seed(42)
        torch.manual_seed(42)
        model = EdgeAwareCircuitGNN(node_dim=8, edge_dim=4, hidden_dim=32)
        model.eval()
        g = voltage_divider_graph
        with torch.no_grad():
            pred = model(g.node_features, g.edge_index, g.edge_features)
        pred_v = target_to_voltage(pred)
        assert torch.isfinite(pred_v).all()

    def test_pipeline_wheatstone(self, wheatstone_graph):
        """Pipeline works on Wheatstone bridge (4 nodes)."""
        random.seed(42)
        torch.manual_seed(42)
        model = EdgeAwareCircuitGNN(node_dim=8, edge_dim=4, hidden_dim=32)
        model.eval()
        g = wheatstone_graph
        with torch.no_grad():
            pred = model(g.node_features, g.edge_index, g.edge_features)
        pred_v = target_to_voltage(pred)
        assert pred_v.shape == (4,)
        assert torch.isfinite(pred_v).all()


# ---- PHASE 7: KCL/KVL on Surrogate Predictions ----


class TestInvariantComputation:
    """Test invariant computation on surrogate predictions."""

    def test_kvl_check_on_voltage_divider(self, voltage_divider, voltage_divider_graph):
        """KVL violation computation returns a finite value."""
        random.seed(42)
        torch.manual_seed(42)
        model = EdgeAwareCircuitGNN(node_dim=8, edge_dim=4, hidden_dim=32)
        model.eval()
        g = voltage_divider_graph
        with torch.no_grad():
            pred = model(g.node_features, g.edge_index, g.edge_features)
        pred_v = target_to_voltage(pred)

        # Build voltage dict
        voltage_dict = dict(zip(g.node_names, pred_v.tolist()))
        voltage_dict["0"] = 0.0

        from backend.circuits.surrogate_eval import _compute_kvl_violation
        kvl_err = _compute_kvl_violation(voltage_divider, voltage_dict)
        assert isinstance(kvl_err, float)
        assert kvl_err >= 0

    def test_kvl_zero_on_oracle(self, voltage_divider, voltage_divider_solution):
        """KVL violation is zero when using oracle voltages."""
        from backend.circuits.surrogate_eval import _compute_kvl_violation
        kvl_err = _compute_kvl_violation(voltage_divider, voltage_divider_solution.node_voltages)
        assert kvl_err < 1e-9

    def test_kcl_zero_on_oracle_resistive(self):
        """KCL violation is ~zero on oracle solution for resistive circuit."""
        from backend.circuits.surrogate_eval import _compute_kcl_violation
        netlist = "V1 N1 0 10\nR1 N1 N2 1000\nR2 N2 0 1000\n"
        circuit = parse_netlist(netlist)
        solution = solve_dc_circuit(circuit)
        kcl_err = _compute_kcl_violation(circuit, solution.node_voltages)
        assert kcl_err < 1e-6
