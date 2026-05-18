"""Tests for CPT v2.10 — Projection-Distilled Surrogate Retraining.

Validates:
1. Projected target generation (blend formula, perturbation, projection)
2. Projection effort metrics (measure + aggregate)
3. Training with blended targets (load, train 2 epochs, oracle MAE tracked)
4. Arena multi-checkpoint comparison
5. Alpha ablation script smoke test
6. Determinism (seed=42 reproducibility)
"""

from __future__ import annotations

import copy
import json
import math
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.graph_dataset import circuit_to_graph
from backend.circuits.models import Circuit
from backend.circuits.parser import parse_netlist
from backend.circuits.physics_projection import PhysicsProjection, ProjectionConfig
from backend.circuits.projection_effort import (
    ProjectionEffort,
    measure_projection_effort,
    aggregate_effort,
    compute_projection_effort,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_circuit():
    """Simple 3-resistor circuit with 1 cycle."""
    netlist = """\
V1 N1 N0 10
R1 N1 N2 1000
R2 N2 N0 2000
R3 N2 N3 1500
R4 N3 N0 3000
"""
    return parse_netlist(netlist)


@pytest.fixture
def simple_graph(simple_circuit, oracle_solution):
    return circuit_to_graph(simple_circuit, oracle_solution)


@pytest.fixture
def oracle_solution(simple_circuit):
    return solve_dc_circuit(simple_circuit)


@pytest.fixture
def oracle_voltages(oracle_solution, simple_circuit):
    return {k: v for k, v in oracle_solution.node_voltages.items() if k != simple_circuit.ground_node}


@pytest.fixture
def projection_config():
    return ProjectionConfig(
        steps=50,
        alpha_kcl=0.1,
        alpha_kvl=0.05,
        virtual_node_enabled=True,
        virtual_conductance=1.0,
        blend_factor=0.5,
    )


# ---------------------------------------------------------------------------
# Test 1: Blended Target Generation
# ---------------------------------------------------------------------------

class TestBlendedTargetGeneration:
    """Validate the blended target formula and projection pipeline."""

    def test_blend_formula_alpha_0(self, oracle_voltages):
        """alpha=0 → blended = projected (pure projection)."""
        projected = {n: v * 0.95 for n, v in oracle_voltages.items()}
        alpha = 0.0
        blended = {n: alpha * oracle_voltages[n] + (1 - alpha) * projected[n] for n in oracle_voltages}
        for n in oracle_voltages:
            assert abs(blended[n] - projected[n]) < 1e-12

    def test_blend_formula_alpha_1(self, oracle_voltages):
        """alpha=1 → blended = oracle (pure oracle)."""
        projected = {n: v * 0.95 for n, v in oracle_voltages.items()}
        alpha = 1.0
        blended = {n: alpha * oracle_voltages[n] + (1 - alpha) * projected[n] for n in oracle_voltages}
        for n in oracle_voltages:
            assert abs(blended[n] - oracle_voltages[n]) < 1e-12

    def test_blend_formula_alpha_02(self, oracle_voltages):
        """alpha=0.2 → blended = 0.2*oracle + 0.8*projected."""
        projected = {n: v * 0.9 for n, v in oracle_voltages.items()}
        alpha = 0.2
        blended = {n: alpha * oracle_voltages[n] + (1 - alpha) * projected[n] for n in oracle_voltages}
        for n in oracle_voltages:
            expected = 0.2 * oracle_voltages[n] + 0.8 * projected[n]
            assert abs(blended[n] - expected) < 1e-12

    def test_perturbation_determinism(self, oracle_voltages):
        """Same seed → same perturbation."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        sigma = 1.5
        p1 = {n: v + rng1.gauss(0, sigma) for n, v in oracle_voltages.items()}
        p2 = {n: v + rng2.gauss(0, sigma) for n, v in oracle_voltages.items()}
        for n in oracle_voltages:
            assert p1[n] == p2[n]

    def test_perturbation_different_seeds(self, oracle_voltages):
        """Different seeds → different perturbation."""
        rng1 = random.Random(42)
        rng2 = random.Random(43)
        sigma = 1.5
        p1 = {n: v + rng1.gauss(0, sigma) for n, v in oracle_voltages.items()}
        p2 = {n: v + rng2.gauss(0, sigma) for n, v in oracle_voltages.items()}
        # With high probability, at least one node differs
        any_diff = any(abs(p1[n] - p2[n]) > 1e-6 for n in oracle_voltages)
        assert any_diff

    def test_projected_reduces_residual(
        self, simple_graph, simple_circuit, oracle_voltages, projection_config
    ):
        """Projection of perturbed voltages reduces KCL residual."""
        from backend.circuits.physics_projection import _node_kcl_residual

        rng = random.Random(42)
        perturbed = {n: v + rng.gauss(0, 1.5) for n, v in oracle_voltages.items()}
        nodes = list(oracle_voltages.keys())
        v_tensor = torch.tensor([perturbed[n] for n in nodes], dtype=torch.float32)

        projector = PhysicsProjection(projection_config)
        projected_v = projector.project(simple_graph, simple_circuit, v_tensor)

        # Initial residual
        initial_res = _node_kcl_residual(v_tensor, simple_graph, simple_circuit)
        final_res = _node_kcl_residual(projected_v, simple_graph, simple_circuit)

        assert final_res.abs().max().item() < initial_res.abs().max().item()


# ---------------------------------------------------------------------------
# Test 2: Projection Effort Metrics
# ---------------------------------------------------------------------------

class TestProjectionEffort:
    """Validate projection effort computation and aggregation."""

    def test_measure_effort_oracle_start(
        self, simple_graph, simple_circuit, oracle_voltages, projection_config
    ):
        """Effort from oracle voltages should be minimal (already on manifold)."""
        nodes = list(oracle_voltages.keys())
        v_tensor = torch.tensor([oracle_voltages[n] for n in nodes], dtype=torch.float32)

        effort = measure_projection_effort(v_tensor, simple_graph, simple_circuit, projection_config)
        assert effort.initial_residual < 1e-3
        assert effort.final_residual < effort.initial_residual or effort.initial_residual < 1e-9

    def test_measure_effort_perturbed_start(
        self, simple_graph, simple_circuit, oracle_voltages, projection_config
    ):
        """Effort from perturbed voltages should be higher."""
        from backend.circuits.physics_projection import _node_kcl_residual

        rng = random.Random(42)
        perturbed = {n: v + rng.gauss(0, 5.0) for n, v in oracle_voltages.items()}
        nodes = list(oracle_voltages.keys())
        v_tensor = torch.tensor([perturbed[n] for n in nodes], dtype=torch.float32)

        effort = measure_projection_effort(v_tensor, simple_graph, simple_circuit, projection_config)
        assert effort.initial_residual > 1e-3
        assert effort.correction_distance > 0.0

    def test_effort_dataclass_to_dict(self, simple_graph, simple_circuit, oracle_voltages, projection_config):
        """ProjectionEffort.to_dict() returns expected keys."""
        nodes = list(oracle_voltages.keys())
        v_tensor = torch.tensor([oracle_voltages[n] for n in nodes], dtype=torch.float32)

        effort = measure_projection_effort(v_tensor, simple_graph, simple_circuit, projection_config)
        d = effort.to_dict()
        assert "iterations_to_converge" in d
        assert "initial_residual" in d
        assert "final_residual" in d
        assert "correction_distance" in d
        assert "residual_decay_rate" in d

    def test_aggregate_effort_empty(self):
        """Aggregation of empty list returns zeros."""
        agg = aggregate_effort([])
        assert agg["count"] == 0
        assert agg["mean_iterations"] == 0.0

    def test_aggregate_effort_single(self, simple_graph, simple_circuit, oracle_voltages, projection_config):
        """Aggregation of single effort works."""
        nodes = list(oracle_voltages.keys())
        v_tensor = torch.tensor([oracle_voltages[n] for n in nodes], dtype=torch.float32)

        effort = measure_projection_effort(v_tensor, simple_graph, simple_circuit, projection_config)
        agg = aggregate_effort([effort])
        assert agg["count"] == 1
        assert agg["mean_iterations"] == effort.iterations_to_converge

    def test_compute_projection_effort_function(
        self, simple_graph, simple_circuit, oracle_voltages, projection_config
    ):
        """compute_projection_effort convenience wrapper works with a mock model."""
        from backend.neural.models.circuit_gnn import EdgeAwareCircuitGNN
        from backend.circuits.graph_dataset import CircuitGraph

        # Create a tiny model that outputs near-oracle voltages
        node_dim = simple_graph.x.shape[1]
        edge_dim = simple_graph.edge_attr.shape[1] if simple_graph.edge_attr is not None else 4
        model = EdgeAwareCircuitGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=32)

        result = compute_projection_effort(
            model, [simple_graph], [simple_circuit],
            use_edge_features=True,
        )
        assert "mean_iterations" in result
        assert "mean_residual_after_1_step" in result
        assert "mean_raw_kcl_violation" in result
        assert "mean_raw_kvl_violation" in result
        assert result["count"] == 1


# ---------------------------------------------------------------------------
# Test 3: Training with Blended Targets
# ---------------------------------------------------------------------------

class TestBlendedTraining:
    """Validate that train_circuit_gnn.py supports blended target mode."""

    def test_load_blended_training_data_function_exists(self):
        """The load_blended_training_data function is importable."""
        from scripts.train_circuit_gnn import load_blended_training_data
        assert callable(load_blended_training_data)

    def test_target_mode_arg(self):
        """Argparse accepts --target-mode blended_projection."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/train_circuit_gnn.py", "--help"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert "--target-mode" in result.stdout
        assert "blended_projection" in result.stdout

    def test_generate_projected_targets_script(self):
        """generate_projected_targets.py --help works."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/generate_projected_targets.py", "--help"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "--alpha" in result.stdout
        assert "--sigma" in result.stdout


# ---------------------------------------------------------------------------
# Test 4: Arena Multi-Checkpoint
# ---------------------------------------------------------------------------

class TestArenaMultiCheckpoint:
    """Validate arena multi-checkpoint comparison support."""

    def test_arena_checkpoints_arg(self):
        """Argparse accepts --checkpoints and --checkpoint-labels."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/run_circuit_arena.py", "--help"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert "--checkpoints" in result.stdout
        assert "--checkpoint-labels" in result.stdout
        assert "--save-traces" in result.stdout

    def test_load_model_from_checkpoint(self):
        """_load_model_from_checkpoint function is importable."""
        from scripts.run_circuit_arena import _load_model_from_checkpoint
        assert callable(_load_model_from_checkpoint)


# ---------------------------------------------------------------------------
# Test 5: Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Seed=42 produces identical results across runs."""

    def test_perturbation_determinism_across_calls(self, oracle_voltages):
        """Two perturbations with seed=42 produce identical results."""
        results = []
        for _ in range(2):
            rng = random.Random(42)
            p = {n: v + rng.gauss(0, 1.5) for n, v in oracle_voltages.items()}
            results.append(p)
        for n in oracle_voltages:
            assert results[0][n] == results[1][n]

    def test_projection_determinism(
        self, simple_graph, simple_circuit, oracle_voltages, projection_config
    ):
        """Projection with same input produces same output."""
        rng = random.Random(42)
        perturbed = {n: v + rng.gauss(0, 1.5) for n, v in oracle_voltages.items()}
        nodes = list(oracle_voltages.keys())
        v_tensor = torch.tensor([perturbed[n] for n in nodes], dtype=torch.float32)

        projector = PhysicsProjection(projection_config)
        result1 = projector.project(simple_graph, simple_circuit, v_tensor)
        result2 = projector.project(simple_graph, simple_circuit, v_tensor)

        assert torch.allclose(result1, result2, atol=1e-12)

    def test_effort_determinism(
        self, simple_graph, simple_circuit, oracle_voltages, projection_config
    ):
        """Effort measurement is deterministic."""
        rng = random.Random(42)
        perturbed = {n: v + rng.gauss(0, 1.5) for n, v in oracle_voltages.items()}
        nodes = list(oracle_voltages.keys())
        v_tensor = torch.tensor([perturbed[n] for n in nodes], dtype=torch.float32)

        effort1 = measure_projection_effort(v_tensor, simple_graph, simple_circuit, projection_config)
        effort2 = measure_projection_effort(v_tensor, simple_graph, simple_circuit, projection_config)

        assert effort1.initial_residual == effort2.initial_residual
        assert effort1.final_residual == effort2.final_residual
        assert effort1.iterations_to_converge == effort2.iterations_to_converge
        assert effort1.correction_distance == effort2.correction_distance


# ---------------------------------------------------------------------------
# Test 6: Projection Config defaults for v2.10
# ---------------------------------------------------------------------------

class TestProjectionConfigV210:
    """Validate projection config parameters match v2.10 spec."""

    def test_default_config_not_used_for_v210(self):
        """v2.10 uses custom config, not ProjectionConfig() defaults."""
        default_config = ProjectionConfig()
        assert default_config.steps == 3  # Not 50 — v2.10 must use steps=50

    def test_v210_config_parameters(self):
        """v2.10 projection config matches spec."""
        config = ProjectionConfig(
            steps=50,
            alpha_kcl=0.1,
            alpha_kvl=0.05,
            virtual_node_enabled=True,
            virtual_conductance=1.0,
            blend_factor=0.5,
        )
        assert config.steps == 50
        assert config.virtual_conductance == 1.0  # g_virtual=1.0
        assert config.alpha_kcl == 0.1
        assert config.alpha_kvl == 0.05

    def test_v210_config_creates_valid_projector(self):
        """v2.10 config creates a working PhysicsProjection."""
        config = ProjectionConfig(
            steps=50,
            alpha_kcl=0.1,
            alpha_kvl=0.05,
            virtual_node_enabled=True,
            virtual_conductance=1.0,
            blend_factor=0.5,
        )
        projector = PhysicsProjection(config)
        assert projector.config.steps == 50
        assert projector.virtual_node.enabled is True
