"""Projection effort metrics for CPT v2.10.

First-class benchmark metric that quantifies how much work the physics
projection layer needs to do to bring a surrogate's predictions onto
the physical manifold. Lower effort = better surrogate.

Key metric: if a surrogate starts closer to the manifold, projection
requires fewer iterations, exhibits faster residual decay, and
applies smaller corrections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch

from backend.circuits.graph_dataset import CircuitGraph
from backend.circuits.models import Circuit
from backend.circuits.physics_projection import (
    PhysicsProjection,
    ProjectionConfig,
    _node_kcl_residual,
)


@dataclass(frozen=True)
class ProjectionEffort:
    """Immutable summary of projection workload for a single circuit.

    Attributes:
        iterations_to_converge: Number of projection steps until residual < tolerance.
            Equals max_steps if convergence was not reached.
        residual_decay_rate: Geometric mean of residual[i+1]/residual[i] across
            consecutive steps. Lower = faster convergence. NaN if < 2 steps.
        initial_residual: Max |KCL residual| before any projection step.
        final_residual: Max |KCL residual| after all projection steps.
        correction_distance: L2 norm of (V_projected - V_initial) / sqrt(N),
            i.e. per-node RMS correction applied by projection.
    """

    iterations_to_converge: int
    residual_decay_rate: float
    initial_residual: float
    final_residual: float
    correction_distance: float

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "iterations_to_converge": self.iterations_to_converge,
            "residual_decay_rate": round(self.residual_decay_rate, 9),
            "initial_residual": round(self.initial_residual, 12),
            "final_residual": round(self.final_residual, 12),
            "correction_distance": round(self.correction_distance, 9),
        }


def measure_projection_effort(
    initial_voltages: torch.Tensor,
    circuit_graph: CircuitGraph,
    circuit: Circuit,
    projection_config: ProjectionConfig | None = None,
    tolerance: float = 1e-9,
) -> ProjectionEffort:
    """Measure projection effort starting from initial_voltages.

    This applies the full physics projection pipeline (KCL + virtual node +
    KVL + power) while recording per-step residuals, then computes the
    effort metrics.

    Args:
        initial_voltages: Surrogate-predicted voltages (num_nodes,) tensor.
        circuit_graph: CircuitGraph with adjacency and cycle info.
        circuit: Circuit domain model.
        projection_config: Configuration for projection. If None, uses default
            with virtual_node_enabled=True, steps=50, tolerance=1e-9.
        tolerance: Residual threshold for convergence detection.

    Returns:
        ProjectionEffort dataclass with all metrics.
    """
    if projection_config is None:
        projection_config = ProjectionConfig(
            steps=50,
            alpha_kcl=0.1,
            alpha_kvl=0.05,
            alpha_power=0.05,
            virtual_node_enabled=True,
            virtual_conductance=1.0,
            blend_factor=0.5,
            clamp_value=1e4,
        )

    projector = PhysicsProjection(projection_config)

    # Compute initial residual before any projection
    initial_res = _node_kcl_residual(initial_voltages, circuit_graph, circuit)
    initial_max_res = initial_res.abs().max().item() if initial_res.numel() > 0 else 0.0

    # Run projection with step-by-step metrics
    step_metrics = projector.project_step_metrics(
        circuit_graph, circuit, initial_voltages
    )

    # Final projected voltages
    projected_v = projector.project(circuit_graph, circuit, initial_voltages)

    # Compute final residual
    final_res = _node_kcl_residual(projected_v, circuit_graph, circuit)
    final_max_res = final_res.abs().max().item() if final_res.numel() > 0 else 0.0

    # Determine convergence iteration
    iterations_to_converge = len(step_metrics)
    for i, m in enumerate(step_metrics):
        if m["kcl_max_residual"] < tolerance:
            iterations_to_converge = i + 1
            break

    # Compute residual decay rate (geometric mean of consecutive ratios)
    decay_rate = float("nan")
    if len(step_metrics) >= 2:
        ratios: List[float] = []
        for i in range(len(step_metrics) - 1):
            r_curr = step_metrics[i]["kcl_max_residual"]
            r_next = step_metrics[i + 1]["kcl_max_residual"]
            if r_curr > 0.0:
                ratio = r_next / r_curr
                # Clamp to avoid extreme values from near-zero residuals
                ratio = max(ratio, 1e-15)
                ratios.append(ratio)
        if ratios:
            # Geometric mean
            log_sum = sum(math.log(r) for r in ratios if r > 0)
            decay_rate = math.exp(log_sum / len(ratios))

    # Correction distance: ||V_projected - V_initial||_2 / sqrt(N)
    n_nodes = initial_voltages.numel()
    correction_dist = (projected_v - initial_voltages).norm().item() / max(
        math.sqrt(n_nodes), 1.0
    )

    return ProjectionEffort(
        iterations_to_converge=iterations_to_converge,
        residual_decay_rate=decay_rate,
        initial_residual=initial_max_res,
        final_residual=final_max_res,
        correction_distance=correction_dist,
    )


def compute_projection_effort(
    model: torch.nn.Module,
    eval_graphs: list,
    eval_circuits: list,
    use_edge_features: bool = True,
    ablation: str = "full",
) -> dict:
    """Convenience wrapper: compute aggregate projection effort for a model.

    Runs the model on each eval circuit, measures projection effort, and
    returns aggregate stats.  Used by run_circuit_arena.py --checkpoints.
    """
    from backend.circuits.surrogate_eval import denormalize_voltages, get_vmax

    efforts: list[ProjectionEffort] = []
    residuals_after_1: list[float] = []
    raw_kcl: list[float] = []
    raw_kvl: list[float] = []

    projection_config = ProjectionConfig(
        steps=50,
        alpha_kcl=0.1,
        alpha_kvl=0.05,
        alpha_power=0.05,
        virtual_node_enabled=True,
        virtual_conductance=1.0,
        blend_factor=0.5,
        clamp_value=1e4,
    )
    projector = PhysicsProjection(projection_config)

    for graph, circuit in zip(eval_graphs, eval_circuits):
        vmax = get_vmax(circuit)
        with torch.no_grad():
            raw_pred = model(
                graph.node_features,
                graph.edge_index,
                graph.edge_features if use_edge_features and graph.edge_features is not None else None,
            )
            voltages = denormalize_voltages(raw_pred.squeeze(-1), vmax)

        effort = measure_projection_effort(voltages, graph, circuit, projection_config)
        efforts.append(effort)

        # Residual after 1 step
        step_metrics = projector.project_step_metrics(graph, circuit, voltages)
        if step_metrics:
            residuals_after_1.append(step_metrics[0]["kcl_max_residual"])
        else:
            residuals_after_1.append(0.0)

        # Raw KCL/KVL violations (before projection)
        raw_kcl.append(effort.initial_residual)
        from backend.circuits.physics_projection import _cycle_kvl_residual
        if graph.cycle_matrix.numel() > 0:
            kvl_res = _cycle_kvl_residual(voltages, graph)
            raw_kvl.append(kvl_res.abs().max().item() if kvl_res.numel() > 0 else 0.0)
        else:
            raw_kvl.append(0.0)

    agg = aggregate_effort(efforts)
    agg["mean_residual_after_1_step"] = sum(residuals_after_1) / max(len(residuals_after_1), 1)
    agg["mean_raw_kcl_violation"] = sum(raw_kcl) / max(len(raw_kcl), 1)
    agg["mean_raw_kvl_violation"] = sum(raw_kvl) / max(len(raw_kvl), 1)
    return agg


def aggregate_effort(
    efforts: List[ProjectionEffort],
) -> Dict[str, float]:
    """Aggregate projection effort metrics across many circuits.

    Returns dict with mean, median, and percentiles for each metric.
    """
    if not efforts:
        return {
            "mean_iterations": 0.0,
            "median_iterations": 0.0,
            "p90_iterations": 0.0,
            "mean_decay_rate": float("nan"),
            "mean_initial_residual": 0.0,
            "mean_final_residual": 0.0,
            "mean_correction_distance": 0.0,
            "count": 0,
        }

    iters = sorted([e.iterations_to_converge for e in efforts])
    decay_rates = [e.residual_decay_rate for e in efforts if not math.isnan(e.residual_decay_rate)]
    init_res = [e.initial_residual for e in efforts]
    final_res = [e.final_residual for e in efforts]
    corr_dist = [e.correction_distance for e in efforts]

    n = len(efforts)
    p90_idx = min(int(n * 0.9), n - 1)

    return {
        "mean_iterations": sum(iters) / n,
        "median_iterations": iters[n // 2],
        "p90_iterations": iters[p90_idx],
        "mean_decay_rate": sum(decay_rates) / len(decay_rates) if decay_rates else float("nan"),
        "mean_initial_residual": sum(init_res) / n,
        "mean_final_residual": sum(final_res) / n,
        "mean_correction_distance": sum(corr_dist) / n,
        "count": n,
    }
