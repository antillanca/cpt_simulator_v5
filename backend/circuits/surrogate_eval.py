"""Oracle-vs-surrogate evaluation for DC circuits.

Computes: MAE, RMSE, max voltage error, KCL/KVL violations on surrogate
predictions, replay consistency.

Supports two voltage transforms:
1. per_circuit_vmax: normalize by max voltage source value (default)
2. signed_log1p: signed log1p transform (legacy)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from backend.circuits.graph_dataset import CircuitGraph
from backend.circuits.models import Circuit, CircuitSolution
from backend.neural.models.circuit_gnn import CircuitGNN, EdgeAwareCircuitGNN


# --- Voltage transforms ---

def voltage_to_target(v: torch.Tensor) -> torch.Tensor:
    """Signed log1p transform (legacy)."""
    return torch.sign(v) * torch.log1p(v.abs())


def target_to_voltage(t: torch.Tensor) -> torch.Tensor:
    """Inverse signed log1p transform (legacy)."""
    return torch.sign(t) * torch.expm1(t.abs())


def get_vmax(circuit: Circuit) -> float:
    """Get max absolute voltage source value for normalization."""
    if circuit.voltage_sources:
        return max(abs(vs.voltage) for vs in circuit.voltage_sources)
    return 1.0


def normalize_voltages(v: torch.Tensor, vmax: float) -> torch.Tensor:
    """Normalize voltages by vmax."""
    return v / max(vmax, 1e-12)


def denormalize_voltages(norm: torch.Tensor, vmax: float) -> torch.Tensor:
    """Denormalize predictions back to voltage space."""
    return norm * vmax


# --- Evaluation result ---

@dataclass(frozen=True)
class SurrogateEvalResult:
    """Aggregate evaluation result."""

    mae: float
    rmse: float
    max_voltage_error: float
    kcl_max_violation: float
    kvl_max_violation: float
    replay_consistency: float  # max abs diff between two runs
    count: int
    per_circuit_mae: tuple[float, ...] = ()


# --- KCL/KVL violation computation ---

def _compute_kcl_violation(
    circuit: Circuit,
    predicted_voltages: Dict[str, float],
) -> float:
    """Compute max KCL violation at any node given predicted voltages.

    KCL: sum of currents leaving each node = 0.
    For resistors: I = (V_a - V_b) / R, flowing from a to b.
    For voltage sources: current is unknown from voltages alone.
    For current sources: inject I at positive, extract at negative.
    """
    max_err = 0.0
    all_nodes = list(circuit.all_nodes) + [circuit.ground_node]

    for node in sorted(all_nodes):
        current_sum = 0.0

        for r in circuit.resistors:
            va = predicted_voltages.get(r.node_a, 0.0)
            vb = predicted_voltages.get(r.node_b, 0.0)
            i_r = (va - vb) / r.resistance_ohm
            if r.node_a == node:
                current_sum -= i_r
            elif r.node_b == node:
                current_sum += i_r

        for cs in circuit.current_sources:
            if cs.positive == node:
                current_sum += cs.current
            elif cs.negative == node:
                current_sum -= cs.current

        has_vs = any(vs.positive == node or vs.negative == node for vs in circuit.voltage_sources)
        if not has_vs:
            max_err = max(max_err, abs(current_sum))

    return max_err


def _compute_kvl_violation(
    circuit: Circuit,
    predicted_voltages: Dict[str, float],
) -> float:
    """Compute max KVL violation for voltage sources.

    KVL for voltage source: V_source = V_positive - V_negative.
    """
    max_err = 0.0
    for vs in circuit.voltage_sources:
        vp = predicted_voltages.get(vs.positive, 0.0)
        vn = predicted_voltages.get(vs.negative, 0.0)
        measured = vp - vn
        err = abs(measured - vs.voltage)
        max_err = max(max_err, err)

    return max_err


# --- Surrogate evaluation ---

def evaluate_surrogate(
    model: torch.nn.Module,
    graphs: Sequence[CircuitGraph],
    circuits: Sequence[Circuit],
    use_edge_features: bool = True,
    voltage_transform: str = "per_circuit_vmax",
) -> SurrogateEvalResult:
    """Evaluate surrogate against oracle on a list of (graph, circuit) pairs.

    Returns metrics including MAE, RMSE, max error, KCL/KVL violations,
    and replay consistency.
    """
    model.eval()
    maes: list[float] = []
    rmses: list[float] = []
    max_errs: list[float] = []
    kcl_violations: list[float] = []
    kvl_violations: list[float] = []
    replay_diffs: list[float] = []

    with torch.no_grad():
        for idx, (g, circuit) in enumerate(zip(graphs, circuits)):
            if g.node_features.size(0) == 0:
                continue

            vmax = get_vmax(circuit)

            # First inference
            if use_edge_features:
                pred_norm_1 = model(g.node_features, g.edge_index, g.edge_features)
            else:
                pred_norm_1 = model(g.node_features, g.edge_index)

            # Second inference (replay consistency)
            if use_edge_features:
                pred_norm_2 = model(g.node_features, g.edge_index, g.edge_features)
            else:
                pred_norm_2 = model(g.node_features, g.edge_index)

            replay_diff = (pred_norm_1 - pred_norm_2).abs().max().item()
            replay_diffs.append(replay_diff)

            # Denormalize predictions back to voltage space
            if voltage_transform == "per_circuit_vmax":
                pred_v1 = denormalize_voltages(pred_norm_1, vmax)
            else:
                pred_v1 = target_to_voltage(pred_norm_1)

            # Compute voltage errors
            diff = pred_v1 - g.target_voltages
            mae = diff.abs().mean().item()
            rmse = diff.pow(2).mean().sqrt().item()
            max_err = diff.abs().max().item()

            maes.append(mae)
            rmses.append(rmse)
            max_errs.append(max_err)

            # Invariant violations using predicted voltages
            pred_voltage_dict = dict(zip(g.node_names, pred_v1.tolist()))
            pred_voltage_dict[circuit.ground_node] = 0.0

            kcl_v = _compute_kcl_violation(circuit, pred_voltage_dict)
            kvl_v = _compute_kvl_violation(circuit, pred_voltage_dict)
            kcl_violations.append(kcl_v)
            kvl_violations.append(kvl_v)

    n = max(len(maes), 1)
    return SurrogateEvalResult(
        mae=sum(maes) / n if maes else 0.0,
        rmse=sum(rmses) / n if rmses else 0.0,
        max_voltage_error=max(max_errs) if max_errs else 0.0,
        kcl_max_violation=max(kcl_violations) if kcl_violations else 0.0,
        kvl_max_violation=max(kvl_violations) if kvl_violations else 0.0,
        replay_consistency=max(replay_diffs) if replay_diffs else 0.0,
        count=len(maes),
        per_circuit_mae=tuple(maes),
    )


# --- Baselines ---

def evaluate_mean_baseline(
    graphs: Sequence[CircuitGraph],
    mean_voltage: float,
) -> Dict[str, float]:
    """Evaluate a mean predictor baseline: always predict mean_voltage."""
    maes: list[float] = []
    rmses: list[float] = []
    max_errs: list[float] = []

    for g in graphs:
        if g.node_features.size(0) == 0:
            continue
        pred = torch.full_like(g.target_voltages, mean_voltage)
        diff = pred - g.target_voltages
        maes.append(diff.abs().mean().item())
        rmses.append(diff.pow(2).mean().sqrt().item())
        max_errs.append(diff.abs().max().item())

    n = max(len(maes), 1)
    return {
        "mae": sum(maes) / n,
        "rmse": sum(rmses) / n,
        "max_error": max(max_errs) if max_errs else 0.0,
        "count": len(maes),
    }


def evaluate_linear_baseline(
    graphs: Sequence[CircuitGraph],
) -> Dict[str, float]:
    """Evaluate a linear baseline: predict voltage from mean of node features."""
    maes: list[float] = []
    rmses: list[float] = []
    max_errs: list[float] = []

    for g in graphs:
        if g.node_features.size(0) == 0:
            continue
        # Simple: use mean of node features as a weak signal
        pred = g.node_features.mean(dim=1) * g.target_voltages.abs().mean().item()
        diff = pred - g.target_voltages
        maes.append(diff.abs().mean().item())
        rmses.append(diff.pow(2).mean().sqrt().item())
        max_errs.append(diff.abs().max().item())

    n = max(len(maes), 1)
    return {
        "mae": sum(maes) / n,
        "rmse": sum(rmses) / n,
        "max_error": max(max_errs) if max_errs else 0.0,
        "count": len(maes),
    }
