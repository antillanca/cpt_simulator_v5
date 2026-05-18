"""Deterministic failure taxonomy for circuit surrogate validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

from backend.circuits.graph_dataset import CircuitGraph
from backend.circuits.models import Circuit

FAILURE_TYPES = [
    "topology_collapse",
    "extreme_resistance_instability",
    "disconnected_graph_confusion",
    "symmetry_failure",
    "node_aliasing",
    "conservation_drift",
    "ood_generalization_failure",
    "cycle_drift_failure",
    "dense_mesh_leakage",
    "bridge_node_instability",
]


def _is_connected(circuit: Circuit) -> bool:
    ground = circuit.ground_node
    adj: dict[str, set[str]] = {}
    nodes = set(circuit.all_nodes) | {ground}
    for node in nodes:
        adj[node] = set()
    for r in circuit.resistors:
        adj.setdefault(r.node_a, set()).add(r.node_b)
        adj.setdefault(r.node_b, set()).add(r.node_a)
    for vs in circuit.voltage_sources:
        adj.setdefault(vs.positive, set()).add(vs.negative)
        adj.setdefault(vs.negative, set()).add(vs.positive)
    for cs in circuit.current_sources:
        adj.setdefault(cs.positive, set()).add(cs.negative)
        adj.setdefault(cs.negative, set()).add(cs.positive)
    visited = {ground}
    queue = [ground]
    while queue:
        node = queue.pop(0)
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return all(node in visited for node in circuit.all_nodes)


def _stable_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _circuit_resistance_stats(circuit: Circuit) -> tuple[float, float]:
    if not circuit.resistors:
        return 0.0, 0.0
    resistances = [float(r.resistance_ohm) for r in circuit.resistors]
    return max(resistances), min(resistances)


def _voltage_at(node: str, node_names: Sequence[str], voltages: torch.Tensor) -> torch.Tensor:
    if node not in node_names:
        return voltages.new_zeros(())
    return voltages[node_names.index(node)]


def _component_current_leaving(node: str, circuit: Circuit, node_names: Sequence[str], voltages: torch.Tensor) -> torch.Tensor:
    current = voltages.new_zeros(())
    for resistor in circuit.resistors:
        if resistor.node_a == node:
            va = _voltage_at(resistor.node_a, node_names, voltages)
            vb = _voltage_at(resistor.node_b, node_names, voltages)
            current = current + (va - vb) / resistor.resistance_ohm
        elif resistor.node_b == node:
            va = _voltage_at(resistor.node_a, node_names, voltages)
            vb = _voltage_at(resistor.node_b, node_names, voltages)
            current = current + (vb - va) / resistor.resistance_ohm
    for source in circuit.current_sources:
        if source.positive == node:
            current = current + source.current
        elif source.negative == node:
            current = current - source.current
    return current


def compute_invariant_violations(
    circuit: Circuit,
    node_names: Sequence[str],
    predicted_voltages: torch.Tensor | Sequence[float],
) -> dict[str, Any]:
    voltages = torch.as_tensor(predicted_voltages, dtype=torch.float32).detach().cpu()
    node_names = tuple(node_names)
    kcl_max = 0.0
    kvl_max = 0.0
    power_delivered = 0.0
    power_dissipated = 0.0

    for node in sorted(set(node_names)):
        if node == circuit.ground_node:
            continue
        net = _component_current_leaving(node, circuit, node_names, voltages)
        has_voltage_source = any(vs.positive == node or vs.negative == node for vs in circuit.voltage_sources)
        if not has_voltage_source:
            kcl_max = max(kcl_max, abs(float(net.item())))

    inferred_vs_currents: dict[str, float] = {}
    for source in circuit.voltage_sources:
        leaving = _component_current_leaving(source.positive, circuit, node_names, voltages)
        inferred = -float(leaving.item())
        inferred_vs_currents[source.name] = inferred
        vp = _voltage_at(source.positive, node_names, voltages)
        vn = _voltage_at(source.negative, node_names, voltages)
        kvl_max = max(kvl_max, abs(float((vp - vn - source.voltage).item())))
        power_delivered += float((vn - vp).item()) * inferred

    for source in circuit.current_sources:
        vp = _voltage_at(source.positive, node_names, voltages)
        vn = _voltage_at(source.negative, node_names, voltages)
        power_delivered += float((vp - vn).item()) * source.current

    for resistor in circuit.resistors:
        va = _voltage_at(resistor.node_a, node_names, voltages)
        vb = _voltage_at(resistor.node_b, node_names, voltages)
        current = (va - vb) / resistor.resistance_ohm
        power_dissipated += float(((va - vb) * current).item())

    power_violation = abs(power_delivered - power_dissipated)
    return {
        "kcl_max_violation": kcl_max,
        "kvl_max_violation": kvl_max,
        "power_conservation_violation": power_violation,
        "power_delivered": power_delivered,
        "power_dissipated": power_dissipated,
        "inferred_voltage_source_currents": dict(sorted(inferred_vs_currents.items())),
    }


def _unique_ratio(values: torch.Tensor, *, digits: int = 6) -> float:
    if values.numel() == 0:
        return 1.0
    rounded = [round(float(v), digits) for v in values.tolist()]
    return len(set(rounded)) / max(len(rounded), 1)


def classify_failure(
    circuit: Circuit,
    graph: CircuitGraph,
    predicted_voltages: torch.Tensor | Sequence[float],
    oracle_voltages: torch.Tensor | Sequence[float] | None = None,
    *,
    invariant_metrics: dict[str, Any] | None = None,
    ood: bool = True,
) -> dict[str, Any]:
    predicted = torch.as_tensor(predicted_voltages, dtype=torch.float32).detach().cpu()
    oracle = None if oracle_voltages is None else torch.as_tensor(oracle_voltages, dtype=torch.float32).detach().cpu()
    invariant_metrics = dict(invariant_metrics or {})

    reasons: list[str] = []
    failure_type = "ood_generalization_failure"

    if graph.node_features.numel() == 0 or not graph.node_names or not _is_connected(circuit):
        failure_type = "disconnected_graph_confusion"
        reasons.append("graph is disconnected or degenerate")
    else:
        max_resistance, min_resistance = _circuit_resistance_stats(circuit)
        pred_range = float(predicted.max().item() - predicted.min().item()) if predicted.numel() else 0.0
        oracle_range = float(oracle.max().item() - oracle.min().item()) if oracle is not None and oracle.numel() else 0.0
        mean_abs_error = (
            float(torch.mean((predicted - oracle).abs()).item())
            if oracle is not None and oracle.numel() == predicted.numel()
            else float("inf")
        )
        unique_ratio = _unique_ratio(predicted)

        if max_resistance > 1e6 or (min_resistance > 0.0 and min_resistance < 1e-3):
            if mean_abs_error > 0.5:
                failure_type = "extreme_resistance_instability"
                reasons.append("resistance range is extreme")
        if failure_type == "ood_generalization_failure" and pred_range < 1e-4 and oracle_range > 1e-2:
            failure_type = "topology_collapse"
            reasons.append("predictions collapsed to nearly constant values")
        if failure_type == "ood_generalization_failure" and unique_ratio < 0.8 and mean_abs_error > 0.1:
            failure_type = "node_aliasing"
            reasons.append("multiple nodes map to the same prediction bucket")
        if failure_type == "ood_generalization_failure" and oracle is not None and oracle_range < 1e-2 and pred_range > 1.0:
            failure_type = "symmetry_failure"
            reasons.append("symmetric circuit expected similar voltages but prediction diverged")

    kcl = _stable_float(invariant_metrics.get("kcl_max_violation", 0.0))
    kvl = _stable_float(invariant_metrics.get("kvl_max_violation", 0.0))
    power = _stable_float(invariant_metrics.get("power_conservation_violation", 0.0))
    if any(value > 1e-6 for value in (kcl, kvl, power)):
        if failure_type == "ood_generalization_failure":
            # Sharp topology failure root causes
            if kcl > 1e-2:
                num_resistors = len(circuit.resistors)
                num_nodes = len(circuit.all_nodes)
                if num_resistors > num_nodes:
                    failure_type = "cycle_drift_failure"
                    reasons.append("KCL drift in closed cycles")
                elif num_nodes > 10:
                    failure_type = "dense_mesh_leakage"
                    reasons.append("high degree node connectivity leakage")
                else:
                    failure_type = "bridge_node_instability"
                    reasons.append("bridge/tree node prediction instability")
            else:
                failure_type = "conservation_drift"
        reasons.append("invariant violations exceed tolerance")

    if failure_type in ("ood_generalization_failure", "cycle_drift_failure", "dense_mesh_leakage", "bridge_node_instability") and not ood:
        failure_type = "conservation_drift" if reasons else "topology_collapse"

    if not reasons:
        reasons.append("general OOD error without a sharper taxonomy match")

    return {
        "circuit_name": circuit.name,
        "failure_type": failure_type,
        "reasons": tuple(sorted(dict.fromkeys(reasons))),
        "predicted_node_count": int(predicted.numel()),
        "oracle_node_count": 0 if oracle is None else int(oracle.numel()),
        "mean_abs_error": None if oracle is None or oracle.numel() != predicted.numel() else float(torch.mean((predicted - oracle).abs()).item()),
        "predicted_voltage_range": float(predicted.max().item() - predicted.min().item()) if predicted.numel() else 0.0,
        "oracle_voltage_range": float(oracle.max().item() - oracle.min().item()) if oracle is not None and oracle.numel() else 0.0,
        "invariant_metrics": {key: _stable_float(value) for key, value in sorted(invariant_metrics.items())},
        "ood": bool(ood),
    }


def summarize_failures(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(case.get("failure_type", "unknown") for case in cases)
    return {
        "count": len(cases),
        "failure_counts": {key: counts[key] for key in sorted(counts)},
        "dominant_failure": counts.most_common(1)[0][0] if counts else "none",
        "taxonomy": list(FAILURE_TYPES),
    }


def classify_failure_batch(
    cases: Sequence[tuple[Circuit, CircuitGraph, torch.Tensor, torch.Tensor | None, dict[str, Any] | None]]
) -> list[dict[str, Any]]:
    results = []
    for circuit, graph, pred, oracle, metrics in cases:
        results.append(classify_failure(circuit, graph, pred, oracle, invariant_metrics=metrics, ood=True))
    return results
