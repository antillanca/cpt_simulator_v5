#!/usr/bin/env python3
"""CPT v2.9F — Projection Convergence Analysis.

Measures convergence properties of physics projection with and without
virtual node across all topology families.

Metrics:
 1. convergence_per_step: MAE/KCL/KVL decay per step
 2. residual_decay_rate: exponential fit to residual decay
 3. effective_spectral_decay: estimated spectral radius from convergence
 4. radial_vs_mesh_convergence: compare convergence rates across families

Output: deterministic JSON + markdown summary

Usage:
  python -m backend.circuits.projection_analysis
  python -m backend.circuits.projection_analysis --steps 10 --perturbation 1.5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.graph_dataset import CircuitGraph, circuit_to_graph
from backend.circuits.models import Circuit
from backend.circuits.physics_projection import PhysicsProjection, ProjectionConfig


# ---------------------------------------------------------------------------
# Topology families (shared with arena)
# ---------------------------------------------------------------------------

def _build_topology_families() -> Dict[str, List[Circuit]]:
    """Build circuit families for convergence analysis."""
    from backend.circuits.models import Resistor, VoltageSource, CurrentSource

    families: Dict[str, List[Circuit]] = {}

    # --- RADIAL ---
    radial = []
    for n_nodes in [3, 5, 8, 12]:
        resistors = []
        nodes = ['1']
        for i in range(1, n_nodes):
            prev = nodes[-1]
            new_node = f'n{i}'
            nodes.append(new_node)
            resistors.append(Resistor(name=f'R{i}', node_a=prev, node_b=new_node, resistance_ohm=float(100 * i)))
        resistors.append(Resistor(name=f'R{n_nodes}', node_a=nodes[-1], node_b='0', resistance_ohm=100.0))
        radial.append(Circuit(
            name=f'radial_{n_nodes}',
            resistors=tuple(resistors),
            voltage_sources=(VoltageSource(name='V1', positive='1', negative='0', voltage=10.0),),
        ))
    families['radial'] = radial

    # --- LADDER ---
    ladder = []
    for depth in [2, 3, 4]:
        resistors = []
        for i in range(depth):
            # Horizontal rungs
            if i == 0:
                resistors.append(Resistor(name=f'Rh{i}', node_a='1', node_b=f'n{i}', resistance_ohm=100.0))
            else:
                resistors.append(Resistor(name=f'Rh{i}', node_a=f'n{i-1}', node_b=f'n{i}', resistance_ohm=100.0 + 50 * i))
            # Vertical rungs to ground
            resistors.append(Resistor(name=f'Rv{i}', node_a=f'n{i}', node_b='0', resistance_ohm=200.0))
        ladder.append(Circuit(
            name=f'ladder_{depth}',
            resistors=tuple(resistors),
            voltage_sources=(VoltageSource(name='V1', positive='1', negative='0', voltage=10.0),),
        ))
    families['ladder'] = ladder

    # --- MESH ---
    mesh = []
    mesh.append(Circuit(
        name='mesh_2x2',
        resistors=(
            Resistor(name='R1', node_a='1', node_b='n1', resistance_ohm=100.0),
            Resistor(name='R2', node_a='1', node_b='n2', resistance_ohm=200.0),
            Resistor(name='R3', node_a='n1', node_b='0', resistance_ohm=200.0),
            Resistor(name='R4', node_a='n2', node_b='0', resistance_ohm=100.0),
            Resistor(name='R5', node_a='n1', node_b='n2', resistance_ohm=150.0),
        ),
        voltage_sources=(VoltageSource(name='V1', positive='1', negative='0', voltage=10.0),),
    ))
    mesh.append(Circuit(
        name='mesh_ladder_3',
        resistors=(
            Resistor(name='R1', node_a='1', node_b='n1', resistance_ohm=100.0),
            Resistor(name='R2', node_a='n1', node_b='n2', resistance_ohm=200.0),
            Resistor(name='R3', node_a='n2', node_b='0', resistance_ohm=300.0),
            Resistor(name='R4', node_a='1', node_b='n3', resistance_ohm=150.0),
            Resistor(name='R5', node_a='n3', node_b='n4', resistance_ohm=250.0),
            Resistor(name='R6', node_a='n4', node_b='0', resistance_ohm=350.0),
            Resistor(name='R7', node_a='n1', node_b='n3', resistance_ohm=180.0),
            Resistor(name='R8', node_a='n2', node_b='n4', resistance_ohm=220.0),
        ),
        voltage_sources=(VoltageSource(name='V1', positive='1', negative='0', voltage=12.0),),
    ))
    families['mesh'] = mesh

    # --- DENSE_MESH ---
    dense_mesh = []
    # Fully connected 4-node
    dense_mesh.append(Circuit(
        name='dense_4node',
        resistors=(
            Resistor(name='R1', node_a='1', node_b='n1', resistance_ohm=100.0),
            Resistor(name='R2', node_a='1', node_b='n2', resistance_ohm=150.0),
            Resistor(name='R3', node_a='n1', node_b='n2', resistance_ohm=200.0),
            Resistor(name='R4', node_a='n1', node_b='0', resistance_ohm=250.0),
            Resistor(name='R5', node_a='n2', node_b='0', resistance_ohm=300.0),
            Resistor(name='R6', node_a='1', node_b='0', resistance_ohm=400.0),
        ),
        voltage_sources=(VoltageSource(name='V1', positive='1', negative='0', voltage=10.0),),
    ))
    families['dense_mesh'] = dense_mesh

    # --- BRIDGE ---
    bridge = []
    for r_bridge in [50.0, 150.0, 500.0, 1000.0]:
        bridge.append(Circuit(
            name=f'bridge_R{int(r_bridge)}',
            resistors=(
                Resistor(name='R1', node_a='1', node_b='n1', resistance_ohm=100.0),
                Resistor(name='R2', node_a='1', node_b='n2', resistance_ohm=200.0),
                Resistor(name='R3', node_a='n1', node_b='0', resistance_ohm=200.0),
                Resistor(name='R4', node_a='n2', node_b='0', resistance_ohm=100.0),
                Resistor(name='R5', node_a='n1', node_b='n2', resistance_ohm=r_bridge),
            ),
            voltage_sources=(VoltageSource(name='V1', positive='1', negative='0', voltage=10.0),),
        ))
    families['bridge'] = bridge

    # --- CURRENT_SOURCE ---
    cs = []
    for i_val in [0.1, 0.5, 1.0, 2.0]:
        cs.append(Circuit(
            name=f'current_{i_val}A',
            resistors=(
                Resistor(name='R1', node_a='n1', node_b='0', resistance_ohm=100.0),
                Resistor(name='R2', node_a='n1', node_b='0', resistance_ohm=200.0),
            ),
            current_sources=(CurrentSource(name='I1', positive='n1', negative='0', current=i_val),),
        ))
    families['current_source'] = cs

    return families


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

def _fit_exponential_decay(values: List[float]) -> Tuple[float, float]:
    """Fit exponential decay to a sequence: y_k ~ A * rho^k.

    Returns (A, rho) where rho is the decay rate per step.
    If insufficient data or non-positive values, returns (0.0, 1.0).
    """
    if len(values) < 2:
        return 0.0, 1.0

    # Filter out zeros/negatives for log fit
    pos_vals = [(i, v) for i, v in enumerate(values) if v > 1e-15]
    if len(pos_vals) < 2:
        return values[0] if values else 0.0, 0.0

    log_vals = [math.log(v) for _, v in pos_vals]
    indices = [float(i) for i, _ in pos_vals]
    n = len(log_vals)

    if n < 2:
        return values[0], 0.0

    # Linear regression on log(y) vs step
    sum_x = sum(indices)
    sum_y = sum(log_vals)
    sum_xy = sum(x * y for x, y in zip(indices, log_vals))
    sum_x2 = sum(x * x for x in indices)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-15:
        return values[0], 1.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    rho = math.exp(slope)  # decay rate per step
    A = math.exp(intercept)

    return A, rho


def _estimate_spectral_radius(residuals_per_step: List[float]) -> float:
    """Estimate effective spectral radius from residual decay.

    For Jacobi iteration on a linear system, the spectral radius rho
    determines convergence rate: ||e_k|| ~ rho^k * ||e_0||.

    We estimate rho from the ratio of consecutive residuals.
    """
    if len(residuals_per_step) < 2:
        return 1.0

    ratios = []
    for i in range(1, len(residuals_per_step)):
        prev = residuals_per_step[i - 1]
        curr = residuals_per_step[i]
        if prev > 1e-15:
            ratios.append(curr / prev)

    if not ratios:
        return 1.0

    return sum(ratios) / len(ratios)


def analyze_convergence(
    circuit: Circuit,
    graph: CircuitGraph,
    perturbation_scale: float = 1.5,
    steps: int = 10,
    seed: int = 42,
    alpha_kcl: float = 1.0,
    alpha_kvl: float = 0.5,
) -> Dict[str, Any]:
    """Run convergence analysis for a single circuit.

    Compares baseline Jacobi vs Virtual Node projection.

    Returns dict with:
    - baseline_metrics: per-step metrics without virtual node
    - virtual_metrics: per-step metrics with virtual node
    - baseline_decay_rate: exponential decay rate (rho) for KCL residual
    - virtual_decay_rate: same with virtual node
    - baseline_spectral_radius: estimated effective spectral radius
    - virtual_spectral_radius: same with virtual node
    - mae_baseline: final MAE without virtual node
    - mae_virtual: final MAE with virtual node
    """
    torch.manual_seed(seed)

    v_oracle = graph.target_voltages
    noise = torch.randn_like(v_oracle) * perturbation_scale

    # Don't perturb voltage-source nodes
    node_idx = {name: i for i, name in enumerate(graph.node_names)}
    vs_nodes = set()
    for vs in circuit.voltage_sources:
        vs_nodes.add(vs.positive)
        vs_nodes.add(vs.negative)
    for name in vs_nodes:
        if name in node_idx:
            noise[node_idx[name]] = 0.0
    v_pred = v_oracle + noise

    results = {}

    # --- Baseline (no virtual node) ---
    baseline_cfg = ProjectionConfig(
        alpha_kcl=alpha_kcl,
        alpha_kvl=alpha_kvl,
        steps=steps,
        virtual_node_enabled=False,
    )
    baseline_proj = PhysicsProjection(baseline_cfg)
    baseline_metrics = baseline_proj.project_step_metrics(graph, circuit, v_pred)
    v_baseline = baseline_proj.project(graph, circuit, v_pred)
    mae_baseline = (v_baseline - v_oracle).abs().mean().item()

    baseline_kcl = [m["kcl_max_residual"] for m in baseline_metrics]
    _, baseline_rho = _fit_exponential_decay(baseline_kcl)
    baseline_spectral = _estimate_spectral_radius(baseline_kcl)

    results["baseline_metrics"] = baseline_metrics
    results["baseline_decay_rate"] = baseline_rho
    results["baseline_spectral_radius"] = baseline_spectral
    results["mae_baseline"] = mae_baseline

    # --- Virtual Node ---
    virtual_cfg = ProjectionConfig(
        alpha_kcl=alpha_kcl,
        alpha_kvl=alpha_kvl,
        steps=steps,
        virtual_node_enabled=True,
        virtual_conductance=0.1,
        blend_factor=0.5,
    )
    virtual_proj = PhysicsProjection(virtual_cfg)
    virtual_metrics = virtual_proj.project_step_metrics(graph, circuit, v_pred)
    v_virtual = virtual_proj.project(graph, circuit, v_pred)
    mae_virtual = (v_virtual - v_oracle).abs().mean().item()

    virtual_kcl = [m["kcl_max_residual"] for m in virtual_metrics]
    _, virtual_rho = _fit_exponential_decay(virtual_kcl)
    virtual_spectral = _estimate_spectral_radius(virtual_kcl)

    results["virtual_metrics"] = virtual_metrics
    results["virtual_decay_rate"] = virtual_rho
    results["virtual_spectral_radius"] = virtual_spectral
    results["mae_virtual"] = mae_virtual
    results["improvement_pct"] = (1 - mae_virtual / max(mae_baseline, 1e-12)) * 100

    return results


def run_full_analysis(
    steps: int = 10,
    perturbation: float = 1.5,
    alpha_kcl: float = 1.0,
    alpha_kvl: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run convergence analysis across all topology families.

    Returns dict with per-family and summary results.
    """
    families = _build_topology_families()

    all_results: Dict[str, Any] = {}
    family_summary: Dict[str, Dict] = {}

    for family_name, circuits in families.items():
        family_data = []
        for circuit in circuits:
            solution = solve_dc_circuit(circuit)
            graph = circuit_to_graph(circuit, solution)
            if graph.target_voltages.numel() == 0:
                continue

            analysis = analyze_convergence(
                circuit, graph,
                perturbation_scale=perturbation,
                steps=steps,
                seed=seed,
                alpha_kcl=alpha_kcl,
                alpha_kvl=alpha_kvl,
            )
            analysis["circuit_name"] = circuit.name
            family_data.append(analysis)

        if not family_data:
            continue

        # Aggregate family metrics
        n = len(family_data)
        family_summary[family_name] = {
            "count": n,
            "avg_baseline_spectral": sum(d["baseline_spectral_radius"] for d in family_data) / n,
            "avg_virtual_spectral": sum(d["virtual_spectral_radius"] for d in family_data) / n,
            "avg_baseline_decay": sum(d["baseline_decay_rate"] for d in family_data) / n,
            "avg_virtual_decay": sum(d["virtual_decay_rate"] for d in family_data) / n,
            "avg_improvement_pct": sum(d["improvement_pct"] for d in family_data) / n,
        }
        all_results[family_name] = family_data

    all_results["summary"] = family_summary
    return all_results


def format_markdown(results: Dict[str, Any]) -> str:
    """Format convergence analysis results as markdown."""
    lines = []
    lines.append("# CPT v2.9F — Projection Convergence Analysis")
    lines.append("")
    lines.append("## Family-Level Summary")
    lines.append("")
    lines.append(f"| {'Family':<14} | {'Count':>5} | {'Base Spectral':>12} | {'VNode Spectral':>13} | {'Base Decay':>10} | {'VNode Decay':>11} | {'Improvement':>11} |")
    lines.append(f"|{'-':-16}|{'-':-7}|{'-':-14}|{'-':-15}|{'-':-12}|{'-':-13}|{'-':-13}|")

    summary = results.get("summary", {})
    for family_name in sorted(summary.keys()):
        d = summary[family_name]
        lines.append(
            f"| {family_name:<14} | {d['count']:>5} | {d['avg_baseline_spectral']:>12.4f} | "
            f"{d['avg_virtual_spectral']:>13.4f} | {d['avg_baseline_decay']:>10.4f} | "
            f"{d['avg_virtual_decay']:>11.4f} | {d['avg_improvement_pct']:>10.1f}% |"
        )

    lines.append("")
    lines.append("## Spectral Interpretation")
    lines.append("")
    lines.append("- **Base Spectral**: Effective spectral radius of Jacobi iteration (baseline)")
    lines.append("- **VNode Spectral**: Effective spectral radius with virtual node correction")
    lines.append("- **Base Decay**: Exponential decay rate of KCL residual per step (baseline)")
    lines.append("- **VNode Decay**: Exponential decay rate of KCL residual per step (virtual node)")
    lines.append("- **Improvement**: MAE reduction percentage from virtual node vs baseline")
    lines.append("")
    lines.append("Lower spectral radius = faster convergence.")
    lines.append("Decay rate < 1.0 means residuals are shrinking per step.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='CPT v2.9F Projection Convergence Analysis')
    parser.add_argument('--steps', type=int, default=10, help='Projection steps')
    parser.add_argument('--perturbation', type=float, default=1.5, help='Voltage perturbation scale (V)')
    parser.add_argument('--alpha-kcl', type=float, default=1.0, help='KCL step size')
    parser.add_argument('--alpha-kvl', type=float, default=0.5, help='KVL step size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--json', type=str, default=None, help='Output JSON file')
    parser.add_argument('--markdown', type=str, default=None, help='Output markdown file')
    args = parser.parse_args()

    print("CPT v2.9F Convergence Analysis")
    print(f"  steps={args.steps}, perturbation={args.perturbation}, alpha_kcl={args.alpha_kcl}")
    print()

    results = run_full_analysis(
        steps=args.steps,
        perturbation=args.perturbation,
        alpha_kcl=args.alpha_kcl,
        alpha_kvl=args.alpha_kvl,
        seed=args.seed,
    )

    md = format_markdown(results)
    print(md)

    if args.json:
        # Strip per-circuit metrics for compact JSON (keep summary only)
        output = {"summary": results.get("summary", {})}
        Path(args.json).write_text(json.dumps(output, indent=2))
        print(f"JSON saved to {args.json}")

    if args.markdown:
        Path(args.markdown).write_text(md)
        print(f"Markdown saved to {args.markdown}")


if __name__ == '__main__':
    main()
