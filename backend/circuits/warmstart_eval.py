#!/usr/bin/env python3
"""CPT v2.9F — Warm-Start Solver Experiment.

TOP PRIORITY: validates whether imperfect surrogate voltages reduce
oracle solver iterations. This is the real CPT breakthrough path.

Hypothesis: Even imperfect voltages are valuable if they reduce
solver iterations.

Uses a Jacobi iterative MNA solver (not direct numpy.linalg.solve)
to measure iteration counts from different initial conditions:
  1. Zero init (cold start)
  2. Surrogate init (perturbed oracle)
  3. Projected init (surrogate + physics projection)

Metrics:
  - iteration_count
  - runtime
  - convergence stability

Usage:
  python -m backend.circuits.warmstart_eval
  python -m backend.circuits.warmstart_eval --steps 5 --perturbation 1.5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.graph_dataset import CircuitGraph, circuit_to_graph
from backend.circuits.models import Circuit
from backend.circuits.physics_projection import PhysicsProjection, ProjectionConfig


# ---------------------------------------------------------------------------
# Iterative MNA solver (Jacobi)
# ---------------------------------------------------------------------------

def _build_mna_system(circuit: Circuit) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, int]]:
    """Build MNA matrix system for iterative solving.

    Returns (A, b, node_order, node_index) for the system Ax = b
    where x = [v_node1, v_node2, ..., i_vs1, i_vs2, ...].
    """
    nodes = list(circuit.all_nodes)  # sorted, ground excluded
    n = len(nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}

    m = len(circuit.voltage_sources)
    vs_index = {vs.name: idx for idx, vs in enumerate(circuit.voltage_sources)}

    size = n + m
    A = np.zeros((size, size), dtype=np.float64)
    b = np.zeros(size, dtype=np.float64)

    # Stamp resistors
    for r in circuit.resistors:
        g = 1.0 / r.resistance_ohm
        a_idx = node_index.get(r.node_a)
        b_idx = node_index.get(r.node_b)

        if a_idx is not None:
            A[a_idx, a_idx] += g
        if b_idx is not None:
            A[b_idx, b_idx] += g
        if a_idx is not None and b_idx is not None:
            A[a_idx, b_idx] -= g
            A[b_idx, a_idx] -= g

    # Stamp voltage sources
    for vs in circuit.voltage_sources:
        j_idx = n + vs_index[vs.name]
        pos_idx = node_index.get(vs.positive)
        neg_idx = node_index.get(vs.negative)

        if pos_idx is not None:
            A[pos_idx, j_idx] += 1.0
            A[j_idx, pos_idx] += 1.0
        if neg_idx is not None:
            A[neg_idx, j_idx] -= 1.0
            A[j_idx, neg_idx] -= 1.0
        b[j_idx] = vs.voltage

    # Stamp current sources
    for cs in circuit.current_sources:
        pos_idx = node_index.get(cs.positive)
        neg_idx = node_index.get(cs.negative)
        if pos_idx is not None:
            b[pos_idx] += cs.current
        if neg_idx is not None:
            b[neg_idx] -= cs.current

    return A, b, nodes, node_index


def solve_mna_jacobi(
    circuit: Circuit,
    x_init: Optional[np.ndarray] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    omega: float = 1.0,
) -> Tuple[np.ndarray, int, bool, List[float]]:
    """Solve MNA system using Jacobi iterative method.

    Args:
        circuit: Circuit to solve
        x_init: Initial guess for solution vector. None = zero init.
        max_iterations: Maximum iterations before giving up
        tolerance: Convergence tolerance (max residual)
        omega: Over-relaxation factor (1.0 = standard Jacobi)

    Returns:
        (solution, iterations, converged, residual_history)
    """
    A, b, nodes, node_index = _build_mna_system(circuit)
    n = A.shape[0]

    # Extract diagonal for Jacobi
    D = np.diag(A).copy()
    # Avoid division by zero on diagonal
    D[np.abs(D) < 1e-15] = 1.0

    # Initial guess
    if x_init is not None:
        x = x_init.copy()
    else:
        x = np.zeros(n, dtype=np.float64)

    converged = False
    iterations = 0
    residual_history = []

    for it in range(max_iterations):
        # Jacobi update: x_new = x + omega * (b - A*x) / D
        residual = b - A @ x
        max_residual = np.max(np.abs(residual))
        residual_history.append(max_residual)

        if max_residual < tolerance:
            converged = True
            iterations = it + 1
            break

        # Jacobi step
        correction = omega * residual / D
        x = x + correction
        iterations = it + 1

    return x, iterations, converged, residual_history


def extract_voltages_from_solution(
    solution: np.ndarray,
    nodes: List[str],
    circuit: Circuit,
) -> Dict[str, float]:
    """Extract node voltages from MNA solution vector."""
    node_voltages = {"0": 0.0}
    for i, node in enumerate(nodes):
        node_voltages[node] = round(float(solution[i]), 9)
    return node_voltages


# ---------------------------------------------------------------------------
# Warm-start experiment
# ---------------------------------------------------------------------------

def run_warmstart_experiment(
    circuit: Circuit,
    graph: CircuitGraph,
    perturbation_scale: float = 1.5,
    projection_steps: int = 5,
    alpha_kcl: float = 1.0,
    alpha_kvl: float = 0.5,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    seed: int = 42,
    virtual_node: bool = True,
) -> Dict[str, Any]:
    """Run warm-start experiment for a single circuit.

    Compares:
      1. Zero init (cold start)
      2. Surrogate init (perturbed oracle)
      3. Projected init (surrogate + physics projection)
      4. Projected+VirtualNode init (surrogate + projection + virtual node)

    Returns dict with iteration counts and convergence info.
    """
    torch.manual_seed(seed)

    A, b, nodes, node_index = _build_mna_system(circuit)
    n_nodes = len(nodes)
    n_total = A.shape[0]  # nodes + voltage source currents

    # Oracle solution (direct solve for ground truth)
    x_oracle = np.linalg.solve(A, b)
    oracle_voltages = extract_voltages_from_solution(x_oracle, nodes, circuit)

    # --- Strategy 1: Zero init ---
    t0 = time.time()
    x_zero, iters_zero, conv_zero, res_zero = solve_mna_jacobi(
        circuit, x_init=None, max_iterations=max_iterations, tolerance=tolerance,
    )
    t_zero = time.time() - t0

    # --- Strategy 2: Surrogate init (perturbed oracle) ---
    v_oracle = graph.target_voltages
    noise = torch.randn_like(v_oracle) * perturbation_scale
    node_idx = {name: i for i, name in enumerate(graph.node_names)}
    vs_nodes = set()
    for vs in circuit.voltage_sources:
        vs_nodes.add(vs.positive)
        vs_nodes.add(vs.negative)
    for name in vs_nodes:
        if name in node_idx:
            noise[node_idx[name]] = 0.0
    v_surrogate = v_oracle + noise

    # Build surrogate init vector
    x_surrogate_init = np.zeros(n_total, dtype=np.float64)
    for i, node in enumerate(nodes):
        if node in node_idx:
            x_surrogate_init[i] = v_surrogate[node_idx[node]].item()
        else:
            x_surrogate_init[i] = v_oracle[node_idx.get(node, 0)].item() if node in node_idx else 0.0

    t0 = time.time()
    x_surr, iters_surr, conv_surr, res_surr = solve_mna_jacobi(
        circuit, x_init=x_surrogate_init, max_iterations=max_iterations, tolerance=tolerance,
    )
    t_surr = time.time() - t0

    # --- Strategy 3: Projected init (no virtual node) ---
    baseline_cfg = ProjectionConfig(
        alpha_kcl=alpha_kcl,
        alpha_kvl=alpha_kvl,
        steps=projection_steps,
        virtual_node_enabled=False,
    )
    baseline_proj = PhysicsProjection(baseline_cfg)
    v_projected = baseline_proj.project(graph, circuit, v_surrogate)

    x_projected_init = np.zeros(n_total, dtype=np.float64)
    for i, node in enumerate(nodes):
        if node in node_idx:
            x_projected_init[i] = v_projected[node_idx[node]].item()
        else:
            x_projected_init[i] = 0.0

    t0 = time.time()
    x_proj, iters_proj, conv_proj, res_proj = solve_mna_jacobi(
        circuit, x_init=x_projected_init, max_iterations=max_iterations, tolerance=tolerance,
    )
    t_proj = time.time() - t0

    # --- Strategy 4: Projected + Virtual Node init ---
    if virtual_node:
        virtual_cfg = ProjectionConfig(
            alpha_kcl=alpha_kcl,
            alpha_kvl=alpha_kvl,
            steps=projection_steps,
            virtual_node_enabled=True,
            virtual_conductance=0.1,
            blend_factor=0.5,
        )
        virtual_proj = PhysicsProjection(virtual_cfg)
        v_virtual = virtual_proj.project(graph, circuit, v_surrogate)

        x_virtual_init = np.zeros(n_total, dtype=np.float64)
        for i, node in enumerate(nodes):
            if node in node_idx:
                x_virtual_init[i] = v_virtual[node_idx[node]].item()
            else:
                x_virtual_init[i] = 0.0

        t0 = time.time()
        x_vn, iters_vn, conv_vn, res_vn = solve_mna_jacobi(
            circuit, x_init=x_virtual_init, max_iterations=max_iterations, tolerance=tolerance,
        )
        t_vn = time.time() - t0
    else:
        iters_vn = iters_proj
        conv_vn = conv_proj
        t_vn = t_proj
        res_vn = res_proj

    return {
        "circuit_name": circuit.name,
        "n_nodes": n_nodes,
        "zero_iters": iters_zero,
        "zero_converged": conv_zero,
        "zero_time_s": t_zero,
        "surrogate_iters": iters_surr,
        "surrogate_converged": conv_surr,
        "surrogate_time_s": t_surr,
        "projected_iters": iters_proj,
        "projected_converged": conv_proj,
        "projected_time_s": t_proj,
        "virtual_iters": iters_vn,
        "virtual_converged": conv_vn,
        "virtual_time_s": t_vn,
        "surrogate_saving_pct": (1 - iters_surr / max(iters_zero, 1)) * 100,
        "projected_saving_pct": (1 - iters_proj / max(iters_zero, 1)) * 100,
        "virtual_saving_pct": (1 - iters_vn / max(iters_zero, 1)) * 100,
    }


def run_full_warmstart_eval(
    steps: int = 5,
    perturbation: float = 1.5,
    alpha_kcl: float = 1.0,
    alpha_kvl: float = 0.5,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run warm-start evaluation across all topology families."""
    from backend.circuits.run_circuit_arena import _build_topology_families

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

            result = run_warmstart_experiment(
                circuit, graph,
                perturbation_scale=perturbation,
                projection_steps=steps,
                alpha_kcl=alpha_kcl,
                alpha_kvl=alpha_kvl,
                max_iterations=max_iterations,
                tolerance=tolerance,
                seed=seed,
            )
            family_data.append(result)

        if not family_data:
            continue

        n = len(family_data)
        family_summary[family_name] = {
            "count": n,
            "avg_zero_iters": sum(d["zero_iters"] for d in family_data) / n,
            "avg_surrogate_iters": sum(d["surrogate_iters"] for d in family_data) / n,
            "avg_projected_iters": sum(d["projected_iters"] for d in family_data) / n,
            "avg_virtual_iters": sum(d["virtual_iters"] for d in family_data) / n,
            "avg_surrogate_saving_pct": sum(d["surrogate_saving_pct"] for d in family_data) / n,
            "avg_projected_saving_pct": sum(d["projected_saving_pct"] for d in family_data) / n,
            "avg_virtual_saving_pct": sum(d["virtual_saving_pct"] for d in family_data) / n,
            "all_converged": all(d["zero_converged"] and d["projected_converged"] and d["virtual_converged"] for d in family_data),
        }
        all_results[family_name] = family_data

    all_results["summary"] = family_summary
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='CPT v2.9F Warm-Start Solver Experiment')
    parser.add_argument('--steps', type=int, default=5, help='Projection steps')
    parser.add_argument('--perturbation', type=float, default=1.5, help='Voltage perturbation scale (V)')
    parser.add_argument('--alpha-kcl', type=float, default=1.0, help='KCL step size')
    parser.add_argument('--alpha-kvl', type=float, default=0.5, help='KVL step size')
    parser.add_argument('--max-iters', type=int, default=1000, help='Max Jacobi iterations')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Convergence tolerance')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--json', type=str, default=None, help='Output JSON file')
    args = parser.parse_args()

    print("CPT v2.9F Warm-Start Solver Experiment")
    print(f"  steps={args.steps}, perturbation={args.perturbation}, tolerance={args.tolerance}")
    print()

    results = run_full_warmstart_eval(
        steps=args.steps,
        perturbation=args.perturbation,
        alpha_kcl=args.alpha_kcl,
        alpha_kvl=args.alpha_kvl,
        max_iterations=args.max_iters,
        tolerance=args.tolerance,
        seed=args.seed,
    )

    # Print results table
    summary = results.get("summary", {})
    header = f'{"Family":<16} {"Zero":>6} {"Surr":>6} {"Proj":>6} {"VNode":>6} {"Sur%":>6} {"Proj%":>6} {"VN%":>6} {"Conv":>5}'
    print(header)
    print("-" * len(header))

    for family_name in sorted(summary.keys()):
        d = summary[family_name]
        conv = "OK" if d["all_converged"] else "FAIL"
        print(
            f'{family_name:<16} {d["avg_zero_iters"]:>6.0f} {d["avg_surrogate_iters"]:>6.0f} '
            f'{d["avg_projected_iters"]:>6.0f} {d["avg_virtual_iters"]:>6.0f} '
            f'{d["avg_surrogate_saving_pct"]:>5.1f}% {d["avg_projected_saving_pct"]:>5.1f}% '
            f'{d["avg_virtual_saving_pct"]:>5.1f}% {conv:>5}'
        )

    if args.json:
        output = {"summary": summary}
        Path(args.json).write_text(json.dumps(output, indent=2))
        print(f"\nJSON saved to {args.json}")


if __name__ == '__main__':
    main()
