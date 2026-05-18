#!/usr/bin/env python3
"""CPT v2.9F — Circuit Arena: evaluate GNN surrogate with/without physics projection.

Supports topology families: radial, ladder, bridge, mesh, dense_mesh, current_source.
Reports raw vs corrected metrics with convergence analysis per family.

Usage:
  python -m backend.circuits.run_circuit_arena --steps 5 --alpha-kcl 1.0
  python -m backend.circuits.run_circuit_arena --virtual-node --virtual-conductance 0.1 --blend-factor 0.5
  python -m backend.circuits.run_circuit_arena --no-projection
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.graph_dataset import CircuitGraph, circuit_to_graph
from backend.circuits.models import Circuit
from backend.circuits.physics_projection import PhysicsProjection, ProjectionConfig
from backend.circuits.surrogate_eval import (
    SurrogateEvalResult,
    _compute_kcl_violation,
    _compute_kvl_violation,
    get_vmax,
)


# ---------------------------------------------------------------------------
# Topology families for arena evaluation
# ---------------------------------------------------------------------------

def _build_topology_families() -> Dict[str, List[Circuit]]:
    """Build a curated set of circuit families spanning sparse to dense."""
    from backend.circuits.models import Resistor, VoltageSource, CurrentSource

    families: Dict[str, List[Circuit]] = {}

    # --- RADIAL (sparse, the problematic case for GNN) ---
    radial_circuits = []
    for n_nodes in [3, 5, 8, 12]:
        resistors = []
        nodes = ['1']  # voltage source positive
        for i in range(1, n_nodes):
            prev = nodes[-1]
            new_node = f'n{i}'
            nodes.append(new_node)
            resistors.append(Resistor(
                name=f'R{i}', node_a=prev, node_b=new_node, resistance_ohm=float(100 * i),
            ))
        resistors.append(Resistor(
            name=f'R{n_nodes}', node_a=nodes[-1], node_b='0', resistance_ohm=100.0,
        ))
        radial_circuits.append(Circuit(
            name=f'radial_{n_nodes}',
            resistors=tuple(resistors),
            voltage_sources=(VoltageSource(name='V1', positive='1', negative='0', voltage=10.0),),
        ))
    families['radial'] = radial_circuits

    # --- LADDER (sequential with parallel branches) ---
    ladder_circuits = []
    for depth in [2, 3, 4]:
        resistors = []
        for i in range(depth):
            if i == 0:
                resistors.append(Resistor(name=f'Rh{i}', node_a='1', node_b=f'n{i}', resistance_ohm=100.0))
            else:
                resistors.append(Resistor(name=f'Rh{i}', node_a=f'n{i-1}', node_b=f'n{i}', resistance_ohm=100.0 + 50 * i))
            resistors.append(Resistor(name=f'Rv{i}', node_a=f'n{i}', node_b='0', resistance_ohm=200.0))
        ladder_circuits.append(Circuit(
            name=f'ladder_{depth}',
            resistors=tuple(resistors),
            voltage_sources=(VoltageSource(name='V1', positive='1', negative='0', voltage=10.0),),
        ))
    families['ladder'] = ladder_circuits

    # --- MESH (dense, well-conditioned for GNN) ---
    mesh_circuits = []
    mesh_circuits.append(Circuit(
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
    mesh_circuits.append(Circuit(
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
    families['mesh'] = mesh_circuits

    # --- DENSE_MESH (fully connected) ---
    dense_mesh_circuits = []
    dense_mesh_circuits.append(Circuit(
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
    families['dense_mesh'] = dense_mesh_circuits

    # --- BRIDGE (moderate sparsity, KVL-critical) ---
    bridge_circuits = []
    for r_bridge in [50.0, 150.0, 500.0, 1000.0]:
        bridge_circuits.append(Circuit(
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
    families['bridge'] = bridge_circuits

    # --- CURRENT_SOURCE (tests current source convention) ---
    cs_circuits = []
    for i_val in [0.1, 0.5, 1.0, 2.0]:
        cs_circuits.append(Circuit(
            name=f'current_{i_val}A',
            resistors=(
                Resistor(name='R1', node_a='n1', node_b='0', resistance_ohm=100.0),
                Resistor(name='R2', node_a='n1', node_b='0', resistance_ohm=200.0),
            ),
            current_sources=(CurrentSource(name='I1', positive='n1', negative='0', current=i_val),),
        ))
    families['current_source'] = cs_circuits

    return families


# ---------------------------------------------------------------------------
# Convergence slope computation
# ---------------------------------------------------------------------------

def _compute_convergence_slope(metrics_list: List[Dict[str, float]], key: str = "kcl_max_residual") -> float:
    """Compute slope of metric decay per step (linear regression).

    Returns negative slope for decaying metrics, 0.0 if insufficient data.
    """
    values = [m.get(key, 0.0) for m in metrics_list]
    if len(values) < 2:
        return 0.0

    n = len(values)
    indices = list(range(n))
    sum_x = sum(indices)
    sum_y = sum(values)
    sum_xy = sum(x * y for x, y in zip(indices, values))
    sum_x2 = sum(x * x for x in indices)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-15:
        return 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    return slope


def _compute_residual_decay_factor(metrics_list: List[Dict[str, float]], key: str = "kcl_max_residual") -> float:
    """Compute ratio of final residual to initial residual.

    Returns decay factor < 1.0 for converging, > 1.0 for diverging.
    """
    values = [m.get(key, 0.0) for m in metrics_list]
    if len(values) < 2 or abs(values[0]) < 1e-15:
        return 1.0
    return values[-1] / values[0]


# ---------------------------------------------------------------------------
# Arena runner
# ---------------------------------------------------------------------------

def run_arena(
    projection_config: ProjectionConfig | None = None,
    perturbation_scale: float = 1.0,
    seed: int = 42,
) -> Dict[str, Dict]:
    """Run physics projection arena on all topology families.

    For each family, perturbs oracle voltages and measures how well
    the physics projection recovers the correct solution.

    Returns: dict of family -> metrics dict
    """
    families = _build_topology_families()
    projector = PhysicsProjection(projection_config) if projection_config else None
    torch.manual_seed(seed)

    results: Dict[str, Dict] = {}

    for family_name, circuits in families.items():
        family_raw_mae = []
        family_corr_mae = []
        family_raw_kcl = []
        family_corr_kcl = []
        family_raw_kvl = []
        family_corr_kvl = []
        family_reduction_pct = []
        family_raw_slopes = []
        family_proj_slopes = []
        family_decay_factors = []

        for circuit in circuits:
            solution = solve_dc_circuit(circuit)
            graph = circuit_to_graph(circuit, solution)

            if graph.target_voltages.numel() == 0:
                continue

            # Simulate GNN prediction: perturb oracle by Gaussian noise
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
            v_pred = v_oracle + noise

            # Raw metrics
            raw_mae = (v_pred - v_oracle).abs().mean().item()

            pred_dict = dict(zip(graph.node_names, v_pred.tolist()))
            pred_dict[circuit.ground_node] = 0.0
            raw_kcl = _compute_kcl_violation(circuit, pred_dict)
            raw_kvl = _compute_kvl_violation(circuit, pred_dict)

            family_raw_mae.append(raw_mae)
            family_raw_kcl.append(raw_kcl)
            family_raw_kvl.append(raw_kvl)

            # Raw convergence slope (KCL residual of perturbed input)
            if projector is not None:
                # Compute baseline convergence (without projection) via step metrics
                baseline_cfg = ProjectionConfig(
                    alpha_kcl=projection_config.alpha_kcl if projection_config else 1.0,
                    alpha_kvl=projection_config.alpha_kvl if projection_config else 0.5,
                    steps=projection_config.steps if projection_config else 5,
                    virtual_node_enabled=False,
                )
                baseline_proj = PhysicsProjection(baseline_cfg)
                baseline_metrics = baseline_proj.project_step_metrics(graph, circuit, v_pred)
                raw_slope = _compute_convergence_slope(baseline_metrics)
                family_raw_slopes.append(raw_slope)

                # Corrected metrics (with projection + virtual node)
                v_corr = projector.project(graph, circuit, v_pred)
                corr_mae = (v_corr - v_oracle).abs().mean().item()
                corr_dict = dict(zip(graph.node_names, v_corr.tolist()))
                corr_dict[circuit.ground_node] = 0.0
                corr_kcl = _compute_kcl_violation(circuit, corr_dict)
                corr_kvl = _compute_kvl_violation(circuit, corr_dict)

                family_corr_mae.append(corr_mae)
                family_corr_kcl.append(corr_kcl)
                family_corr_kvl.append(corr_kvl)
                if raw_mae > 1e-12:
                    family_reduction_pct.append((1 - corr_mae / raw_mae) * 100)

                # Projected convergence slope
                proj_metrics = projector.project_step_metrics(graph, circuit, v_pred)
                proj_slope = _compute_convergence_slope(proj_metrics)
                family_proj_slopes.append(proj_slope)

                # Residual decay factor
                decay = _compute_residual_decay_factor(proj_metrics)
                family_decay_factors.append(decay)

        n = max(len(family_raw_mae), 1)
        entry: Dict[str, Any] = {
            'count': len(family_raw_mae),
            'raw_mae': sum(family_raw_mae) / n,
            'raw_kcl': max(family_raw_kcl) if family_raw_kcl else 0.0,
            'raw_kvl': max(family_raw_kvl) if family_raw_kvl else 0.0,
        }
        if projector is not None and family_corr_mae:
            entry.update({
                'corrected_mae': sum(family_corr_mae) / max(len(family_corr_mae), 1),
                'corrected_kcl': max(family_corr_kcl) if family_corr_kcl else 0.0,
                'corrected_kvl': max(family_corr_kvl) if family_corr_kvl else 0.0,
                'mae_reduction_pct': sum(family_reduction_pct) / max(len(family_reduction_pct), 1) if family_reduction_pct else 0.0,
                'raw_convergence_slope': sum(family_raw_slopes) / max(len(family_raw_slopes), 1) if family_raw_slopes else 0.0,
                'projected_convergence_slope': sum(family_proj_slopes) / max(len(family_proj_slopes), 1) if family_proj_slopes else 0.0,
                'residual_decay_factor': sum(family_decay_factors) / max(len(family_decay_factors), 1) if family_decay_factors else 1.0,
                'projection_gain': abs(sum(family_proj_slopes)) / max(abs(sum(family_raw_slopes)), 1e-12) if family_raw_slopes else 0.0,
            })
        results[family_name] = entry

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='CPT v2.9F Circuit Arena')
    parser.add_argument('--steps', type=int, default=5, help='Projection refinement steps')
    parser.add_argument('--alpha-kcl', type=float, default=1.0, help='KCL correction step size')
    parser.add_argument('--alpha-kvl', type=float, default=0.5, help='KVL correction step size')
    parser.add_argument('--omega', type=float, default=1.0, help='SOR over-relaxation factor')
    parser.add_argument('--perturbation', type=float, default=1.0, help='Voltage perturbation scale (V)')
    parser.add_argument('--no-projection', action='store_true', help='Run without projection')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--json', type=str, default=None, help='Output results to JSON file')
    # Virtual node CLI args
    parser.add_argument('--virtual-node', action='store_true', default=True, help='Enable virtual node projection (default: True)')
    parser.add_argument('--no-virtual-node', action='store_true', help='Disable virtual node projection')
    parser.add_argument('--virtual-conductance', type=float, default=0.1, help='Virtual node conductance')
    parser.add_argument('--blend-factor', type=float, default=0.5, help='Virtual node blend factor (beta)')
    args = parser.parse_args()

    if args.no_projection:
        proj_config = None
    else:
        virtual_enabled = not args.no_virtual_node
        proj_config = ProjectionConfig(
            alpha_kcl=args.alpha_kcl,
            alpha_kvl=args.alpha_kvl,
            steps=args.steps,
            omega=args.omega,
            virtual_node_enabled=virtual_enabled,
            virtual_conductance=args.virtual_conductance,
            blend_factor=args.blend_factor,
        )

    print(f'CPT v2.9F Circuit Arena')
    print(f'  projection: {"OFF" if args.no_projection else "ON"}')
    if proj_config:
        vn = "ON" if proj_config.virtual_node_enabled else "OFF"
        print(f'  alpha_kcl={proj_config.alpha_kcl}, alpha_kvl={proj_config.alpha_kvl}, steps={proj_config.steps}')
        print(f'  virtual_node={vn}, virtual_conductance={proj_config.virtual_conductance}, blend_factor={proj_config.blend_factor}')
    print(f'  perturbation_scale={args.perturbation}, seed={args.seed}')
    print()

    t0 = time.time()
    results = run_arena(
        projection_config=proj_config,
        perturbation_scale=args.perturbation,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    # Print results table
    header = f'{"Family":<16} {"Count":>5} {"Raw MAE":>10} {"Corr MAE":>10} {"Reduction":>10} {"Raw KCL":>10} {"Corr KCL":>10} {"Decay":>8}'
    print(header)
    print('-' * len(header))

    for family, data in sorted(results.items()):
        raw_mae = data['raw_mae']
        corr_mae = data.get('corrected_mae', float('nan'))
        reduction = data.get('mae_reduction_pct', float('nan'))
        raw_kcl = data['raw_kcl']
        corr_kcl = data.get('corrected_kcl', float('nan'))
        decay = data.get('residual_decay_factor', float('nan'))

        if args.no_projection:
            print(f'{family:<16} {data["count"]:>5} {raw_mae:>10.4f} {"N/A":>10} {"N/A":>10} {raw_kcl:>10.6f} {"N/A":>10} {"N/A":>8}')
        else:
            print(f'{family:<16} {data["count"]:>5} {raw_mae:>10.4f} {corr_mae:>10.4f} {reduction:>9.1f}% {raw_kcl:>10.6f} {corr_kcl:>10.6f} {decay:>7.4f}')

    print(f'\nElapsed: {elapsed:.2f}s')

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f'Results saved to {args.json}')


if __name__ == '__main__':
    main()
