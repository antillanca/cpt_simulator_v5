#!/usr/bin/env python3
"""Generate projected-target dataset for CPT v2.10 Physics-Aware Surrogate Retraining.

For each circuit in train_10k.jsonl:
1. Obtain oracle solution via MNA solver.
2. Generate a realistic perturbation (Gaussian noise sigma=1.5V, clamped to +/-1000V).
3. Apply virtual-node projection to the perturbed voltages.
4. Compute blended solution: alpha * oracle + (1 - alpha) * projected.
5. Store as JSONL with full metadata and fingerprint.

Deterministic: seed=42 for all randomness.
Output: workspace/datasets/circuits/projected_targets_v210.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.graph_dataset import CircuitGraph, circuit_to_graph
from backend.circuits.models import Circuit
from backend.circuits.parser import parse_netlist
from backend.circuits.physics_projection import (
    PhysicsProjection,
    ProjectionConfig,
    _node_kcl_residual,
)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _fingerprint_row(row: Dict[str, Any]) -> str:
    """SHA-256 fingerprint of canonical JSON excluding the 'fingerprint' field."""
    filtered = {k: v for k, v in row.items() if k != "fingerprint"}
    return hashlib.sha256(_stable_json(filtered).encode("utf-8")).hexdigest()


def _oracle_voltages_by_node(solution, node_names: Tuple[str, ...]) -> List[float]:
    """Extract oracle voltages in graph node order."""
    return [solution.node_voltages.get(n, 0.0) for n in node_names]


def _perturb_oracle(
    oracle_v: torch.Tensor,
    sigma: float = 1.5,
    seed_offset: int = 0,
    clamp_value: float = 1000.0,
) -> torch.Tensor:
    """Add Gaussian noise to oracle voltages, clamped to safe bounds."""
    gen = torch.Generator()
    gen.manual_seed(42 + seed_offset)
    noise = torch.normal(mean=torch.zeros_like(oracle_v), std=sigma, generator=gen)
    perturbed = oracle_v + noise
    return torch.clamp(perturbed, min=-clamp_value, max=clamp_value)


def _max_kcl_residual(
    voltages: torch.Tensor,
    graph: CircuitGraph,
    circuit: Circuit,
) -> float:
    """Compute max |KCL residual| for a voltage vector."""
    res = _node_kcl_residual(voltages, graph, circuit, scale_by_conductance=False)
    if res.numel() == 0:
        return 0.0
    return res.abs().max().item()


def generate_projected_targets(
    dataset_path: Path,
    output_path: Path,
    alpha: float = 0.2,
    sigma: float = 1.5,
    projection_steps: int = 50,
    projection_tolerance: float = 1e-9,
    g_virtual: float = 1.0,
    blend_factor: float = 0.5,
) -> Dict[str, Any]:
    """Generate projected-target dataset.

    Returns summary statistics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Projection config: many iterations for convergence measurement
    proj_config = ProjectionConfig(
        steps=projection_steps,
        alpha_kcl=0.1,
        alpha_kvl=0.05,
        alpha_power=0.05,
        virtual_node_enabled=True,
        virtual_conductance=g_virtual,
        blend_factor=blend_factor,
        clamp_value=1e4,
    )
    projector = PhysicsProjection(proj_config)

    total = 0
    converged = 0
    total_proj_iters = 0
    rows: List[Dict[str, Any]] = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            netlist = row["netlist"]
            circuit_name = row.get("circuit_name", "unnamed")
            circuit_id = row.get("id", f"circuit_{line_idx}")

            try:
                circuit = parse_netlist(netlist, name=circuit_name)
                solution = solve_dc_circuit(circuit)
                graph = circuit_to_graph(circuit, solution)
            except Exception:
                continue

            if graph.node_features.size(0) == 0:
                continue

            node_names = graph.node_names
            oracle_v = graph.target_voltages.clone()

            # Perturb
            perturbed_v = _perturb_oracle(oracle_v, sigma=sigma, seed_offset=line_idx)

            # Apply projection and record per-step metrics
            step_metrics = projector.project_step_metrics(graph, circuit, perturbed_v)
            projected_v = projector.project(graph, circuit, perturbed_v)

            # Find convergence point (residual < tolerance)
            residual_history = [m["kcl_max_residual"] for m in step_metrics]
            iters_to_converge = len(residual_history)
            for i, r in enumerate(residual_history):
                if r < projection_tolerance:
                    iters_to_converge = i + 1
                    break

            # Compute blended solution
            blended_v = alpha * oracle_v + (1.0 - alpha) * projected_v

            # Compute correction distance
            n_nodes = oracle_v.numel()
            correction_distance = (
                (projected_v - perturbed_v).norm().item() / max(n_nodes ** 0.5, 1.0)
            )

            # Build output row
            oracle_dict = {n: oracle_v[i].item() for i, n in enumerate(node_names)}
            perturbed_dict = {n: perturbed_v[i].item() for i, n in enumerate(node_names)}
            projected_dict = {n: projected_v[i].item() for i, n in enumerate(node_names)}
            blended_dict = {n: blended_v[i].item() for i, n in enumerate(node_names)}

            out_row = {
                "circuit_id": circuit_id,
                "circuit_name": circuit_name,
                "netlist": netlist,
                "oracle_solution": oracle_dict,
                "perturbed_solution": perturbed_dict,
                "projected_solution": projected_dict,
                "blended_solution": blended_dict,
                "projection_iterations": iters_to_converge,
                "residual_history": [round(r, 12) for r in residual_history],
                "correction_distance": round(correction_distance, 9),
                "alpha": alpha,
                "sigma": sigma,
            }
            out_row["fingerprint"] = _fingerprint_row(out_row)
            rows.append(out_row)

            total += 1
            if iters_to_converge < len(residual_history) or (
                residual_history and residual_history[-1] < projection_tolerance
            ):
                converged += 1
            total_proj_iters += iters_to_converge

            if total % 1000 == 0:
                print(f"  Processed {total} circuits...")

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    # Dataset fingerprint
    dataset_fp = hashlib.sha256(
        "".join(r["fingerprint"] for r in rows).encode("utf-8")
    ).hexdigest()

    summary = {
        "total_circuits": total,
        "converged_within_limit": converged,
        "avg_projection_iterations": round(total_proj_iters / max(total, 1), 2),
        "alpha": alpha,
        "sigma": sigma,
        "projection_steps": projection_steps,
        "projection_tolerance": projection_tolerance,
        "g_virtual": g_virtual,
        "blend_factor": blend_factor,
        "dataset_fingerprint": dataset_fp,
        "output_path": str(output_path),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate projected-target dataset for v2.10 retraining."
    )
    parser.add_argument(
        "--dataset",
        default="workspace/datasets/circuits/train_10k.jsonl",
        help="Input JSONL dataset path",
    )
    parser.add_argument(
        "--output",
        default="workspace/datasets/circuits/projected_targets_v210.jsonl",
        help="Output projected targets JSONL path",
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="Blending factor (default: 0.2)")
    parser.add_argument("--sigma", type=float, default=1.5, help="Gaussian noise sigma for perturbation (default: 1.5V)")
    parser.add_argument("--projection-steps", type=int, default=50, help="Max projection iterations (default: 50)")
    parser.add_argument("--projection-tolerance", type=float, default=1e-9, help="Convergence tolerance (default: 1e-9)")
    parser.add_argument("--g-virtual", type=float, default=1.0, help="Virtual node conductance (default: 1.0)")
    parser.add_argument("--blend-factor", type=float, default=0.5, help="Virtual node blend factor (default: 0.5)")
    args = parser.parse_args()

    dataset_path = PROJECT_ROOT / args.dataset
    output_path = PROJECT_ROOT / args.output

    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}")
        return 1

    # Deterministic seeds
    torch.manual_seed(42)

    print(f"CPT v2.10 — Projected Target Dataset Generation")
    print(f"  Dataset:  {dataset_path}")
    print(f"  Output:   {output_path}")
    print(f"  Alpha:    {args.alpha}")
    print(f"  Sigma:    {args.sigma}")
    print(f"  Proj steps: {args.projection_steps}")
    print()

    summary = generate_projected_targets(
        dataset_path=dataset_path,
        output_path=output_path,
        alpha=args.alpha,
        sigma=args.sigma,
        projection_steps=args.projection_steps,
        projection_tolerance=args.projection_tolerance,
        g_virtual=args.g_virtual,
        blend_factor=args.blend_factor,
    )

    print(f"\nGeneration complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save summary alongside dataset
    summary_path = output_path.with_suffix(".manifest.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Manifest saved: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
