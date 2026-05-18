#!/usr/bin/env python3
"""Ablation study: sweep blending alpha for CPT v2.10.

For each alpha in {0.0, 0.1, 0.2, 0.5, 1.0}:
1. Load the oracle dataset (train_10k.jsonl).
2. Generate blended targets with the given alpha.
3. Train a CircuitGNN with target_mode=blended_projection.
4. Evaluate projection effort and oracle MAE.
5. Report comparative results.

Usage:
    python scripts/run_ablation_alpha.py [--alphas 0.0 0.1 0.2 0.5 1.0] [--epochs 50]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.graph_dataset import CircuitGraph, circuit_to_graph
from backend.circuits.losses import voltage_loss, invariant_aware_loss
from backend.circuits.models import Circuit
from backend.circuits.parser import parse_netlist
from backend.circuits.physics_projection import PhysicsProjection, ProjectionConfig
from backend.circuits.projection_effort import (
    measure_projection_effort,
    aggregate_effort,
)
from backend.circuits.surrogate_eval import denormalize_voltages, evaluate_surrogate, get_vmax
from backend.neural.models.circuit_gnn import CircuitGNN, EdgeAwareCircuitGNN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_circuits_from_jsonl(path: Path):
    """Load circuits from JSONL, return list of (graph, circuit) pairs."""
    pairs = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            circuit = parse_netlist(rec["netlist"])
            graph = circuit_to_graph(circuit)
            pairs.append((graph, circuit, rec))
    return pairs


def generate_blended_targets(
    oracle_voltages: dict,
    perturbed_voltages: dict,
    projected_voltages: dict,
    alpha: float,
) -> dict:
    """Blend oracle and projected voltages.

    blended = alpha * oracle + (1 - alpha) * projected
    alpha=0.0 → pure projected (v2.10 target)
    alpha=1.0 → pure oracle (baseline target)
    """
    nodes = list(oracle_voltages.keys())
    blended = {}
    for n in nodes:
        o = oracle_voltages[n]
        p = projected_voltages.get(n, o)
        blended[n] = alpha * o + (1.0 - alpha) * p
    return blended


def perturb_voltages(oracle_voltages: dict, sigma: float, rng: random.Random) -> dict:
    """Add Gaussian noise to oracle voltages."""
    return {n: v + rng.gauss(0, sigma) for n, v in oracle_voltages.items()}


def train_model(
    train_graphs: list,
    train_targets: list,  # list of dict {node_name: voltage}
    train_circuits: list,
    epochs: int,
    lr: float,
    seed: int,
    use_edge: bool = True,
    ablation: str = "full",
) -> tuple:
    """Train a CircuitGNN and return (model, history)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    node_dim = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1] if use_edge and train_graphs[0].edge_attr is not None else 4
    hidden_dim = 64

    if use_edge:
        model = EdgeAwareCircuitGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
    else:
        model = CircuitGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn = voltage_loss

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for graph, target_v, circuit in zip(train_graphs, train_targets, train_circuits):
            optimizer.zero_grad()
            vmax = get_vmax(circuit)
            pred = model(graph.node_features, graph.edge_index, graph.edge_features if use_edge else None)
            pred_denorm = denormalize_voltages(pred.squeeze(-1), vmax)

            # Target tensor
            nodes = list(target_v.keys())
            target_tensor = torch.tensor([target_v[n] for n in nodes], dtype=torch.float32)

            loss = loss_fn(pred_denorm, target_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_graphs), 1)
        scheduler.step(avg_loss)
        history.append({"epoch": epoch + 1, "loss": avg_loss})

    return model, history


def evaluate_effort(
    model: torch.nn.Module,
    eval_graphs: list,
    eval_circuits: list,
    use_edge: bool = True,
) -> dict:
    """Evaluate projection effort for a model."""
    efforts = []
    config = ProjectionConfig(
        steps=50, alpha_kcl=0.1, alpha_kvl=0.05,
        virtual_node_enabled=True, virtual_conductance=1.0, blend_factor=0.5,
    )
    for graph, circuit in zip(eval_graphs, eval_circuits):
        vmax = get_vmax(circuit)
        with torch.no_grad():
            raw_pred = model(
                graph.node_features, graph.edge_index,
                graph.edge_features if use_edge and graph.edge_features is not None else None,
            )
            voltages = denormalize_voltages(raw_pred.squeeze(-1), vmax)
        effort = measure_projection_effort(voltages, graph, circuit, config)
        efforts.append(effort)

    return aggregate_effort(efforts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="CPT v2.10 Alpha Ablation Study")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--dataset", default="workspace/datasets/circuits/train_10k.jsonl")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma", type=float, default=1.5, help="Perturbation noise std (V)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="workspace/results/v210_ablation_alpha.json")
    parser.add_argument("--max-circuits", type=int, default=1000, help="Cap circuit count for speed")
    args = parser.parse_args()

    print(f"CPT v2.10 Alpha Ablation Study")
    print(f"Alphas: {args.alphas}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Sigma: {args.sigma}, Seed: {args.seed}")

    # Load circuits
    dataset_path = PROJECT_ROOT / args.dataset
    print(f"Loading dataset: {dataset_path}")
    all_pairs = load_circuits_from_jsonl(dataset_path)
    if args.max_circuits and len(all_pairs) > args.max_circuits:
        rng = random.Random(args.seed)
        rng.shuffle(all_pairs)
        all_pairs = all_pairs[:args.max_circuits]
    print(f"Loaded {len(all_pairs)} circuits")

    # Split 80/20
    split = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split]
    eval_pairs = all_pairs[split:]
    train_graphs = [p[0] for p in train_pairs]
    train_circuits = [p[1] for p in train_pairs]
    eval_graphs = [p[0] for p in eval_pairs]
    eval_circuits = [p[1] for p in eval_pairs]

    # Precompute oracle solutions and perturbed voltages
    print("Computing oracle solutions and perturbed voltages...")
    perturb_rng = random.Random(args.seed)
    projection_config = ProjectionConfig(
        steps=50, alpha_kcl=0.1, alpha_kvl=0.05,
        virtual_node_enabled=True, virtual_conductance=1.0, blend_factor=0.5,
    )
    projector = PhysicsProjection(projection_config)

    oracle_solutions = []
    perturbed_solutions = []
    projected_solutions = []

    for graph, circuit, rec in all_pairs:
        oracle_sol = solve_dc_circuit(circuit)
        oracle_v = {k: v for k, v in oracle_sol.node_voltages.items() if k != circuit.ground_node}
        oracle_solutions.append(oracle_v)

        perturbed_v = perturb_voltages(oracle_v, args.sigma, perturb_rng)
        perturbed_solutions.append(perturbed_v)

        # Project perturbed voltages
        nodes = list(oracle_v.keys())
        v_tensor = torch.tensor([perturbed_v[n] for n in nodes], dtype=torch.float32)
        projected_v_tensor = projector.project(graph, circuit, v_tensor)
        projected_v = {n: projected_v_tensor[i].item() for i, n in enumerate(nodes)}
        projected_solutions.append(projected_v)

    # Run ablation
    results = {}
    for alpha in args.alphas:
        print(f"\n{'='*50}")
        print(f"ALPHA = {alpha}  (oracle weight={alpha}, projected weight={1-alpha})")
        print(f"{'='*50}")

        # Generate blended targets for training set
        blended_targets = []
        for i in range(len(train_pairs)):
            bt = generate_blended_targets(
                oracle_solutions[i],
                perturbed_solutions[i],
                projected_solutions[i],
                alpha,
            )
            blended_targets.append(bt)

        # Train
        t0 = time.time()
        model, history = train_model(
            train_graphs, blended_targets, train_circuits,
            epochs=args.epochs, lr=args.lr, seed=args.seed,
        )
        train_time = time.time() - t0

        # Evaluate effort
        effort = evaluate_effort(model, eval_graphs, eval_circuits)

        # Oracle MAE on eval set
        mae_sum = 0.0
        mae_count = 0
        eval_offset = len(train_pairs)
        for i, (graph, circuit) in enumerate(zip(eval_graphs, eval_circuits)):
            vmax = get_vmax(circuit)
            with torch.no_grad():
                raw_pred = model(
                    graph.node_features, graph.edge_index,
                    graph.edge_features if graph.edge_features is not None else None,
                )
                pred_v = denormalize_voltages(raw_pred.squeeze(-1), vmax)
            oracle_v = oracle_solutions[eval_offset + i]
            nodes = list(oracle_v.keys())
            for j, n in enumerate(nodes):
                mae_sum += abs(pred_v[j].item() - oracle_v[n])
                mae_count += 1
        oracle_mae = mae_sum / max(mae_count, 1)

        results[str(alpha)] = {
            "alpha": alpha,
            "projection_effort": effort,
            "oracle_mae_V": oracle_mae,
            "train_loss_final": history[-1]["loss"] if history else None,
            "train_time_sec": round(train_time, 2),
            "epochs": args.epochs,
        }

        print(f"  Oracle MAE:     {oracle_mae:.6f} V")
        print(f"  Mean proj iters: {effort['mean_iterations']:.1f}")
        print(f"  Median proj iters: {effort['median_iterations']:.1f}")
        print(f"  Mean residual after 1 step: {effort.get('mean_initial_residual', 0):.6e}")
        print(f"  Mean correction distance: {effort['mean_correction_distance']:.6e}")
        print(f"  Train time: {train_time:.1f}s")

    # Save results
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False, default=str))
    print(f"\nAblation results saved: {output_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("ABLATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Alpha':<8} {'Oracle MAE(V)':<14} {'Mean Iters':<12} {'Med Iters':<12} {'Corr Dist':<14}")
    print("-" * 80)
    for alpha_str, r in sorted(results.items(), key=lambda x: float(x[0])):
        eff = r["projection_effort"]
        print(f"{r['alpha']:<8.2f} {r['oracle_mae_V']:<14.6f} {eff['mean_iterations']:<12.1f} {eff['median_iterations']:<12.1f} {eff['mean_correction_distance']:<14.6e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
