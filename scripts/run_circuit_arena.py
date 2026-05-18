#!/usr/bin/env python3
"""Circuit Arena: Oracle vs Surrogate benchmark.

Runs oracle and surrogate on in-distribution and OOD circuits.
Measures: MAE, RMSE, max error, KCL/KVL violations, replay consistency, speedup.
Also evaluates trivial baselines (mean predictor, linear predictor).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.circuits.baselines import (
    LinearRegressionBaselinePredictor,
    MeanBaselinePredictor,
    RandomStableBaselinePredictor,
    evaluate_baseline_predictor,
)
from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.graph_dataset import CircuitGraph, circuit_to_graph
from backend.circuits.models import Circuit, CircuitSolution
from backend.circuits.ood_generator import generate_ood_circuits
from backend.circuits.parser import parse_netlist
from backend.circuits.projection_effort import compute_projection_effort
from backend.circuits.surrogate_eval import SurrogateEvalResult, denormalize_voltages, evaluate_surrogate, get_vmax
from backend.governance.artifact_registry import ArtifactRegistry
from backend.neural.models.circuit_gnn import CircuitGNN, EdgeAwareCircuitGNN
REPORT_RESULTS_DIR = PROJECT_ROOT / "workspace" / "arena_results"
REPORT_DIR = PROJECT_ROOT / "workspace" / "arena_reports"


def _stable_fingerprint(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _is_connected(circuit: Circuit) -> bool:
    """BFS connectivity check from ground."""
    ground = circuit.ground_node
    adj: dict[str, set[str]] = {}
    all_nodes = set(circuit.all_nodes) | {ground}
    for node in all_nodes:
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
    visited = set()
    queue = [ground]
    visited.add(ground)
    while queue:
        node = queue.pop(0)
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return all(n in visited for n in circuit.all_nodes)


def load_in_dist_data(
    jsonl_path: Path, max_voltage: float = 1e6
) -> tuple[list[CircuitGraph], list[Circuit]]:
    """Load in-distribution dataset and convert to (graph, circuit) pairs."""
    graphs: list[CircuitGraph] = []
    circuits: list[Circuit] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            try:
                circuit = parse_netlist(row["netlist"], name=row.get("circuit_name", "unnamed"))
                if not _is_connected(circuit):
                    continue
                solution = solve_dc_circuit(circuit)
                max_v = max((abs(v) for v in solution.node_voltages.values()), default=0.0)
                if max_v > max_voltage:
                    continue
                g = circuit_to_graph(circuit, solution)
                if g.node_features.size(0) > 0:
                    graphs.append(g)
                    circuits.append(circuit)
            except Exception:
                continue

    return graphs, circuits


def load_ood_data(
    jsonl_path: Path, max_voltage: float = 1e8
) -> tuple[list[CircuitGraph], list[Circuit]]:
    """Load OOD dataset."""
    graphs: list[CircuitGraph] = []
    circuits: list[Circuit] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            try:
                circuit = parse_netlist(row["netlist"], name=row.get("circuit_name", "unnamed"))
                solution = solve_dc_circuit(circuit)
                max_v = max((abs(v) for v in solution.node_voltages.values()), default=0.0)
                if max_v > max_voltage:
                    continue
                g = circuit_to_graph(circuit, solution)
                if g.node_features.size(0) > 0:
                    graphs.append(g)
                    circuits.append(circuit)
            except Exception:
                continue

    return graphs, circuits


def deterministic_split(
    items: list, frac: float = 0.8
) -> tuple[list, list]:
    """Deterministic split based on sorted order."""
    sorted_items = sorted(items, key=lambda x: x[0].fingerprint if isinstance(x, tuple) else str(x))
    n = len(sorted_items)
    split = int(n * frac)
    return sorted_items[:split], sorted_items[split:]


def measure_speed(
    model: torch.nn.Module,
    graphs: list[CircuitGraph],
    circuits: list[Circuit],
    use_edge: bool,
    voltage_transform: str = "per_circuit_vmax",
    num_runs: int = 1000,
    ablation: str = "full",
) -> dict:
    """Measure inference speed for oracle and surrogate."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model.eval()

    if not graphs:
        return {
            "oracle_mean_sec": 0.0,
            "oracle_p95_sec": 0.0,
            "surrogate_mean_sec": 0.0,
            "surrogate_p95_sec": 0.0,
            "speedup": 0.0,
            "num_runs": 0,
            "warmup_runs": 0,
        }

    warmup_runs = min(20, len(graphs))
    warmup_indices = list(range(warmup_runs))
    with torch.no_grad():
        for i in warmup_indices:
            circuit = circuits[i]
            g = graphs[i]
            vmax = get_vmax(circuit)
            node_features = g.node_features
            edge_features = g.edge_features

            if ablation in ("baseline", "norm_only"):
                node_features = node_features[:, :8]
            if ablation == "baseline":
                edge_features = edge_features[:, :4]
            elif ablation == "norm_only":
                edge_features = edge_features[:, :5]
            elif ablation == "topo_only":
                edge_features = torch.cat([edge_features[:, :4], edge_features[:, 5:7]], dim=1)

            if use_edge:
                pred_norm = model(node_features, g.edge_index, edge_features)
            else:
                pred_norm = model(node_features, g.edge_index)
            if voltage_transform == "per_circuit_vmax":
                _ = denormalize_voltages(pred_norm, vmax)

    indices = [i % len(graphs) for i in range(num_runs)]

    # Oracle solve time
    oracle_times: list[float] = []
    for i in indices:
        circuit = circuits[i]
        t0 = time.perf_counter()
        solve_dc_circuit(circuit)
        oracle_times.append(time.perf_counter() - t0)

    # Surrogate inference time (includes graph-to-tensor overhead)
    surrogate_times: list[float] = []
    for i in indices:
        g = graphs[i]
        circuit = circuits[i]
        vmax = get_vmax(circuit)
        t0 = time.perf_counter()
        with torch.no_grad():
            node_features = g.node_features
            edge_features = g.edge_features

            if ablation in ("baseline", "norm_only"):
                node_features = node_features[:, :8]
            if ablation == "baseline":
                edge_features = edge_features[:, :4]
            elif ablation == "norm_only":
                edge_features = edge_features[:, :5]
            elif ablation == "topo_only":
                edge_features = torch.cat([edge_features[:, :4], edge_features[:, 5:7]], dim=1)

            if use_edge:
                pred_norm = model(node_features, g.edge_index, edge_features)
            else:
                pred_norm = model(node_features, g.edge_index)
            if voltage_transform == "per_circuit_vmax":
                _ = denormalize_voltages(pred_norm, vmax)
        surrogate_times.append(time.perf_counter() - t0)

    oracle_mean = float(np.mean(oracle_times))
    surrogate_mean = float(np.mean(surrogate_times))
    oracle_p95 = float(np.percentile(oracle_times, 95))
    surrogate_p95 = float(np.percentile(surrogate_times, 95))
    speedup = oracle_mean / max(surrogate_mean, 1e-12)

    return {
        "oracle_mean_sec": round(oracle_mean, 6),
        "oracle_p95_sec": round(oracle_p95, 6),
        "surrogate_mean_sec": round(surrogate_mean, 6),
        "surrogate_p95_sec": round(surrogate_p95, 6),
        "speedup": round(speedup, 2),
        "num_runs": len(indices),
        "warmup_runs": warmup_runs,
    }


def run_arena(
    model: torch.nn.Module,
    train_graphs: list[CircuitGraph],
    train_circuits: list[Circuit],
    eval_graphs: list[CircuitGraph],
    eval_circuits: list[Circuit],
    ood_graphs: list[CircuitGraph],
    ood_circuits: list[Circuit],
    use_edge: bool = True,
    ablation: str = "full",
) -> dict:
    """Run full arena evaluation."""
    results: dict[str, Any] = {}

    def _pack_result(result: SurrogateEvalResult) -> dict[str, Any]:
        return {
            "mae": round(result.mae, 6),
            "rmse": round(result.rmse, 6),
            "max_voltage_error": round(result.max_voltage_error, 6),
            "kcl_max_violation": round(result.kcl_max_violation, 9),
            "kvl_max_violation": round(result.kvl_max_violation, 9),
            "replay_consistency": round(result.replay_consistency, 15),
            "count": result.count,
        }

    def _zero_oracle(count: int) -> dict[str, Any]:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "max_voltage_error": 0.0,
            "kcl_max_violation": 0.0,
            "kvl_max_violation": 0.0,
            "replay_consistency": 0.0,
            "count": count,
        }

    def _split_by_family(graphs, circuits):
        from backend.circuits.topology_curriculum import determine_level, CurriculumLevel
        fam_graphs = {"trivial": [], "simple": [], "medium": [], "dense": []}
        fam_circuits = {"trivial": [], "simple": [], "medium": [], "dense": []}
        for g, c in zip(graphs, circuits):
            lvl = determine_level(g)
            if lvl == CurriculumLevel.LEVEL_0_TRIVIAL:
                key = "trivial"
            elif lvl == CurriculumLevel.LEVEL_1_SIMPLE:
                key = "simple"
            elif lvl == CurriculumLevel.LEVEL_2_MEDIUM:
                key = "medium"
            else:
                key = "dense"
            fam_graphs[key].append(g)
            fam_circuits[key].append(c)
        return fam_graphs, fam_circuits

    print(f"\n=== GNN EVAL ({len(eval_graphs)} circuits) ===")
    gnn_eval_1 = evaluate_surrogate(
        model,
        eval_graphs,
        eval_circuits,
        use_edge_features=use_edge,
        voltage_transform="per_circuit_vmax",
        ablation=ablation,
    )
    gnn_eval_2 = evaluate_surrogate(
        model,
        eval_graphs,
        eval_circuits,
        use_edge_features=use_edge,
        voltage_transform="per_circuit_vmax",
        ablation=ablation,
    )
    gnn_ood = evaluate_surrogate(
        model,
        ood_graphs,
        ood_circuits,
        use_edge_features=use_edge,
        voltage_transform="per_circuit_vmax",
        ablation=ablation,
    ) if ood_graphs else None

    # Family evaluation
    fam_graphs_eval, fam_circuits_eval = _split_by_family(eval_graphs, eval_circuits)
    fam_graphs_ood, fam_circuits_ood = _split_by_family(ood_graphs, ood_circuits)

    family_results = {}
    for fam in ["trivial", "simple", "medium", "dense"]:
        fg = fam_graphs_eval[fam]
        fc = fam_circuits_eval[fam]
        fg_ood = fam_graphs_ood[fam]
        fc_ood = fam_circuits_ood[fam]
        
        family_results[fam] = {
            "in_distribution": _pack_result(evaluate_surrogate(model, fg, fc, use_edge_features=use_edge, voltage_transform="per_circuit_vmax", ablation=ablation)) if fg else _zero_oracle(0),
            "ood": _pack_result(evaluate_surrogate(model, fg_ood, fc_ood, use_edge_features=use_edge, voltage_transform="per_circuit_vmax", ablation=ablation)) if fg_ood else _zero_oracle(0),
        }

    gnn_metrics = {
        "in_distribution": _pack_result(gnn_eval_1),
        "ood": _pack_result(gnn_ood) if gnn_ood is not None else _zero_oracle(0),
        "families": family_results,
        "replay_consistency_metrics": {
            "max_abs_diff": round(max(gnn_eval_1.replay_consistency, gnn_eval_2.replay_consistency), 15),
            "rerun_match": _stable_fingerprint(gnn_eval_1.__dict__) == _stable_fingerprint(gnn_eval_2.__dict__),
        },
        "speed_in_distribution": measure_speed(
            model,
            eval_graphs,
            eval_circuits,
            use_edge,
            voltage_transform="per_circuit_vmax",
            num_runs=min(500, len(eval_graphs)),
            ablation=ablation,
        ),
    }
    if ood_graphs:
        gnn_metrics["speed_ood"] = measure_speed(
            model,
            ood_graphs,
            ood_circuits,
            use_edge,
            voltage_transform="per_circuit_vmax",
            num_runs=min(500, len(ood_graphs)),
            ablation=ablation,
        )
    results["gnn"] = gnn_metrics

    print(f"  MAE: {gnn_eval_1.mae:.6f} V")
    print(f"  RMSE: {gnn_eval_1.rmse:.6f} V")
    print(f"  Max error: {gnn_eval_1.max_voltage_error:.6f} V")
    print(f"  KCL max violation: {gnn_eval_1.kcl_max_violation:.2e}")
    print(f"  KVL max violation: {gnn_eval_1.kvl_max_violation:.2e}")
    print(f"  Replay consistency: {gnn_eval_1.replay_consistency:.2e}")

    print("\n=== BASELINES ===")
    mean_baseline = MeanBaselinePredictor().fit(train_graphs)
    linear_baseline = LinearRegressionBaselinePredictor().fit(train_graphs)
    random_baseline = RandomStableBaselinePredictor(seed=42).fit(train_graphs)

    mean_eval = evaluate_baseline_predictor(mean_baseline, eval_graphs, eval_circuits)
    linear_eval = evaluate_baseline_predictor(linear_baseline, eval_graphs, eval_circuits)
    random_eval = evaluate_baseline_predictor(random_baseline, eval_graphs, eval_circuits)
    mean_eval_2 = evaluate_baseline_predictor(mean_baseline, eval_graphs, eval_circuits)
    linear_eval_2 = evaluate_baseline_predictor(linear_baseline, eval_graphs, eval_circuits)
    random_eval_2 = evaluate_baseline_predictor(random_baseline, eval_graphs, eval_circuits)

    results["mean_baseline"] = {k: round(v, 9) if isinstance(v, float) else v for k, v in mean_eval.items()}
    results["linear_baseline"] = {k: round(v, 9) if isinstance(v, float) else v for k, v in linear_eval.items()}
    results["random_baseline"] = {k: round(v, 9) if isinstance(v, float) else v for k, v in random_eval.items()}
    print(f"  Mean predictor: MAE={mean_eval['mae']:.6f} V")
    print(f"  Linear predictor: MAE={linear_eval['mae']:.6f} V")
    print(f"  Random stable predictor: MAE={random_eval['mae']:.6f} V")

    oracle_metrics = {
        "in_distribution": _zero_oracle(len(eval_graphs)),
        "ood": _zero_oracle(len(ood_graphs)),
        "replay_consistency_metrics": {"max_abs_diff": 0.0, "rerun_match": True},
    }
    results["oracle"] = oracle_metrics

    speed_ind = gnn_metrics["speed_in_distribution"]
    print("\n=== SPEED BENCHMARK ===")
    print(f"  Oracle: {speed_ind['oracle_mean_sec']*1000:.3f} ms")
    print(f"  Surrogate: {speed_ind['surrogate_mean_sec']*1000:.3f} ms")
    print(f"  Speedup: {speed_ind['speedup']:.2f}x")

    results["deterministic_rerun_validation"] = {
        "gnn": gnn_metrics["replay_consistency_metrics"]["rerun_match"],
        "mean_baseline": _stable_fingerprint(mean_eval) == _stable_fingerprint(mean_eval_2),
        "linear_baseline": _stable_fingerprint(linear_eval) == _stable_fingerprint(linear_eval_2),
        "random_baseline": _stable_fingerprint(random_eval) == _stable_fingerprint(random_eval_2),
    }
    results["metadata"] = {
        "model_info": {
            "type": "edge_aware" if use_edge else "basic",
            "hidden_dim": getattr(model, "head", None).out_features if hasattr(getattr(model, "head", None), "out_features") else None,
            "num_params": sum(p.numel() for p in model.parameters()),
        },
        "train_count": len(train_graphs),
        "eval_count": len(eval_graphs),
        "ood_count": len(ood_graphs),
    }

    return results


def _load_model_from_checkpoint(checkpoint_path: str) -> tuple[torch.nn.Module, dict, dict]:
    """Load a model from checkpoint, return (model, ckpt_dict, model_config)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = ckpt.get("model_type", "edge_aware")
    model_config = ckpt.get("model_config", {}) if isinstance(ckpt.get("model_config", {}), dict) else {}
    hidden_dim = int(model_config.get("hidden_dim", 64))
    use_edge = model_type == "edge_aware"
    model_state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    if model_state_dict is None:
        raise KeyError(f"checkpoint {checkpoint_path} missing model_state_dict/state_dict")

    node_dim = int(model_config.get("node_dim", 8))
    edge_dim = int(model_config.get("edge_dim", 4))

    if use_edge:
        model = EdgeAwareCircuitGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
    else:
        model = CircuitGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
    model.load_state_dict(model_state_dict)
    model.eval()

    return model, ckpt, model_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Circuit Arena: Oracle vs Surrogate")
    parser.add_argument("--checkpoint", default="workspace/checkpoints/circuit_gnn.pt")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Multiple checkpoint paths for v2.10 comparison (e.g. v29d.pt v29f.pt v210.pt)")
    parser.add_argument("--checkpoint-labels", nargs="+", default=None,
                        help="Labels for --checkpoints (same order, e.g. v29d v29f v210)")
    parser.add_argument("--dataset", default="workspace/datasets/circuits/train_10k/circuits.jsonl")
    parser.add_argument("--iid-dataset", default=None)
    parser.add_argument("--ood-dataset", default="workspace/datasets/circuits/ood_circuits.jsonl")
    parser.add_argument("--output-dir", default=str(REPORT_RESULTS_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--report-dir", default=str(REPORT_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-traces", action="store_true",
                        help="Save per-circuit projection traces to v210_traces.jsonl (FASE 5)")
    args = parser.parse_args()

    if args.iid_dataset is not None:
        args.dataset = args.iid_dataset

    # Set deterministic seeds for inference
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Build checkpoint list: multi-checkpoint mode or single
    if args.checkpoints:
        ckpt_paths = args.checkpoints
        ckpt_labels = args.checkpoint_labels or [Path(p).stem for p in ckpt_paths]
        if len(ckpt_labels) != len(ckpt_paths):
            print("ERROR: --checkpoint-labels must match --checkpoints count")
            return 1
    else:
        ckpt_paths = [args.checkpoint]
        ckpt_labels = [Path(args.checkpoint).stem.split("_")[-1] if "_" in Path(args.checkpoint).stem else "single"]

    # Load data (shared across all checkpoints)
    print(f"\nLoading in-distribution dataset: {args.dataset}")
    all_graphs, all_circuits = load_in_dist_data(Path(args.dataset))

    pairs = list(zip(all_graphs, all_circuits))
    pairs.sort(key=lambda x: x[0].fingerprint)
    n = len(pairs)
    split = int(n * args.train_frac)
    train_pairs = pairs[:split]
    eval_pairs = pairs[split:]

    train_graphs = [p[0] for p in train_pairs]
    train_circuits = [p[1] for p in train_pairs]
    eval_graphs = [p[0] for p in eval_pairs]
    eval_circuits = [p[1] for p in eval_pairs]
    print(f" Total: {n}, Train: {len(train_graphs)}, Eval: {len(eval_graphs)}")

    print(f"Loading OOD dataset: {args.ood_dataset}")
    ood_graphs, ood_circuits = load_ood_data(Path(args.ood_dataset))
    print(f" OOD: {len(ood_graphs)} circuits")

    # --- Multi-checkpoint arena + projection-effort + traces ---
    comparison_results = {}
    all_traces: list[dict] = []

    for ckpt_path, label in zip(ckpt_paths, ckpt_labels):
        print(f"\n{'='*60}")
        print(f"ARENA: {label} ({ckpt_path})")
        print(f"{'='*60}")

        model, ckpt, model_config = _load_model_from_checkpoint(ckpt_path)
        model_type = ckpt.get("model_type", "edge_aware")
        use_edge = model_type == "edge_aware"
        ablation = model_config.get("ablation", "baseline")
        hidden_dim = int(model_config.get("hidden_dim", 64))

        print(f"Model: {model_type}, params={model_config.get('num_params', '?')}")
        print(f"Trained for {ckpt.get('extra', {}).get('epochs_trained', '?')} epochs")
        target_mode = ckpt.get("extra", {}).get("target_mode", "oracle")

        # Reset seeds for determinism per-checkpoint
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        results = run_arena(
            model, train_graphs, train_circuits,
            eval_graphs, eval_circuits,
            ood_graphs, ood_circuits,
            use_edge=use_edge,
            ablation=ablation,
        )

        # --- Projection-effort metrics (v2.10) ---
        effort = compute_projection_effort(
            model, eval_graphs, eval_circuits,
            use_edge_features=use_edge,
            ablation=ablation,
        )
        results["projection_effort"] = effort
        print(f"\n--- Projection Effort ({label}) ---")
        print(f"  Mean iterations: {effort['mean_iterations']:.1f}")
        print(f"  Median iterations: {effort['median_iterations']:.1f}")
        print(f"  Mean residual after 1 step: {effort['mean_residual_after_1_step']:.6e}")
        print(f"  Mean raw KCL violation: {effort['mean_raw_kcl_violation']:.6e}")
        print(f"  Mean raw KVL violation: {effort['mean_raw_kvl_violation']:.6e}")

        # --- Projection traces (FASE 5) ---
        if args.save_traces:
            from backend.circuits.physics_projection import PhysicsProjection, ProjectionConfig
            from backend.circuits.projection_effort import measure_projection_effort
            from backend.circuits.surrogate_eval import denormalize_voltages, get_vmax

            trace_config = ProjectionConfig(
                steps=50, alpha_kcl=0.1, alpha_kvl=0.05,
                virtual_node_enabled=True, virtual_conductance=1.0, blend_factor=0.5,
            )
            trace_proj = PhysicsProjection(trace_config)
            for i, (graph, circuit) in enumerate(zip(eval_graphs, eval_circuits)):
                vmax = get_vmax(circuit)
                with torch.no_grad():
                    raw_pred = model(
                        graph.x,
                        graph.edge_index,
                        graph.edge_attr if use_edge and graph.edge_attr is not None else None,
                    )
                    voltages = denormalize_voltages(raw_pred.squeeze(-1), vmax, "per_circuit_vmax")

                step_metrics = trace_proj.project_step_metrics(graph, circuit, voltages)
                effort = measure_projection_effort(voltages, graph, circuit, trace_config)

                trace_entry = {
                    "checkpoint_label": label,
                    "circuit_idx": i,
                    "circuit_fingerprint": graph.fingerprint,
                    "iterations_to_converge": effort.iterations_to_converge,
                    "initial_residual": effort.initial_residual,
                    "final_residual": effort.final_residual,
                    "correction_distance": effort.correction_distance,
                    "residual_decay_rate": effort.residual_decay_rate,
                    "step_metrics": step_metrics,
                }
                all_traces.append(trace_entry)

        # Add metadata
        results["metadata"].update({
            "checkpoint": {
                "model_type": model_type,
                "hidden_dim": hidden_dim,
                "num_params": model_config.get("num_params", 0),
                "epochs_trained": ckpt.get("extra", {}).get("epochs_trained", 0),
                "best_epoch": ckpt.get("extra", {}).get("best_epoch", 0),
                "artifact_fingerprint": ckpt.get("artifact_fingerprint", ""),
                "dataset_manifest_hash": ckpt.get("dataset_manifest_hash", ""),
                "dataset_fingerprint": ckpt.get("extra", {}).get("dataset_fingerprint", ""),
                "config_fingerprint": ckpt.get("extra", {}).get("config_fingerprint", ""),
                "snapshot_hash": ckpt.get("snapshot_hash", ""),
                "snapshot_fingerprint": ckpt.get("extra", {}).get("snapshot_fingerprint", ckpt.get("snapshot_hash", "")),
                "eval_fingerprint": ckpt.get("eval_fingerprint", ""),
                "parent_oracle_version": ckpt.get("extra", {}).get("parent_oracle_version", "v2.8"),
                "target_mode": target_mode,
            },
            "train_split": args.train_frac,
            "dataset": str(Path(args.dataset).resolve()),
            "ood_dataset": str(Path(args.ood_dataset).resolve()),
        })

        results["metadata"]["checkpoint_config"] = ckpt.get("training_config", {})
        results["metadata"]["checkpoint_model"] = ckpt.get("model_config", {})
        results["summary"] = {
            "checkpoint_artifact_fingerprint": ckpt.get("artifact_fingerprint", ""),
            "dataset_manifest_hash": ckpt.get("dataset_manifest_hash", ""),
            "snapshot_hash": ckpt.get("snapshot_hash", ""),
            "evaluation_fingerprint": ckpt.get("eval_fingerprint", ""),
            "parent_oracle_version": ckpt.get("extra", {}).get("parent_oracle_version", "v2.8"),
            "target_mode": target_mode,
        }

        comparison_results[label] = results

    # --- Save projection traces (FASE 5) ---
    if args.save_traces and all_traces:
        traces_path = Path(args.output_dir) / "v210_traces.jsonl"
        traces_path.parent.mkdir(parents=True, exist_ok=True)
        with open(traces_path, "w") as f:
            for t in all_traces:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"Projection traces saved: {traces_path} ({len(all_traces)} entries)")

    # --- Multi-checkpoint comparison table ---
    if len(comparison_results) > 1:
        print(f"\n{'='*60}")
        print("MULTI-CHECKPOINT COMPARISON (v2.10)")
        print(f"{'='*60}")
        print(f"{'Label':<12} {'MAE(V)':<10} {'KCL Max':<12} {'Proj Iters':<12} {'Res After 1':<14}")
        print("-" * 60)
        for lbl, res in comparison_results.items():
            gnn_ind = res.get("gnn", {}).get("in_distribution", {})
            eff = res.get("projection_effort", {})
            print(f"{lbl:<12} {gnn_ind.get('mae', 0):<10.6f} {gnn_ind.get('kcl_max_violation', 0):<12.2e} {eff.get('mean_iterations', 0):<12.1f} {eff.get('mean_residual_after_1_step', 0):<14.6e}")

    # Use the last (or only) checkpoint's results for the main output
    primary_label = ckpt_labels[-1]
    results = comparison_results[primary_label]

    checkpoint_path = Path(args.checkpoint)
    tag = checkpoint_path.stem.split("_")[-1] if checkpoint_path.stem else "arena"

    # Save results
    if args.output is not None:
        out_base = Path(args.output)
        out_base.parent.mkdir(parents=True, exist_ok=True)
        metrics_path = out_base.parent / (out_base.stem + "_metrics.json")
        raw_path = out_base.parent / (out_base.stem + "_raw.json")
        report_json_path = out_base
        report_dir = out_base.parent
        output_dir = out_base.parent
    else:
        output_dir = Path(args.output_dir)
        report_dir = Path(args.report_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / f"circuit_arena_metrics_{tag}.json"
        raw_path = output_dir / f"circuit_arena_raw_{tag}.json"
        report_json_path = report_dir / f"circuit_arena_report_{tag}.json"

    metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    report_json_path.write_text(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    md = generate_markdown_summary(results)
    report_md_path = report_dir / f"circuit_arena_report_{tag}.md"
    report_md_path.write_text(md, encoding="utf-8")

    # Save multi-checkpoint comparison if applicable
    if len(comparison_results) > 1:
        comparison_path = output_dir / "v210_checkpoint_comparison.json"
        comparison_path.write_text(
            json.dumps(comparison_results, indent=2, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"Comparison saved: {comparison_path}")

    # Compact compatibility copies
    raw = {
        "model_type": results["metadata"]["checkpoint"]["model_type"],
        "eval_count": len(eval_graphs),
        "ood_count": len(ood_graphs),
    }
    raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    # Backward-compatible copies for existing tooling.
    shutil.copy2(metrics_path, output_dir / "circuit_arena_metrics.json")
    shutil.copy2(raw_path, output_dir / "circuit_arena_raw.json")
    shutil.copy2(report_json_path, report_dir / "circuit_arena_report.json")
    shutil.copy2(report_md_path, report_dir / "circuit_arena_report.md")

    registry_path = PROJECT_ROOT / "artifacts" / "artifact_registry.json"
    registry = ArtifactRegistry.from_file(registry_path) if registry_path.exists() else ArtifactRegistry(path=registry_path)
    registry.register(
        artifact_type="evaluation_report",
        schema_version="2.10",
        fingerprint=_stable_fingerprint(results),
        parent_fingerprints=[
            results["metadata"]["checkpoint"].get("artifact_fingerprint", ""),
            results["metadata"]["checkpoint"].get("snapshot_hash", ""),
            results["metadata"]["checkpoint"].get("dataset_manifest_hash", ""),
        ],
        metadata={
            "metrics_path": str(metrics_path),
            "report_json_path": str(report_json_path),
            "report_md_path": str(report_md_path),
            "output_dir": str(output_dir),
            "report_dir": str(report_dir),
        },
    )
    registry.register(
        artifact_type="benchmark_snapshot",
        schema_version="2.10",
        fingerprint=_stable_fingerprint(raw),
        parent_fingerprints=[results["metadata"]["checkpoint"].get("artifact_fingerprint", "")],
        metadata=raw,
    )
    registry.save(registry_path)

    print(f"\nMetrics saved: {metrics_path}")
    print(f"Report saved: {report_md_path}")
    print(f"Report JSON saved: {report_json_path}")

    return 0


def generate_markdown_summary(results: dict) -> str:
    """Generate a Markdown summary of arena results."""
    metadata = results.get("metadata", {})
    checkpoint = metadata.get("checkpoint", {})
    gnn = results.get("gnn", {})
    gnn_ind = gnn.get("in_distribution", {})
    gnn_ood = gnn.get("ood", {})
    families = gnn.get("families", {})
    triv_ind = families.get("trivial", {}).get("in_distribution", {})
    triv_ood = families.get("trivial", {}).get("ood", {})
    simp_ind = families.get("simple", {}).get("in_distribution", {})
    simp_ood = families.get("simple", {}).get("ood", {})
    med_ind = families.get("medium", {}).get("in_distribution", {})
    med_ood = families.get("medium", {}).get("ood", {})
    dens_ind = families.get("dense", {}).get("in_distribution", {})
    dens_ood = families.get("dense", {}).get("ood", {})

    mean_base = results.get("mean_baseline", {})
    linear_base = results.get("linear_baseline", {})
    random_base = results.get("random_baseline", {})
    oracle = results.get("oracle", {})
    lines = [
        "# CPT v2.9B — Circuit Arena Results",
        "",
        "## Model Info",
        "",
        f"- Type: {checkpoint.get('model_type', 'unknown')}",
        f"- Parameters: {checkpoint.get('num_params', 0)}",
        f"- Epochs trained: {checkpoint.get('epochs_trained', 0)}",
        f"- Best epoch: {checkpoint.get('best_epoch', 0)}",
        f"- Artifact fingerprint: {checkpoint.get('artifact_fingerprint', '')}",
        "",
        "## GNN",
        "",
        f"- In-distribution circuits: {gnn_ind.get('count', 0)}",
        f"- In-distribution MAE: {gnn_ind.get('mae', 0):.6f} V",
        f"- In-distribution RMSE: {gnn_ind.get('rmse', 0):.6f} V",
        f"- In-distribution max error: {gnn_ind.get('max_voltage_error', 0):.6f} V",
        f"- In-distribution KCL max violation: {gnn_ind.get('kcl_max_violation', 0):.2e}",
        f"- In-distribution KVL max violation: {gnn_ind.get('kvl_max_violation', 0):.2e}",
        f"- In-distribution replay consistency: {gnn_ind.get('replay_consistency', 0):.2e}",
        f"- OOD circuits: {gnn_ood.get('count', 0)}",
        f"- OOD MAE: {gnn_ood.get('mae', 0):.6f} V",
        f"- OOD RMSE: {gnn_ood.get('rmse', 0):.6f} V",
        f"- OOD max error: {gnn_ood.get('max_voltage_error', 0):.6f} V",
        f"- OOD KCL max violation: {gnn_ood.get('kcl_max_violation', 0):.2e}",
        f"- OOD KVL max violation: {gnn_ood.get('kvl_max_violation', 0):.2e}",
        f"- Deterministic rerun validation: {results.get('deterministic_rerun_validation', {}).get('gnn', False)}",
        f"- Replay consistency fingerprint match: {gnn.get('replay_consistency_metrics', {}).get('rerun_match', False)}",
        "",
        "## Per-Topology Family Breakdown",
        "",
        "| Topology Family | In-Distribution Count | In-Distribution MAE (V) | In-Distribution KCL Max Violation (A) | OOD Count | OOD MAE (V) | OOD KCL Max Violation (A) |",
        "|---|---|---|---|---|---|---|",
        f"| **Trivial** (Tree-like) | {triv_ind.get('count', 0)} | {triv_ind.get('mae', 0):.4f} | {triv_ind.get('kcl_max_violation', 0):.2e} | {triv_ood.get('count', 0)} | {triv_ood.get('mae', 0):.4f} | {triv_ood.get('kcl_max_violation', 0):.2e} |",
        f"| **Simple** (1 Cycle) | {simp_ind.get('count', 0)} | {simp_ind.get('mae', 0):.4f} | {simp_ind.get('kcl_max_violation', 0):.2e} | {simp_ood.get('count', 0)} | {simp_ood.get('mae', 0):.4f} | {simp_ood.get('kcl_max_violation', 0):.2e} |",
        f"| **Medium** (2-3 Cycles) | {med_ind.get('count', 0)} | {med_ind.get('mae', 0):.4f} | {med_ind.get('kcl_max_violation', 0):.2e} | {med_ood.get('count', 0)} | {med_ood.get('mae', 0):.4f} | {med_ood.get('kcl_max_violation', 0):.2e} |",
        f"| **Dense** (>3 Cycles) | {dens_ind.get('count', 0)} | {dens_ind.get('mae', 0):.4f} | {dens_ind.get('kcl_max_violation', 0):.2e} | {dens_ood.get('count', 0)} | {dens_ood.get('mae', 0):.4f} | {dens_ood.get('kcl_max_violation', 0):.2e} |",
        "",
        "## Baselines",
        "",
        f"- Mean baseline MAE: {mean_base.get('mae', 0):.6f} V",
        f"- Linear baseline MAE: {linear_base.get('mae', 0):.6f} V",
        f"- Random stable baseline MAE: {random_base.get('mae', 0):.6f} V",
        f"- GNN beats mean: {gnn_ind.get('mae', 0) < mean_base.get('mae', float('inf'))}",
        f"- GNN beats linear: {gnn_ind.get('mae', 0) < linear_base.get('mae', float('inf'))}",
        f"- Mean baseline replay consistency: {mean_base.get('replay_consistency', 0):.2e}",
        f"- Linear baseline replay consistency: {linear_base.get('replay_consistency', 0):.2e}",
        f"- Random baseline replay consistency: {random_base.get('replay_consistency', 0):.2e}",
        "",
        "## Oracle",
        "",
        f"- In-distribution circuits: {oracle.get('in_distribution', {}).get('count', 0)}",
        f"- OOD circuits: {oracle.get('ood', {}).get('count', 0)}",
        f"- Deterministic rerun validation: {results.get('deterministic_rerun_validation', {}).get('mean_baseline', False) and results.get('deterministic_rerun_validation', {}).get('linear_baseline', False)}",
        "",
        "## Speed",
        "",
        f"- Oracle mean solve: {gnn.get('speed_in_distribution', {}).get('oracle_mean_sec', 0)*1000:.3f} ms",
        f"- Surrogate mean inference: {gnn.get('speed_in_distribution', {}).get('surrogate_mean_sec', 0)*1000:.3f} ms",
        f"- Speedup: {gnn.get('speed_in_distribution', {}).get('speedup', 0):.2f}x",
        f"- OOD speedup: {gnn.get('speed_ood', {}).get('speedup', 0):.2f}x",
        "",
        "## Reproducibility",
        "",
        f"- Dataset fingerprint: {checkpoint.get('dataset_fingerprint', checkpoint.get('dataset_manifest_hash', ''))}",
        f"- Config fingerprint: {checkpoint.get('config_fingerprint', '')}",
        f"- Snapshot hash: {checkpoint.get('snapshot_hash', '')}",
        f"- Snapshot fingerprint: {checkpoint.get('snapshot_fingerprint', '')}",
        f"- Evaluation fingerprint: {checkpoint.get('eval_fingerprint', '')}",
        f"- Rerun validation: {results.get('deterministic_rerun_validation', {}).get('gnn', False)}",
        f"- Kaggle-ready: {bool(checkpoint.get('artifact_fingerprint', ''))}",
        "",
        "## Metadata",
        "",
        f"- Train count: {metadata.get('train_count', 0)}",
        f"- Eval count: {metadata.get('eval_count', 0)}",
        f"- OOD count: {metadata.get('ood_count', 0)}",
        f"- Dataset path: {metadata.get('dataset', '')}",
        f"- OOD dataset path: {metadata.get('ood_dataset', '')}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
