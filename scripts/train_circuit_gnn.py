#!/usr/bin/env python3
"""Deterministic training loop for CircuitGNN surrogate.

Trains a tiny GNN to predict DC circuit node voltages.
All seeds are fixed for full reproducibility.

Key design: per-circuit voltage normalization.
- Each circuit's node voltages are divided by V_max (max |voltage source|)
- Model predicts normalized voltages in roughly [0, 1]
- At inference, predictions are rescaled by V_max
- This makes the learning problem well-conditioned regardless of absolute voltage
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.graph_dataset import CircuitGraph, circuit_to_graph
from backend.circuits.models import Circuit
from backend.circuits.physics_loss import PhysicsInformedLoss
from backend.circuits.parser import parse_netlist
from backend.governance.artifact_registry import ArtifactRegistry
from backend.neural.checkpoints.fingerprint import hash_optimizer_state, hash_state_dict
from backend.neural.checkpoints.schema import build_checkpoint_payload
from backend.neural.checkpoints.validator import validate_checkpoint_payload
from backend.neural.training_snapshot import TrainingSnapshot, fingerprint_jsonl, fingerprint_mapping
from backend.neural.models.circuit_gnn import CircuitGNN, EdgeAwareCircuitGNN
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "kaggle_v29b.yaml"
DEFAULT_DATASET = "workspace/datasets/circuits/train_10k/circuits.jsonl"
DEFAULT_OUTPUT = "workspace/checkpoints/circuit_gnn_v29d.pt"
DEFAULT_OOD_DATASET = "workspace/datasets/circuits/ood_circuits.jsonl"
BASE_EPOCHS = 20
BASE_LR = 5e-4
BASE_WEIGHT_DECAY = 1e-5
BASE_HIDDEN_DIM = 96
BASE_SEED = 42
BASE_MODEL_TYPE = "edge_aware"
BASE_TRAIN_FRAC = 0.8
BASE_BATCH_SIZE = 32


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _fingerprint_payload(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def load_training_profile(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Training profile must be a mapping")
    return payload


def resolve_dataset_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    fallback = path.parent / path.stem / "circuits.jsonl"
    if fallback.exists():
        return fallback
    if path.exists():
        return path
    alt = path.with_name("circuits.jsonl")
    if alt.exists():
        return alt
    return path


def deterministic_created_at(seed: int, epochs: int, sample_count: int) -> float:
    return float(seed) + (epochs / 100.0) + (sample_count / 100000.0)


def _process_memory_mb() -> float | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = float(usage.ru_maxrss)
        if platform.system().lower() == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        return None


def _print_model_size(model: torch.nn.Module, max_params: int) -> int:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    estimated_checkpoint_size_mb = total_params * 4.0 / (1024.0 * 1024.0)
    print("=" * 60)
    print("MODEL PARAMETER COUNT")
    print(f"total_params: {total_params}")
    print(f"trainable_params: {trainable_params}")
    print(f"estimated_checkpoint_size_mb: {estimated_checkpoint_size_mb:.2f}")
    print("=" * 60)
    if total_params > max_params:
        raise ValueError(f"model parameter count {total_params} exceeds max_params={max_params}")
    return total_params


def _merge_profile(args: argparse.Namespace, profile: dict[str, Any]) -> dict[str, Any]:
    seed = int(profile.get("seed", BASE_SEED))
    device_cfg = dict(profile.get("device", {}))
    training_cfg = dict(profile.get("training", {}))
    dataset_cfg = dict(profile.get("dataset", {}))
    model_cfg = dict(profile.get("model", {}))
    eval_cfg = dict(profile.get("evaluation", {}))

    config = {
        "seed": seed,
        "device": {"prefer_cuda": bool(device_cfg.get("prefer_cuda", True))},
        "training": {
            "epochs": int(training_cfg.get("epochs", BASE_EPOCHS)),
            "batch_size": int(training_cfg.get("batch_size", BASE_BATCH_SIZE)),
            "learning_rate": float(training_cfg.get("learning_rate", BASE_LR)),
        },
        "dataset": {
            "train_path": str(dataset_cfg.get("train_path", DEFAULT_DATASET)),
            "eval_split": float(dataset_cfg.get("eval_split", BASE_TRAIN_FRAC)),
        },
        "model": {
            "hidden_dim": int(model_cfg.get("hidden_dim", BASE_HIDDEN_DIM)),
            "max_params": int(model_cfg.get("max_params", 250_000)),
        },
        "evaluation": {
            "voltage_tolerance": float(eval_cfg.get("voltage_tolerance", 1e-3)),
            "invariant_tolerance": float(eval_cfg.get("invariant_tolerance", 1e-6)),
        },
        "output": str(args.output or DEFAULT_OUTPUT),
        "model_type": str(args.model_type or BASE_MODEL_TYPE),
    }
    if args.dataset is not None:
        config["dataset"]["train_path"] = args.dataset
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.hidden_dim is not None:
        config["model"]["hidden_dim"] = args.hidden_dim
    if args.train_frac is not None:
        config["dataset"]["eval_split"] = 1.0 - float(args.train_frac)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.weight_decay is not None:
        config["training"]["weight_decay"] = args.weight_decay
    return config


def set_deterministic_seeds(seed: int = 42) -> None:
    """Set all random seeds for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- Per-circuit voltage normalization ---

def get_vmax(circuit: Circuit) -> float:
    """Get max absolute voltage source value for normalization."""
    if circuit.voltage_sources:
        return max(abs(vs.voltage) for vs in circuit.voltage_sources)
    return 1.0


def normalize_voltages(voltages: torch.Tensor, vmax: float) -> torch.Tensor:
    """Normalize voltages by vmax. Returns values in roughly [-1, 1]."""
    return voltages / max(vmax, 1e-12)


def denormalize_voltages(normalized: torch.Tensor, vmax: float) -> torch.Tensor:
    """Denormalize predictions back to voltage space."""
    return normalized * vmax


@dataclass(frozen=True)
class TrainingGraph:
    """Graph with pre-computed normalization factor."""
    graph: CircuitGraph
    circuit: Circuit
    vmax: float


def _is_connected_circuit(circuit: Circuit) -> bool:
    """Check if all non-ground nodes are connected to ground via some path."""
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


def load_training_data(
    jsonl_path: Path, max_voltage: float = 1e6
) -> List[TrainingGraph]:
    """Load a JSONL dataset and convert to TrainingGraphs."""
    graphs: List[TrainingGraph] = []
    skipped_disconnected = 0
    skipped_extreme = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            netlist = row["netlist"]
            name = row.get("circuit_name", "unnamed")
            try:
                circuit = parse_netlist(netlist, name=name)
                if not _is_connected_circuit(circuit):
                    skipped_disconnected += 1
                    continue
                solution = solve_dc_circuit(circuit)
                max_abs_v = max((abs(v) for v in solution.node_voltages.values()), default=0.0)
                if max_abs_v > max_voltage:
                    skipped_extreme += 1
                    continue
                g = circuit_to_graph(circuit, solution)
                if g.node_features.size(0) > 0:
                    vmax = get_vmax(circuit)
                    graphs.append(TrainingGraph(graph=g, circuit=circuit, vmax=vmax))
            except Exception:
                continue

    print(f"Loaded {len(graphs)} graphs (skipped {skipped_disconnected} disconnected, "
          f"{skipped_extreme} extreme voltage)")
    return graphs


def deterministic_split(
    data: List[TrainingGraph], train_frac: float = 0.8
) -> Tuple[List[TrainingGraph], List[TrainingGraph]]:
    """Deterministic train/eval split based on sorted fingerprints."""
    sorted_data = sorted(data, key=lambda d: d.graph.fingerprint)
    n = len(sorted_data)
    split = int(n * train_frac)
    return sorted_data[:split], sorted_data[split:]


def train_one_epoch(
    model: torch.nn.Module,
    train_data: List[TrainingGraph],
    optimizer: torch.optim.Optimizer,
    loss_fn: PhysicsInformedLoss,
    use_edge_features: bool = False,
) -> Dict[str, float]:
    """Train one epoch, processing one graph at a time for determinism."""
    model.train()
    total_loss = 0.0
    total_voltage = 0.0
    total_kcl = 0.0
    total_kvl = 0.0
    total_power = 0.0
    count = 0
    device = next(model.parameters()).device

    sorted_data = sorted(train_data, key=lambda d: d.graph.fingerprint)

    for td in sorted_data:
        g = td.graph
        if g.node_features.size(0) == 0:
            continue

        optimizer.zero_grad()
        node_features = g.node_features.to(device)
        edge_index = g.edge_index.to(device)
        edge_features = g.edge_features.to(device)
        target_voltages = g.target_voltages.to(device)

        if use_edge_features:
            pred = model(node_features, edge_index, edge_features)
        else:
            pred = model(node_features, edge_index)

        pred_voltage = denormalize_voltages(pred, td.vmax)
        loss_voltage = loss_fn.compute_voltage_loss(pred_voltage, target_voltages)
        loss_kcl = loss_fn.compute_kcl_loss(pred_voltage, g, circuit=td.circuit)
        loss_kvl = loss_fn.compute_kvl_loss(pred_voltage, g)
        loss_power = loss_fn.compute_power_loss(pred_voltage, g, circuit=td.circuit)
        loss = loss_voltage + loss_fn.lambda_kcl * loss_kcl + loss_fn.lambda_kvl * loss_kvl + loss_fn.lambda_power * loss_power
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_voltage += loss_voltage.item()
        total_kcl += loss_kcl.item()
        total_kvl += loss_kvl.item()
        total_power += loss_power.item()
        count += 1

    n = max(count, 1)
    return {
        "total_loss": total_loss / n,
        "voltage_loss": total_voltage / n,
        "kcl_loss": total_kcl / n,
        "kvl_loss": total_kvl / n,
        "power_loss": total_power / n,
        "count": count,
    }


def evaluate(
    model: torch.nn.Module,
    eval_data: List[TrainingGraph],
    loss_fn: PhysicsInformedLoss,
    use_edge_features: bool = False,
) -> Dict[str, float]:
    """Evaluate model on a set of graphs. Returns metrics in VOLTAGE space."""
    model.eval()
    device = next(model.parameters()).device
    total_mae = 0.0
    total_rmse = 0.0
    total_max_err = 0.0
    total_voltage_loss = 0.0
    total_kcl = 0.0
    total_kvl = 0.0
    total_power = 0.0
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for td in eval_data:
            g = td.graph
            if g.node_features.size(0) == 0:
                continue

            node_features = g.node_features.to(device)
            edge_index = g.edge_index.to(device)
            edge_features = g.edge_features.to(device)
            true_voltage = g.target_voltages.to(device)

            if use_edge_features:
                pred_norm = model(node_features, edge_index, edge_features)
            else:
                pred_norm = model(node_features, edge_index)

            pred_voltage = denormalize_voltages(pred_norm, td.vmax)

            diff = pred_voltage - true_voltage
            mae = diff.abs().mean().item()
            rmse = diff.pow(2).mean().sqrt().item()
            max_err = diff.abs().max().item()

            v_loss = loss_fn.compute_voltage_loss(pred_voltage, true_voltage).item()
            kcl = loss_fn.compute_kcl_loss(pred_voltage, g, circuit=td.circuit).item()
            kvl = loss_fn.compute_kvl_loss(pred_voltage, g).item()
            power = loss_fn.compute_power_loss(pred_voltage, g, circuit=td.circuit).item()
            total = v_loss + (loss_fn.lambda_kcl * kcl) + (loss_fn.lambda_kvl * kvl) + (loss_fn.lambda_power * power)

            total_mae += mae
            total_rmse += rmse
            total_max_err = max(total_max_err, max_err)
            total_voltage_loss += v_loss
            total_kcl += kcl
            total_kvl += kvl
            total_power += power
            total_loss += total
            count += 1

    n = max(count, 1)
    return {
        "loss": total_loss / n,
        "voltage_loss": total_voltage_loss / n,
        "kcl_penalty": total_kcl / n,
        "kvl_penalty": total_kvl / n,
        "power_penalty": total_power / n,
        "mae": total_mae / n,
        "rmse": total_rmse / n,
        "max_error": total_max_err,
        "count": count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CircuitGNN surrogate.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-type", choices=["basic", "edge_aware"], default=None)
    parser.add_argument("--train-frac", type=float, default=None)
    parser.add_argument("--lambda-kcl", type=float, default=None)
    parser.add_argument("--lambda-kvl", type=float, default=None)
    parser.add_argument("--lambda-power", type=float, default=None)
    args = parser.parse_args()

    profile = load_training_profile(args.config) if args.config else {}
    config = _merge_profile(args, profile)
    config["physics"] = {
        "lambda_kcl": float(args.lambda_kcl if args.lambda_kcl is not None else 5.0),
        "lambda_kvl": float(args.lambda_kvl if args.lambda_kvl is not None else 5.0),
        "lambda_power": float(args.lambda_power if args.lambda_power is not None else 1.0),
    }
    seed = int(config["seed"])
    dataset_path = resolve_dataset_path(config["dataset"]["train_path"])
    output_path = Path(config.get("output", DEFAULT_OUTPUT))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    set_deterministic_seeds(seed)

    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}")
        return 1

    print(f"Training profile: {args.config}")
    print(f"Resolved dataset: {dataset_path}")
    print(f"Output checkpoint: {output_path}")

    print(f"Loading dataset: {dataset_path}")
    all_data = load_training_data(dataset_path)

    if not all_data:
        print("ERROR: No valid graphs loaded")
        return 1

    # Stats
    all_v = torch.cat([d.graph.target_voltages for d in all_data])
    print(f"Voltage stats: mean={all_v.mean():.4f}, std={all_v.std():.4f}, "
          f"min={all_v.min():.4f}, max={all_v.max():.4f}")
    all_vmax = torch.tensor([d.vmax for d in all_data])
    print(f"Vmax stats: mean={all_vmax.mean():.4f}, std={all_vmax.std():.4f}")
    all_norm = torch.cat([normalize_voltages(d.graph.target_voltages, d.vmax) for d in all_data])
    print(f"Normalized voltage stats: mean={all_norm.mean():.4f}, std={all_norm.std():.4f}, "
          f"min={all_norm.min():.4f}, max={all_norm.max():.4f}")

    eval_split = float(config["dataset"]["eval_split"])
    train_frac = 1.0 - eval_split
    train_data, eval_data = deterministic_split(all_data, train_frac)
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Create model
    model_type = args.model_type or BASE_MODEL_TYPE
    use_edge = model_type == "edge_aware"
    hidden_dim = int(config["model"]["hidden_dim"])
    max_params = int(config["model"]["max_params"])
    if use_edge:
        model = EdgeAwareCircuitGNN(
            node_dim=8,
            edge_dim=4,
            hidden_dim=hidden_dim,
        )
    else:
        model = CircuitGNN(
            node_dim=8,
            edge_dim=4,
            hidden_dim=hidden_dim,
        )

    n_params = _print_model_size(model, max_params)
    print(f"Model: {model_type}, Parameters: {n_params}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    device = torch.device("cuda" if config["device"]["prefer_cuda"] and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    loss_fn = PhysicsInformedLoss(
        lambda_kcl=float(config["physics"]["lambda_kcl"]),
        lambda_kvl=float(config["physics"]["lambda_kvl"]),
        lambda_power=float(config["physics"]["lambda_power"]),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", BASE_WEIGHT_DECAY)),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_eval_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    best_optimizer_state = None
    best_eval_metrics: Dict[str, float] = {}
    history = []
    start_time = time.perf_counter()

    print(f"\nTraining for {int(config['training']['epochs'])} epochs...")
    for epoch in range(int(config["training"]["epochs"])):
        t0 = time.perf_counter()
        train_metrics = train_one_epoch(model, train_data, optimizer, loss_fn, use_edge_features=use_edge)
        train_elapsed = time.perf_counter() - t0
        eval_t0 = time.perf_counter()
        eval_metrics = evaluate(model, eval_data, loss_fn, use_edge_features=use_edge)
        eval_elapsed = time.perf_counter() - eval_t0
        scheduler.step(eval_metrics["loss"])
        total_elapsed = time.perf_counter() - start_time
        samples_sec = len(train_data) / max(train_elapsed, 1e-12)

        lr_now = optimizer.param_groups[0]["lr"]
        entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_metrics["total_loss"], 6),
            "train_voltage_loss": round(train_metrics["voltage_loss"], 6),
            "train_kcl_loss": round(train_metrics["kcl_loss"], 9),
            "train_kvl_loss": round(train_metrics["kvl_loss"], 9),
            "train_power_loss": round(train_metrics["power_loss"], 9),
            "eval_loss": round(eval_metrics["loss"], 6),
            "eval_voltage_loss": round(eval_metrics["voltage_loss"], 6),
            "eval_kcl_penalty": round(eval_metrics["kcl_penalty"], 9),
            "eval_kvl_penalty": round(eval_metrics["kvl_penalty"], 9),
            "eval_power_penalty": round(eval_metrics["power_penalty"], 9),
            "eval_mae_V": round(eval_metrics["mae"], 6),
            "eval_rmse_V": round(eval_metrics["rmse"], 6),
            "eval_max_error_V": round(eval_metrics["max_error"], 6),
            "lr": round(lr_now, 8),
            "train_time_sec": round(train_elapsed, 6),
            "eval_time_sec": round(eval_elapsed, 6),
        }
        history.append(entry)
        print(
            f"Epoch {epoch+1:3d}/{int(config['training']['epochs'])}: "
            f"train_loss={entry['train_loss']:.6f} "
            f"train_v={entry['train_voltage_loss']:.6f} "
            f"train_kcl={entry['train_kcl_loss']:.2e} "
            f"train_kvl={entry['train_kvl_loss']:.2e} "
            f"train_p={entry['train_power_loss']:.2e} "
            f"eval_loss={entry['eval_loss']:.6f} "
            f"eval_v={entry['eval_voltage_loss']:.6f} "
            f"eval_kcl={entry['eval_kcl_penalty']:.2e} "
            f"eval_kvl={entry['eval_kvl_penalty']:.2e} "
            f"eval_p={entry['eval_power_penalty']:.2e} "
            f"eval_mae={entry['eval_mae_V']:.4f}V "
            f"eval_rmse={entry['eval_rmse_V']:.4f}V "
            f"eval_max={entry['eval_max_error_V']:.4f}V "
            f"lr={lr_now:.2e} "
            f"{samples_sec:.1f} samples/s "
            f"train_t={train_elapsed:.1f}s "
            f"eval_t={eval_elapsed:.1f}s"
        )

        if eval_metrics["loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["loss"]
            best_epoch = epoch + 1
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_eval_metrics = dict(eval_metrics)

    if best_state_dict is None:
        print("ERROR: training did not produce a valid checkpoint")
        return 1

    model.load_state_dict(best_state_dict)
    model.eval()

    final_eval_metrics = evaluate(model, eval_data, loss_fn, use_edge_features=use_edge)
    final_total_elapsed = time.perf_counter() - start_time
    final_samples_sec = len(train_data) * int(config["training"]["epochs"]) / max(final_total_elapsed, 1e-12)

    dataset_fingerprint = fingerprint_jsonl(dataset_path)
    config_fingerprint = fingerprint_mapping({k: v for k, v in config.items() if k != "output"})
    model_fingerprint = hash_state_dict(best_state_dict)
    snapshot = TrainingSnapshot.create(
        seed=seed,
        dataset_fingerprint=dataset_fingerprint,
        config=config,
        model_fingerprint=model_fingerprint,
        repo_root=PROJECT_ROOT,
        cuda_enabled=torch.cuda.is_available(),
        device_name=str(device),
    )
    snapshot_path = PROJECT_ROOT / "workspace" / "training_snapshots" / "training_snapshot.json"
    snapshot_payload = snapshot.to_dict()
    snapshot_payload.update(
        {
            "dataset_path": str(config["dataset"]["train_path"]),
            "resolved_dataset_path": str(dataset_path.resolve()),
            "config_path": str(args.config) if args.config else "",
            "resolved_config_path": str(Path(args.config).resolve()) if args.config else "",
            "eval_fingerprint": _fingerprint_payload(best_eval_metrics or final_eval_metrics),
            "evaluation_fingerprint": _fingerprint_payload(best_eval_metrics or final_eval_metrics),
            "parent_oracle_version": "v2.8",
            "checkpoint_fingerprint": _fingerprint_payload({"weights_hash": model_fingerprint, "snapshot": snapshot.fingerprint()}),
            "best_epoch": best_epoch,
        }
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot_payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    created_at = deterministic_created_at(seed, int(config["training"]["epochs"]), len(all_data))
    fingerprint_history = [
        {key: value for key, value in entry.items() if key not in {"train_time_sec", "eval_time_sec"}}
        for entry in history
    ]
    eval_fingerprint = _fingerprint_payload(
        {
            "dataset_fingerprint": dataset_fingerprint,
            "config_fingerprint": config_fingerprint,
            "best_epoch": best_epoch,
            "eval_metrics": best_eval_metrics or final_eval_metrics,
            "history": fingerprint_history,
        }
    )
    checkpoint_payload = build_checkpoint_payload(
        model_type=model_type,
        model_config={
            "node_dim": 8,
            "edge_dim": 4,
            "hidden_dim": hidden_dim,
            "max_params": max_params,
            "use_edge_features": use_edge,
            "num_params": n_params,
        },
        training_config={
            "seed": seed,
            "config_path": str(args.config) if args.config else "",
            "dataset_path": str(config["dataset"]["train_path"]),
            "epochs": int(config["training"]["epochs"]),
            "batch_size": int(config["training"]["batch_size"]),
            "learning_rate": float(config["training"]["learning_rate"]),
            "weight_decay": float(config["training"].get("weight_decay", BASE_WEIGHT_DECAY)),
            "train_frac": train_frac,
            "eval_split": eval_split,
            "device": str(device),
            "model_type": model_type,
            "lambda_kcl": float(config["physics"]["lambda_kcl"]),
            "lambda_kvl": float(config["physics"]["lambda_kvl"]),
            "lambda_power": float(config["physics"]["lambda_power"]),
        },
        dataset_manifest_hash=dataset_fingerprint,
        snapshot_hash=snapshot.fingerprint(),
        weights_hash=model_fingerprint,
        optimizer_state_hash=hash_optimizer_state(best_optimizer_state),
        eval_fingerprint=eval_fingerprint,
        curriculum_coverage={
            "dataset_count": len(all_data),
            "train_count": len(train_data),
            "eval_count": len(eval_data),
            "best_epoch": best_epoch,
        },
        seed=seed,
        created_at=created_at,
        state_dict=best_state_dict,
        optimizer_state=best_optimizer_state,
        extra={
            "history": history,
            "best_epoch": best_epoch,
            "epochs_trained": int(config["training"]["epochs"]),
            "num_params": n_params,
            "best_eval_metrics": best_eval_metrics or final_eval_metrics,
            "final_eval_metrics": final_eval_metrics,
            "dataset_fingerprint": dataset_fingerprint,
            "config_fingerprint": config_fingerprint,
            "physics": dict(config["physics"]),
            "snapshot_fingerprint": snapshot.fingerprint(),
            "parent_oracle_version": "v2.8",
        },
    )
    errors = validate_checkpoint_payload(checkpoint_payload, allow_legacy=False)
    if errors:
        print("ERROR: checkpoint validation failed")
        for error in errors:
            print(f"  - {error}")
        return 1

    torch.save(checkpoint_payload, output_path)

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_payload = {
        "schema_version": "2.9d",
        "model_type": model_type,
        "device": str(device),
        "num_params": n_params,
        "seed": seed,
        "dataset_fingerprint": dataset_fingerprint,
        "config_fingerprint": config_fingerprint,
        "physics": dict(config["physics"]),
        "snapshot_fingerprint": snapshot.fingerprint(),
        "checkpoint_fingerprint": checkpoint_payload["artifact_fingerprint"],
        "best_epoch": best_epoch,
        "history": history,
        "best_eval_metrics": best_eval_metrics or final_eval_metrics,
        "final_eval_metrics": final_eval_metrics,
        "train_count": len(train_data),
        "eval_count": len(eval_data),
        "parent_oracle_version": "v2.8",
        "reproducible": True,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    registry_path = PROJECT_ROOT / "artifacts" / "artifact_registry.json"
    registry = ArtifactRegistry.from_file(registry_path) if registry_path.exists() else ArtifactRegistry(path=registry_path)
    registry.register(
        artifact_type="checkpoint",
        schema_version="2.7.6",
        fingerprint=checkpoint_payload["artifact_fingerprint"],
        parent_fingerprints=[dataset_fingerprint, config_fingerprint, snapshot.fingerprint(), eval_fingerprint],
        metadata={
            "path": str(output_path),
            "model_type": model_type,
            "best_epoch": best_epoch,
            "dataset_path": str(dataset_path.resolve()),
        },
    )
    registry.register(
        artifact_type="training_snapshot",
        schema_version="2.9b",
        fingerprint=snapshot.fingerprint(),
        parent_fingerprints=[dataset_fingerprint, config_fingerprint, model_fingerprint, eval_fingerprint, "v2.8"],
        metadata=snapshot_payload,
    )
    registry.register(
        artifact_type="evaluation_report",
        schema_version="2.9d",
        fingerprint=_fingerprint_payload(metrics_payload),
        parent_fingerprints=[checkpoint_payload["artifact_fingerprint"], snapshot.fingerprint()],
        metadata=metrics_payload,
    )
    registry.save(registry_path)

    print(f"\nTraining complete. Best eval loss: {best_eval_loss:.6f} at epoch {best_epoch}")
    print(f"Final eval MAE: {final_eval_metrics['mae']:.6f} V")
    print(f"Checkpoint saved to: {output_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Training snapshot saved to: {snapshot_path}")
    print(f"Total training duration: {final_total_elapsed:.2f}s")
    print(f"Samples/sec: {len(train_data) * int(config['training']['epochs']) / max(final_total_elapsed, 1e-12):.2f}")
    memory_mb = _process_memory_mb()
    if memory_mb is not None:
        print(f"Peak RSS MB: {memory_mb:.2f}")
    checkpoint_size_mb = output_path.stat().st_size / (1024.0 * 1024.0)
    print(f"Checkpoint size MB: {checkpoint_size_mb:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
