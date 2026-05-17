#!/usr/bin/env python3
"""Run CPT v2.9C surrogate validation analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.circuits.failure_analysis import classify_failure, compute_invariant_violations, summarize_failures
from backend.circuits.graph_dataset import CircuitGraph
from backend.circuits.invariants import validate_invariants
from backend.circuits.parser import parse_netlist
from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.surrogate_eval import denormalize_voltages, get_vmax
from backend.governance.artifact_registry import ArtifactRegistry
from scripts.run_circuit_arena import load_in_dist_data, load_ood_data, measure_speed
from scripts.train_circuit_gnn import (
    BASE_MODEL_TYPE,
    load_training_profile,
    resolve_dataset_path,
)
from backend.neural.models.circuit_gnn import CircuitGNN, EdgeAwareCircuitGNN
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training" / "kaggle_v29b.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "workspace" / "checkpoints" / "circuit_gnn_v29b.pt"
DEFAULT_DATASET = PROJECT_ROOT / "workspace" / "datasets" / "circuits" / "train_10k.jsonl"
DEFAULT_OOD_DATASET = PROJECT_ROOT / "workspace" / "datasets" / "circuits" / "ood_circuits.jsonl"
DETERMINISM_DIR = PROJECT_ROOT / "workspace" / "determinism_checks"
FAILURE_DIR = PROJECT_ROOT / "workspace" / "failure_analysis"
INVARIANT_DIR = PROJECT_ROOT / "workspace" / "invariant_validation"
SPEED_DIR = PROJECT_ROOT / "workspace" / "speed_validation"


def _stable_fingerprint(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_model(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = ckpt.get("model_type", BASE_MODEL_TYPE)
    model_config = ckpt.get("model_config", {}) if isinstance(ckpt.get("model_config", {}), dict) else {}
    hidden_dim = int(model_config.get("hidden_dim", 64))
    use_edge = model_type == "edge_aware"
    model_state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    if model_state_dict is None:
        raise KeyError("checkpoint missing model_state_dict/state_dict")
    model = EdgeAwareCircuitGNN(node_dim=8, edge_dim=4, hidden_dim=hidden_dim) if use_edge else CircuitGNN(node_dim=8, edge_dim=4, hidden_dim=hidden_dim)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model, ckpt, use_edge


def _predict_case(model: torch.nn.Module, graph: CircuitGraph, circuit, use_edge: bool, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        node_features = graph.node_features.to(device)
        edge_index = graph.edge_index.to(device)
        edge_features = graph.edge_features.to(device)
        if use_edge:
            pred_norm = model(node_features, edge_index, edge_features)
        else:
            pred_norm = model(node_features, edge_index)
        return denormalize_voltages(pred_norm, get_vmax(circuit)).detach().cpu()


def _benchmark_predicates(model, graphs, circuits, use_edge, device) -> dict[str, Any]:
    if not graphs:
        return {"count": 0, "kcl_max_violation": 0.0, "kvl_max_violation": 0.0, "power_conservation_violation": 0.0}
    kcl_values = []
    kvl_values = []
    power_values = []
    records = []
    for graph, circuit in zip(graphs, circuits):
        pred = _predict_case(model, graph, circuit, use_edge, device)
        invariants = compute_invariant_violations(circuit, graph.node_names, pred)
        kcl_values.append(float(invariants["kcl_max_violation"]))
        kvl_values.append(float(invariants["kvl_max_violation"]))
        power_values.append(float(invariants["power_conservation_violation"]))
        records.append(
            {
                "circuit_name": circuit.name,
                "kcl_max_violation": float(invariants["kcl_max_violation"]),
                "kvl_max_violation": float(invariants["kvl_max_violation"]),
                "power_conservation_violation": float(invariants["power_conservation_violation"]),
                "predicted_voltages": [float(v) for v in pred.tolist()],
            }
        )
    return {
        "count": len(records),
        "kcl_max_violation": max(kcl_values),
        "kvl_max_violation": max(kvl_values),
        "power_conservation_violation": max(power_values),
        "records": records,
    }


def _evaluate_split(model, graphs, circuits, use_edge, device, ood: bool = False):
    records = []
    for graph, circuit in zip(graphs, circuits):
        pred1 = _predict_case(model, graph, circuit, use_edge, device)
        pred2 = _predict_case(model, graph, circuit, use_edge, device)
        solution = solve_dc_circuit(circuit)
        oracle_tensor = torch.tensor([solution.node_voltages[name] for name in graph.node_names], dtype=torch.float32)
        diff = pred1 - oracle_tensor
        invariants = compute_invariant_violations(circuit, graph.node_names, pred1)
        classification = classify_failure(
            circuit,
            graph,
            pred1,
            oracle_tensor,
            invariant_metrics=invariants,
            ood=ood,
        )
        records.append(
            {
                "circuit_name": circuit.name,
                "fingerprint": graph.fingerprint,
                "mae": float(diff.abs().mean().item()),
                "rmse": float(diff.pow(2).mean().sqrt().item()),
                "max_error": float(diff.abs().max().item()),
                "replay_consistency": float((pred1 - pred2).abs().max().item()),
                "kcl_max_violation": float(invariants["kcl_max_violation"]),
                "kvl_max_violation": float(invariants["kvl_max_violation"]),
                "power_conservation_violation": float(invariants["power_conservation_violation"]),
                "classification": classification,
                "predicted_voltages": [float(v) for v in pred1.tolist()],
                "oracle_voltages": [float(v) for v in oracle_tensor.tolist()],
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v2.9C surrogate validation analysis.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--ood-dataset", default=str(DEFAULT_OOD_DATASET))
    parser.add_argument("--tag", default="v29c")
    parser.add_argument("--arena-results", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    profile = load_training_profile(args.config) if args.config else {}
    dataset_path = resolve_dataset_path(profile.get("dataset", {}).get("train_path", args.dataset))
    eval_split = float(profile.get("dataset", {}).get("eval_split", 0.2))
    train_frac = 1.0 - eval_split

    model, ckpt, use_edge = _load_model(Path(args.checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() and profile.get("device", {}).get("prefer_cuda", True) else "cpu")
    model = model.to(device)

    all_graphs, all_circuits = load_in_dist_data(Path(dataset_path))
    pairs = list(zip(all_graphs, all_circuits))
    pairs.sort(key=lambda x: x[0].fingerprint)
    train_pairs = pairs[: int(len(pairs) * train_frac)]
    eval_pairs = pairs[int(len(pairs) * train_frac) :]
    train_graphs = [p[0] for p in train_pairs]
    train_circuits = [p[1] for p in train_pairs]
    eval_graphs = [p[0] for p in eval_pairs]
    eval_circuits = [p[1] for p in eval_pairs]

    ood_graphs, ood_circuits = load_ood_data(Path(args.ood_dataset))
    ood_graphs = ood_graphs[:64]
    ood_circuits = ood_circuits[:64]

    sorted_pairs = sorted(list(zip(all_graphs, all_circuits)), key=lambda x: x[0].fingerprint)
    split = int(len(sorted_pairs) * train_frac)
    train_pairs = sorted_pairs[:split]
    eval_pairs = sorted_pairs[split:]
    train_graphs = [p[0] for p in train_pairs]
    train_circuits = [p[1] for p in train_pairs]
    eval_graphs = [p[0] for p in eval_pairs]
    eval_circuits = [p[1] for p in eval_pairs]

    iid_records_a = _evaluate_split(model, eval_graphs, eval_circuits, use_edge, device, ood=False)
    iid_records_b = _evaluate_split(model, eval_graphs, eval_circuits, use_edge, device, ood=False)
    ood_records_a = _evaluate_split(model, ood_graphs, ood_circuits, use_edge, device, ood=True)
    ood_records_b = _evaluate_split(model, ood_graphs, ood_circuits, use_edge, device, ood=True)

    determinism = {
        "run_a_fingerprint": _stable_fingerprint(iid_records_a + ood_records_a),
        "run_b_fingerprint": _stable_fingerprint(iid_records_b + ood_records_b),
        "metrics_equal": _stable_fingerprint({"iid": iid_records_a, "ood": ood_records_a}) == _stable_fingerprint({"iid": iid_records_b, "ood": ood_records_b}),
        "deterministic": iid_records_a == iid_records_b and ood_records_a == ood_records_b,
    }

    iid_inv = {
        "count": len(iid_records_a),
        "iid_kcl_violation": max((r["kcl_max_violation"] for r in iid_records_a), default=0.0),
        "iid_kvl_violation": max((r["kvl_max_violation"] for r in iid_records_a), default=0.0),
        "iid_power_violation": max((r["power_conservation_violation"] for r in iid_records_a), default=0.0),
    }
    ood_inv = {
        "count": len(ood_records_a),
        "ood_kcl_violation": max((r["kcl_max_violation"] for r in ood_records_a), default=0.0),
        "ood_kvl_violation": max((r["kvl_max_violation"] for r in ood_records_a), default=0.0),
        "ood_power_violation": max((r["power_conservation_violation"] for r in ood_records_a), default=0.0),
    }
    replay_inv = {
        "count": len(iid_records_a) + len(ood_records_a),
        "replay_max_abs_diff": max((r["replay_consistency"] for r in iid_records_a + ood_records_a), default=0.0),
        "replay_deterministic": determinism["deterministic"],
    }

    failure_summary = summarize_failures([r["classification"] for r in ood_records_a])
    baseline_metrics = {
        "model_type": ckpt.get("model_type", ""),
        "num_params": int(ckpt.get("model_config", {}).get("num_params", 0)) if isinstance(ckpt.get("model_config", {}), dict) else 0,
    }
    speed_graphs = eval_graphs[:1] or all_graphs[:1]
    speed_circuits = eval_circuits[:1] or all_circuits[:1]
    speed_metrics = measure_speed(model, speed_graphs, speed_circuits, use_edge, num_runs=1000)

    output = {
        "schema_version": "2.9c",
        "dataset_path": str(dataset_path),
        "ood_dataset_path": str(args.ood_dataset),
        "checkpoint_path": str(args.checkpoint),
        "determinism": determinism,
        "iid": {
            "records": iid_records_a,
            "summary": {
                "mae": max((r["mae"] for r in iid_records_a), default=0.0),
                "rmse": max((r["rmse"] for r in iid_records_a), default=0.0),
                "kcl_max_violation": iid_inv["iid_kcl_violation"],
                "kvl_max_violation": iid_inv["iid_kvl_violation"],
            },
        },
        "ood": {
            "records": ood_records_a,
            "summary": {
                "mae": max((r["mae"] for r in ood_records_a), default=0.0),
                "rmse": max((r["rmse"] for r in ood_records_a), default=0.0),
                "kcl_max_violation": ood_inv["ood_kcl_violation"],
                "kvl_max_violation": ood_inv["ood_kvl_violation"],
            },
        },
        "invariants": {
            "iid_kcl_violation": iid_inv["iid_kcl_violation"],
            "ood_kcl_violation": ood_inv["ood_kcl_violation"],
            "iid_kvl_violation": iid_inv["iid_kvl_violation"],
            "ood_kvl_violation": ood_inv["ood_kvl_violation"],
            "iid_power_violation": iid_inv["iid_power_violation"],
            "ood_power_violation": ood_inv["ood_power_violation"],
            "replay_max_abs_diff": replay_inv["replay_max_abs_diff"],
        },
        "replay": replay_inv,
        "failure_summary": failure_summary,
        "speed": speed_metrics,
        "model": baseline_metrics,
        "train_count": len(train_graphs),
        "eval_count": len(eval_graphs),
        "ood_count": len(ood_graphs),
    }
    output["analysis_fingerprint"] = _stable_fingerprint(output)

    determinism_dir = DETERMINISM_DIR
    failure_dir = FAILURE_DIR
    invariant_dir = INVARIANT_DIR
    speed_dir = SPEED_DIR
    for directory in (determinism_dir, failure_dir, invariant_dir, speed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    determinism_path = determinism_dir / f"{args.tag}_determinism.json"
    determinism_path.write_text(json.dumps(determinism, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    
    if args.output is not None:
        failure_path = Path(args.output)
        failure_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        failure_path = failure_dir / f"{args.tag}_failure_analysis.json"
        
    failure_path.write_text(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    invariant_path = invariant_dir / f"{args.tag}_invariants.json"
    invariant_path.write_text(json.dumps(output["invariants"], indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    speed_path = speed_dir / f"{args.tag}_speed.json"
    speed_path.write_text(json.dumps(speed_metrics, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# CPT v2.9C Failure Analysis",
        "",
        f"- Deterministic: {determinism['deterministic']}",
        f"- Run A fingerprint: {determinism['run_a_fingerprint']}",
        f"- Run B fingerprint: {determinism['run_b_fingerprint']}",
        "",
        "## Failure Summary",
        f"- Dominant failure: {failure_summary.get('dominant_failure', 'none')}",
        f"- OOD cases: {failure_summary.get('count', 0)}",
        "",
        "## Invariants",
        f"- IID KCL violation: {iid_inv['iid_kcl_violation']:.2e}",
        f"- OOD KCL violation: {ood_inv['ood_kcl_violation']:.2e}",
        f"- IID KVL violation: {iid_inv['iid_kvl_violation']:.2e}",
        f"- OOD KVL violation: {ood_inv['ood_kvl_violation']:.2e}",
        f"- IID power violation: {iid_inv['iid_power_violation']:.2e}",
        f"- OOD power violation: {ood_inv['ood_power_violation']:.2e}",
        "",
        "## Speed",
        f"- Oracle mean: {speed_metrics.get('oracle_mean_sec', 0):.6f} s",
        f"- Oracle p95: {speed_metrics.get('oracle_p95_sec', 0):.6f} s",
        f"- Surrogate mean: {speed_metrics.get('surrogate_mean_sec', 0):.6f} s",
        f"- Surrogate p95: {speed_metrics.get('surrogate_p95_sec', 0):.6f} s",
        f"- Speedup: {speed_metrics.get('speedup', 0):.2f}x",
    ]
    (failure_dir / f"{args.tag}_failure_analysis.md").write_text("\n".join(md_lines), encoding="utf-8")

    registry_path = PROJECT_ROOT / "artifacts" / "artifact_registry.json"
    registry = ArtifactRegistry.from_file(registry_path) if registry_path.exists() else ArtifactRegistry(path=registry_path)
    registry.register(
        artifact_type="evaluation_report",
        schema_version="2.9d" if args.tag == "v29d" else "2.9c",
        fingerprint=output["analysis_fingerprint"],
        parent_fingerprints=[
            ckpt.get("artifact_fingerprint", ""),
            output["determinism"]["run_a_fingerprint"],
            output["determinism"]["run_b_fingerprint"],
        ],
        metadata={
            "failure_path": str(failure_path),
            "invariant_path": str(invariant_path),
            "speed_path": str(speed_path),
        },
    )
    registry.save(registry_path)

    print(json.dumps({
        "determinism_path": str(determinism_path),
        "failure_path": str(failure_path),
        "invariant_path": str(invariant_path),
        "speed_path": str(speed_path),
        "analysis_fingerprint": output["analysis_fingerprint"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
