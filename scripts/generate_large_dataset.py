#!/usr/bin/env python3
"""Large deterministic oracle dataset generator with shard/resume support."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from backend.core_truth.sandbox import sandbox_manager
from backend.datasets.manifest import DatasetManifest
from backend.governance.artifact_policy import ArtifactPolicy, load_artifact_policy
from backend.datasets.oracle_generator import (
    _build_trace_export,
    _default_question,
    _equations_from_rule,
    _load_modules_registry,
    _module_executable_rule,
    _stable_hash,
)
from backend.verifiers import verify_simulation

DATASET_VERSION = "2.7.0"
STATE_FILE = "generation_state.json"
SHARD_MANIFEST_FILE = "shard_manifest.json"
DATASET_MANIFEST_FILE = "dataset.manifest.json"


@dataclass(frozen=True)
class GenerationConfig:
    seed: int
    output_dir: Path
    num_samples: int
    shard_size: int = 10000
    layer_weights: dict[int, float] | None = None
    module_filter: list[str] | None = None
    layer_filter: list[int] | None = None
    ood_ratio: float = 0.0
    resume: bool = False
    modules_path: Path = Path("backend/core_truth/modules.json")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _module_items(
    registry: dict[str, Any],
    module_filter: list[str] | None = None,
    layer_filter: list[int] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    modules = registry.get("modules", {})
    items = list(modules.items())
    if module_filter is not None:
        keyset = set(module_filter)
        items = [item for item in items if item[0] in keyset]
    if layer_filter is not None:
        layers = set(int(layer) for layer in layer_filter)
        items = [item for item in items if int(item[1].get("level", -1)) in layers]
    return sorted(items, key=lambda item: (int(item[1].get("level", 0)), int(item[1].get("order", 0) or 0), item[0]))


def _module_weight(module: dict[str, Any], layer_weights: dict[int, float] | None) -> float:
    layer = int(module.get("level", 0))
    if layer_weights and layer in layer_weights:
        return float(layer_weights[layer])
    return 1.0


def _weighted_cycle(items: list[tuple[str, dict[str, Any]]], layer_weights: dict[int, float] | None) -> list[str]:
    if not items:
        return []
    weights = {key: max(_module_weight(module, layer_weights), 0.0) for key, module in items}
    if all(weight == 0.0 for weight in weights.values()):
        weights = {key: 1.0 for key in weights}

    order: list[str] = []
    totals = {key: 0.0 for key in weights}
    total_weight = sum(weights.values())
    item_order = [key for key, _module in items]
    for _ in range(max(1, len(items) * 16)):
        for key in weights:
            totals[key] += weights[key]
        chosen = max(item_order, key=lambda key: (totals[key], -item_order.index(key)))
        totals[chosen] -= total_weight
        order.append(chosen)
    return order


def iter_balanced_tasks(cfg: GenerationConfig) -> Iterator[dict[str, Any]]:
    """Yield deterministic generation tasks respecting layer/module weights."""

    registry = _load_modules_registry(cfg.modules_path)
    items = _module_items(registry, cfg.module_filter, cfg.layer_filter)
    if not items:
        return

    weight_order = _weighted_cycle(items, cfg.layer_weights)
    if not weight_order:
        weight_order = [key for key, _ in items]

    module_map = {key: module for key, module in items}
    for sample_index in range(cfg.num_samples):
        module_key = weight_order[sample_index % len(weight_order)]
        module = module_map[module_key]
        parameters = {}
        if cfg.ood_ratio > 0.0 and (sample_index % max(1, int(round(1.0 / cfg.ood_ratio))) == 0):
            parameters = {"ood_marker": (cfg.seed + sample_index) % 7}
        yield {
            "sample_index": sample_index,
            "seed": cfg.seed,
            "module_key": module_key,
            "module": module,
            "parameters": parameters,
            "curriculum_layer": int(module.get("level", 0)),
            "is_ood": bool(parameters),
        }


def _sample_id(module_key: str, seed: int, sample_index: int, parameters: dict[str, Any]) -> str:
    payload = {"module_key": module_key, "seed": seed, "sample_index": sample_index, "parameters": parameters}
    return _stable_hash(payload)[:16]


def _row_timestamp(seed: int, sample_index: int) -> float:
    return float(seed) + (float(sample_index) / 1000.0)


def _row_fingerprint(row: dict[str, Any]) -> str:
    payload = dict(row)
    payload.pop("row_fingerprint", None)
    return _stable_hash(payload)


def _row_from_task(cfg: GenerationConfig, task: dict[str, Any]) -> dict[str, Any]:
    module_key = task["module_key"]
    module = task["module"]
    parameters = task["parameters"]
    sample_index = int(task["sample_index"])
    seed = int(task["seed"])

    rule_text = _module_executable_rule(module)
    if rule_text is None:
        raise ValueError(f"Module {module_key} does not expose an executable rule.")

    initial_state = dict(module.get("initial_state", {}) or {})
    initial_state.update(parameters)

    sandbox_result = sandbox_manager.run_rule(
        rule_text,
        initial_state,
        frames=int(module.get("simulation_frames", 1)),
        collect_trace=True,
    )
    if sandbox_result.get("status") != "ok":
        raise RuntimeError(f"Sandbox failed for {module_key}: {sandbox_result}")

    invariants = list(module.get("invariants", [])) or ["logic_basic"]
    verification = verify_simulation(sandbox_result.get("trace", {}), invariants)
    module_version = _stable_hash(module)
    trace_export = _build_trace_export(
        rule_text,
        sandbox_result.get("trace", {}),
        module_key,
        module_version,
        seed,
        sample_index,
        invariants,
        verification,
    )

    row = {
        "sample_id": _sample_id(module_key, seed, sample_index, parameters),
        "question": _default_question(module_key, module, parameters),
        "structured_state": {
            "initial_state": initial_state,
            "parameters": parameters,
            "module": module_key,
            "module_version": module_version,
            "ood": bool(task.get("is_ood", False)),
        },
        "reasoning_trace": trace_export.to_dict().get("steps", []),
        "equations_used": _equations_from_rule(rule_text),
        "invariants_checked": invariants,
        "final_answer": sandbox_result.get("particle", sandbox_result),
        "verification_status": verification,
        "module_source": f"{cfg.modules_path}::{module_key}",
        "curriculum_layer": int(module.get("level", 0)),
        "seed": seed,
        "timestamp": _row_timestamp(seed, sample_index),
        "dataset_version": DATASET_VERSION,
        "snapshot_hash": _stable_hash({"seed": seed, "module_key": module_key, "module_version": module_version})[:32],
        "module_hash": module_version,
        "trace_export": trace_export.to_dict(),
        "module_key": module_key,
        "row_fingerprint": "",
    }
    row["row_fingerprint"] = _row_fingerprint(row)
    return row


def generate_large_dataset(
    cfg: GenerationConfig,
    *,
    policy: ArtifactPolicy | None = None,
    strict_policy: bool = False,
) -> Path:
    """Generate sharded JSONL oracle dataset and return output directory."""

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    registry = _load_modules_registry(cfg.modules_path)
    items = _module_items(registry, cfg.module_filter, cfg.layer_filter)
    if not items:
        raise ValueError("No modules matched the provided filters.")

    state_path = cfg.output_dir / STATE_FILE
    shard_manifest_path = cfg.output_dir / SHARD_MANIFEST_FILE
    dataset_manifest_path = cfg.output_dir / DATASET_MANIFEST_FILE
    partial_dir = cfg.output_dir / ".partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    existing_state = _load_state(state_path) if cfg.resume else {}
    completed_records = int(existing_state.get("completed_records", 0))
    next_shard_index = int(existing_state.get("next_shard_index", 0))

    if cfg.resume and existing_state and existing_state.get("seed") not in (None, cfg.seed):
        raise ValueError("Cannot resume a generation state created with a different seed.")

    shard_manifest = _load_state(shard_manifest_path) if shard_manifest_path.exists() else {
        "source": str(cfg.modules_path),
        "shard_size": cfg.shard_size,
        "shards": [],
        "total_records": 0,
    }
    shards: list[dict[str, Any]] = list(shard_manifest.get("shards", []))

    for partial_file in partial_dir.glob("*.jsonl"):
        if partial_file.exists() and not cfg.resume:
            partial_file.unlink()

    buffer: list[dict[str, Any]] = []
    row_count = completed_records
    shard_index = next_shard_index
    start_index = completed_records
    generated_this_run = 0

    for sample_index, task in enumerate(iter_balanced_tasks(cfg)):
        if sample_index < start_index:
            continue
        row = _row_from_task(cfg, task)
        buffer.append(row)
        row_count += 1
        generated_this_run += 1

        if len(buffer) >= cfg.shard_size:
            shard_name = f"shard_{shard_index:06d}.jsonl"
            shard_partial = partial_dir / f"{shard_name}.partial"
            shard_final = cfg.output_dir / shard_name
            shard_partial.write_text("\n".join(json.dumps(item, sort_keys=True, ensure_ascii=False) for item in buffer) + "\n", encoding="utf-8")
            os.replace(shard_partial, shard_final)
            shards.append(
                {
                    "name": shard_name,
                    "path": str(shard_final),
                    "records": len(buffer),
                    "hash": _stable_hash([item["row_fingerprint"] for item in buffer]),
                }
            )
            shard_index += 1
            buffer = []
            shard_manifest = {
                "source": str(cfg.modules_path),
                "shard_size": cfg.shard_size,
                "shards": shards,
                "total_records": row_count,
            }
            shard_manifest["fingerprint"] = _stable_hash(shard_manifest)
            shard_manifest_path.write_text(json.dumps(shard_manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
            _save_state(
                state_path,
                {
                    "seed": cfg.seed,
                    "completed_records": row_count,
                    "next_shard_index": shard_index,
                    "num_samples": cfg.num_samples,
                    "shard_size": cfg.shard_size,
                    "resume": cfg.resume,
                },
            )

    if buffer:
        shard_name = f"shard_{shard_index:06d}.jsonl"
        shard_partial = partial_dir / f"{shard_name}.partial"
        shard_final = cfg.output_dir / shard_name
        shard_partial.write_text("\n".join(json.dumps(item, sort_keys=True, ensure_ascii=False) for item in buffer) + "\n", encoding="utf-8")
        os.replace(shard_partial, shard_final)
        shards.append(
            {
                "name": shard_name,
                "path": str(shard_final),
                "records": len(buffer),
                "hash": _stable_hash([item["row_fingerprint"] for item in buffer]),
            }
        )

    shard_manifest = {
        "source": str(cfg.modules_path),
        "shard_size": cfg.shard_size,
        "shards": shards,
        "total_records": min(cfg.num_samples, completed_records + generated_this_run),
    }
    shard_manifest["fingerprint"] = _stable_hash(shard_manifest)
    shard_manifest_path.write_text(json.dumps(shard_manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    dataset_manifest = DatasetManifest.from_oracle_result(
        output_path=cfg.output_dir,
        modules_used=[key for key, _module in items],
        module_versions={key: _stable_hash(module) for key, module in items},
        seed=cfg.seed,
        snapshot_hash=_stable_hash({"seed": cfg.seed, "num_samples": cfg.num_samples})[:32],
        module_hash=_stable_hash([key for key, _module in items]),
        record_count=shard_manifest["total_records"],
        parameter_sweeps={"layer_weights": list((cfg.layer_weights or {}).items())},
        curriculum_coverage=sorted({int(module.get("level", 0)) for _, module in items}),
        benchmark_fingerprint=shard_manifest["fingerprint"],
    )
    dataset_manifest.shard_list = [shard["name"] for shard in shards]
    dataset_manifest.save(dataset_manifest_path, policy=policy, strict_policy=strict_policy)

    _save_state(
        state_path,
        {
            "seed": cfg.seed,
            "completed_records": shard_manifest["total_records"],
            "next_shard_index": len(shards),
            "num_samples": cfg.num_samples,
            "shard_size": cfg.shard_size,
            "resume": cfg.resume,
        },
    )
    return cfg.output_dir


def _parse_layer_weights(values: list[str] | None) -> dict[int, float] | None:
    if not values:
        return None
    weights: dict[int, float] = {}
    for item in values:
        layer_text, weight_text = item.split("=", 1)
        weights[int(layer_text.strip())] = float(weight_text.strip())
    return weights


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a large deterministic oracle dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory for shards and manifests.")
    parser.add_argument("--modules", default="backend/core_truth/modules.json", help="Module registry path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1000000)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--module", action="append", dest="module_filter", default=None)
    parser.add_argument("--layer", action="append", dest="layer_filter", type=int, default=None)
    parser.add_argument("--layer-weight", action="append", dest="layer_weights", default=None, help="Format: layer=weight")
    parser.add_argument("--ood-ratio", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--policy", default="configs/artifact_policy.yaml", help="Artifact policy path.")
    parser.add_argument("--strict-policy", action="store_true", help="Fail on policy mismatches.")
    args = parser.parse_args()

    policy = load_artifact_policy(Path(args.policy))
    cfg = GenerationConfig(
        seed=args.seed,
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        shard_size=args.shard_size,
        layer_weights=_parse_layer_weights(args.layer_weights),
        module_filter=args.module_filter,
        layer_filter=args.layer_filter,
        ood_ratio=args.ood_ratio,
        resume=args.resume,
        modules_path=Path(args.modules),
    )
    out = generate_large_dataset(cfg, policy=policy, strict_policy=args.strict_policy)
    print(json.dumps({"output_dir": str(out)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
