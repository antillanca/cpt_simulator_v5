#!/usr/bin/env python3
"""Deterministic oracle-vs-model evaluation runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.datasets.loader import load_jsonl, load_sharded_dataset
from backend.neural.tiny_experiments import MODEL_TYPES, CharTokenizer, TinyTransformerModel, load_checkpoint, set_deterministic
from backend.validation.model_eval import ModelEvaluator
from backend.validation.oracle_arena import ArenaExample, aggregate_arena_results, compare_oracle_vs_model


def _load_records(dataset_path: Path, shard_dir: Path | None = None, manifest_path: Path | None = None) -> list[dict[str, Any]]:
    if shard_dir is not None and manifest_path is not None:
        rows, _manifest = load_sharded_dataset(shard_dir, manifest_path, validate=False)
        return rows
    return load_jsonl(dataset_path, validate=False)


def _build_model_from_checkpoint(checkpoint: Path | None, model_type: str, tokenizer: CharTokenizer, device: str):
    if checkpoint is not None and checkpoint.exists():
        model, loaded_tokenizer, payload = load_checkpoint(checkpoint, device=device)
        return model, loaded_tokenizer, payload
    model_cls = MODEL_TYPES.get(model_type, TinyTransformerModel)
    model = model_cls(len(tokenizer.itos), tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id)
    model.tokenizer = tokenizer
    model.eval()
    return model, tokenizer, {}


def evaluate_dataset(
    dataset_path: Path,
    *,
    checkpoint: Path | None = None,
    predictions_path: Path | None = None,
    model_type: str = "transformer",
    seed: int = 42,
    output_path: Path | None = None,
    limit: int | None = None,
    shard_dir: Path | None = None,
    manifest_path: Path | None = None,
    layer_filter: list[int] | None = None,
    module_filter: list[str] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    set_deterministic(seed)
    records = _load_records(dataset_path, shard_dir=shard_dir, manifest_path=manifest_path)
    if layer_filter is not None:
        layer_set = set(int(layer) for layer in layer_filter)
        records = [row for row in records if int(row.get("curriculum_layer", -1)) in layer_set]
    if module_filter is not None:
        module_set = set(module_filter)
        records = [row for row in records if str(row.get("module_source", "")) in module_set or str(row.get("module_key", "")) in module_set]
    if limit is not None:
        records = records[:limit]

    tokenizer = CharTokenizer.build(records) if records else CharTokenizer([" "])
    model, tokenizer, checkpoint_payload = _build_model_from_checkpoint(checkpoint, model_type, tokenizer, device)

    if predictions_path is not None:
        predictions = load_jsonl(predictions_path, validate=False)
    else:
        predictions = []
        for index, record in enumerate(records):
            prediction = model.predict({"question": record.get("question", ""), "structured_state": record.get("structured_state", {}), "sample_id": record.get("sample_id", "")})
            prediction.setdefault("structured_state", record.get("structured_state", {}))
            prediction.setdefault("module_source", record.get("module_source", ""))
            prediction.setdefault("curriculum_layer", record.get("curriculum_layer", -1))
            predictions.append(prediction)

    arena_results = []
    for index, (record, prediction) in enumerate(zip(records, predictions)):
        arena_results.append(
            compare_oracle_vs_model(
                ArenaExample(
                    sample_id=str(record.get("sample_id", index)),
                    question=str(record.get("question", "")),
                    oracle=record,
                    model_output=prediction,
                    metadata={
                        "module_source": record.get("module_source", ""),
                        "curriculum_layer": record.get("curriculum_layer", -1),
                        "initial_state": record.get("structured_state", {}).get("initial_state", {}),
                        "is_ood": bool(record.get("structured_state", {}).get("ood", False)),
                    },
                )
            )
        )

    evaluator = ModelEvaluator(model_type=model_type)
    eval_result = evaluator.evaluate(predictions, records)
    report = {
        "dataset_path": str(dataset_path),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "model_type": model_type,
        "seed": seed,
        "total_samples": len(records),
        "evaluation": eval_result.to_dict(),
        "arena": {
            "results": [result.to_dict() for result in arena_results],
            "by_module": aggregate_arena_results(arena_results, group_by="module"),
            "by_layer": aggregate_arena_results(arena_results, group_by="layer"),
            "by_category": aggregate_arena_results(arena_results, group_by="category"),
        },
        "checkpoint_meta": checkpoint_payload.get("extra", {}) if checkpoint_payload else {},
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic oracle-vs-model evaluation.")
    parser.add_argument("--dataset", required=True, help="Path to oracle JSONL dataset")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--predictions", default=None, help="Optional predictions JSONL path")
    parser.add_argument("--model-type", default="transformer", choices=["transformer", "tiny_transformer", "seq2seq", "gnn", "pinn"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="eval_result.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--layer", action="append", dest="layers", type=int, default=None)
    parser.add_argument("--module", action="append", dest="modules", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    report = evaluate_dataset(
        Path(args.dataset),
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        predictions_path=Path(args.predictions) if args.predictions else None,
        model_type="transformer" if args.model_type == "tiny_transformer" else args.model_type,
        seed=args.seed,
        output_path=Path(args.output),
        limit=args.limit,
        shard_dir=Path(args.shard_dir) if args.shard_dir else None,
        manifest_path=Path(args.manifest) if args.manifest else None,
        layer_filter=args.layers,
        module_filter=args.modules,
        device=args.device,
    )
    print(json.dumps(
        {
            "total_samples": report["total_samples"],
            "evaluation_fingerprint": report["evaluation"]["fingerprint"],
            "arena_fingerprint": report["arena"]["by_module"],
        },
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
