#!/usr/bin/env python3
"""Deterministic tiny-model training entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.neural.tiny_experiments import TrainConfig, train_model


def _resolve_data_path(args: argparse.Namespace) -> Path:
    if args.dataset:
        return Path(args.dataset)
    if args.shard_dir and args.manifest:
        return Path(args.manifest)
    raise SystemExit("Must provide --dataset or both --shard-dir and --manifest")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic tiny-model training")
    parser.add_argument("--dataset", type=str, default=None, help="Path to JSONL dataset file")
    parser.add_argument("--shard-dir", type=str, default=None, help="Path to shard directory")
    parser.add_argument("--manifest", type=str, default=None, help="Path to shard manifest JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--model-type", type=str, default="transformer", choices=["transformer", "tiny_transformer", "seq2seq", "gnn", "pinn"])
    parser.add_argument("--output-dir", type=str, default="training_runs")
    parser.add_argument("--output-checkpoint", type=str, default=None)
    parser.add_argument("--train-split", type=float, default=0.8)
    args = parser.parse_args()

    data_path = _resolve_data_path(args)
    cfg = TrainConfig(
        model_type="transformer" if args.model_type == "tiny_transformer" else args.model_type,
        seed=args.seed,
        data_path=data_path,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        device=args.device,
        eval_every=args.eval_every,
        save_every=args.save_every,
        train_split=args.train_split,
        shard_dir=Path(args.shard_dir) if args.shard_dir else None,
        manifest_path=Path(args.manifest) if args.manifest else None,
        output_checkpoint=Path(args.output_checkpoint) if args.output_checkpoint else None,
    )
    result = train_model(cfg)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
