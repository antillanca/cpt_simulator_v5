#!/usr/bin/env python3
"""Prepare a reproducible Kaggle v2.9D training export."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.neural.training_snapshot import fingerprint_jsonl, fingerprint_mapping, git_commit_hash, stable_fingerprint



def load_kaggle_profile(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Kaggle profile must be a mapping")
    return payload


def resolve_dataset_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.suffix == ".jsonl":
        alt = path.parent / path.stem / "circuits.jsonl"
        if alt.exists():
            return alt
    fallback = path.with_name("circuits.jsonl")
    if fallback.exists():
        return fallback
    if path.exists():
        return path
    return path


def prepare_kaggle_export(config_path: str | Path, output_dir: str | Path, checkpoint_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_kaggle_profile(config_path)
    
    raw_dataset = config["dataset"]["train_path"]
    # Check if we should override dataset path from command line
    dataset_path = resolve_dataset_path(raw_dataset)
    
    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    normalized_config = {
        "seed": int(config["seed"]),
        "device": dict(config.get("device", {})),
        "training": dict(config.get("training", {})),
        "dataset": {
            "train_path": str(config["dataset"]["train_path"]),
            "eval_split": float(config["dataset"]["eval_split"]),
        },
        "model": dict(config.get("model", {})),
        "evaluation": dict(config.get("evaluation", {})),
        "output": str(checkpoint_path),
        "model_type": "edge_aware",
        "physics": dict(config.get("physics", {"lambda_kcl": 5.0, "lambda_kvl": 5.0, "lambda_power": 1.0}))
    }

    config_fingerprint = fingerprint_mapping(normalized_config)
    dataset_fingerprint = fingerprint_jsonl(dataset_path)
    profile_fingerprint = stable_fingerprint(
        {
            "config_fingerprint": config_fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
            "config_path": str(config_path.resolve()),
        }
    )
    run_script = export_dir / "run_kaggle_v29d.sh"
    run_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"',
                'cd "$REPO_ROOT"',
                "python scripts/train_circuit_gnn.py --config configs/training/kaggle_v29d.yaml",
                "python scripts/run_circuit_arena.py --checkpoint workspace/checkpoints/circuit_gnn_v29d_full.pt --output-dir workspace/arena_results",
                "python scripts/generate_v29d_report.py --config configs/training/kaggle_v29d.yaml",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        run_script.chmod(0o755)
    except Exception:
        pass

    shutil.copy2(config_path, export_dir / config_path.name)
    manifest = {
        "schema_version": "2.9d",
        "profile": "kaggle_v29d",
        "config_path": str(config_path.resolve()),
        "dataset_path": str(dataset_path.resolve()) if dataset_path.exists() else str(dataset_path),
        "git_commit": git_commit_hash(PROJECT_ROOT),
        "config_fingerprint": config_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "profile_fingerprint": profile_fingerprint,
        "run_command": "bash run_kaggle_v29d.sh",
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "snapshot": "workspace/training_snapshots/training_snapshot.json",
            "arena_results": "workspace/arena_results/v29d_full_arena_metrics.json",
            "arena_reports": "workspace/arena_reports/v29d_comparison.json",
            "docs_report": "docs/V29D_PHYSICS_INFORMED_COMPARISON.md",
        },
    }
    (export_dir / "kaggle_v29d_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Kaggle v2.9D export.")
    parser.add_argument("--config", default="configs/training/kaggle_v29d.yaml")
    parser.add_argument("--checkpoint", default="workspace/checkpoints/circuit_gnn_v29d_full.pt")
    parser.add_argument("--dataset", default="workspace/datasets/circuits/train_10k.jsonl")
    parser.add_argument("--output-dir", default="workspace/kaggle_exports/v29d")
    args = parser.parse_args()

    manifest = prepare_kaggle_export(args.config, args.output_dir, args.checkpoint)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
