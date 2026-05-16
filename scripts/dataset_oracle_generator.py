#!/usr/bin/env python3
"""CLI for deterministic oracle dataset generation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from backend.datasets.oracle_generator import OracleDatasetGenerator
from scripts.validation_runner import validate_dataset_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CPT oracle JSONL datasets from sandbox execution.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--modules", default="backend/core_truth/modules.json", help="Module registry path.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit.")
    parser.add_argument("--module", action="append", dest="modules_filter", default=None, help="Restrict to a module key. May be repeated.")
    parser.add_argument("--layer", action="append", dest="layers_filter", type=int, default=None, help="Restrict to a curriculum layer. May be repeated.")
    parser.add_argument("--exclude-tabular", action="store_true", help="Skip tabular modules.")
    parser.add_argument("--no-validate", action="store_true", help="Skip automatic validation after generation.")
    parser.add_argument("--validation-output", default=None, help="Optional validation report path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generator = OracleDatasetGenerator(
        output_path=Path(args.output),
        modules_path=Path(args.modules),
        seed=args.seed,
        include_tabular_modules=not args.exclude_tabular,
    )
    result = generator.generate_batch(
        module_keys=args.modules_filter,
        curriculum_layers=args.layers_filter,
        limit=args.limit,
    )
    print(json.dumps({
        "output_path": str(result.output_path),
        "manifest_path": str(result.manifest_path),
        "samples_generated": result.samples_generated,
        "modules_used": result.modules_used,
        "seed": result.seed,
        "dataset_fingerprint": result.dataset_fingerprint,
    }, indent=2, sort_keys=True))

    if not args.no_validate:
        validation = validate_dataset_file(
            result.output_path,
            validation_path=args.validation_output,
            modules_path=args.modules,
        )
        print(json.dumps(validation["report"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
