#!/usr/bin/env python3
"""Run the full deterministic oracle pipeline locally.

This command performs:
1. Oracle dataset generation
2. CPT-Bench execution
3. Validation of both artifacts

The resulting JSON summary is meant to be machine-readable and easy to hand
off to another agent.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from backend.benchmarks.cpt_bench import CPTBenchSuite
from backend.datasets.oracle_generator import OracleDatasetGenerator
from backend.validation.thresholds import InvariantThresholds
from scripts.validation_runner import validate_benchmark_file, validate_dataset_file


def _load_layers(value: list[int] | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in value]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full CPT v2.6 oracle pipeline.")
    parser.add_argument("--dataset-output", required=True, help="Path for the oracle JSONL dataset.")
    parser.add_argument("--benchmark-output", default=None, help="Path for the benchmark report JSON.")
    parser.add_argument("--modules", default="backend/core_truth/modules.json", help="Path to modules.json.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed.")
    parser.add_argument("--limit", type=int, default=None, help="Optional dataset sample limit.")
    parser.add_argument("--module", action="append", dest="modules_filter", default=None, help="Restrict to a module key. Repeatable.")
    parser.add_argument("--layer", action="append", dest="layers_filter", type=int, default=None, help="Restrict to a curriculum layer. Repeatable.")
    parser.add_argument("--exclude-tabular", action="store_true", help="Skip tabular modules.")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation runs.")
    parser.add_argument("--validation-output", default=None, help="Optional validation output path prefix or file path.")
    parser.add_argument("--benchmark-validation-output", default=None, help="Optional benchmark validation output path.")
    parser.add_argument("--dataset-validation-output", default=None, help="Optional dataset validation output path.")
    parser.add_argument("--benchmark-report-output", default=None, help="Optional benchmark report output path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    layers_filter = _load_layers(args.layers_filter)

    dataset_generator = OracleDatasetGenerator(
        output_path=Path(args.dataset_output),
        modules_path=Path(args.modules),
        seed=args.seed,
        include_tabular_modules=not args.exclude_tabular,
    )
    dataset_result = dataset_generator.generate_batch(
        module_keys=args.modules_filter,
        curriculum_layers=layers_filter,
        limit=args.limit,
    )

    benchmark_output = (
        Path(args.benchmark_report_output)
        if args.benchmark_report_output
        else Path("reports/benchmarks") / f"cpt_bench_v2.6.0_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json"
    )
    benchmark_result = CPTBenchSuite().write_report(benchmark_output)

    dataset_validation = None
    benchmark_validation = None
    if not args.no_validate:
        dataset_validation = validate_dataset_file(
            dataset_result.output_path,
            validation_path=args.dataset_validation_output,
            modules_path=args.modules,
        )
        benchmark_validation = validate_benchmark_file(
            benchmark_result.report_path,
            validation_path=args.benchmark_validation_output,
        )

    summary: dict[str, Any] = {
        "dataset": {
            "output_path": str(dataset_result.output_path),
            "manifest_path": str(dataset_result.manifest_path),
            "samples_generated": dataset_result.samples_generated,
            "modules_used": dataset_result.modules_used,
            "seed": dataset_result.seed,
            "dataset_fingerprint": dataset_result.dataset_fingerprint,
        },
        "benchmark": {
            "report_path": str(benchmark_result.report_path),
            "version": benchmark_result.version,
            "cases": benchmark_result.metrics["cases"],
            "pass_rate": benchmark_result.metrics["pass_rate"],
            "fingerprint": benchmark_result.fingerprint,
            "metrics": benchmark_result.metrics,
        },
        "validation": {
            "dataset": dataset_validation["report"] if dataset_validation else None,
            "benchmark": benchmark_validation["report"] if benchmark_validation else None,
        },
        "thresholds": InvariantThresholds.from_env().to_dict(),
        "modules_path": str(args.modules),
        "seed": args.seed,
        "layers_filter": layers_filter,
        "modules_filter": args.modules_filter,
    }

    output = {
        "summary": summary,
        "dataset_validation_output": dataset_validation["output_path"] if dataset_validation else None,
        "benchmark_validation_output": benchmark_validation["output_path"] if benchmark_validation else None,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

