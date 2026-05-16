#!/usr/bin/env python3
"""Run CPT-Bench locally and emit a versioned report."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from backend.benchmarks.cpt_bench import CPTBenchSuite
from scripts.validation_runner import validate_benchmark_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CPT-Bench.")
    parser.add_argument("--output", default=None, help="Report output path. Defaults to a versioned file in reports/benchmarks.")
    parser.add_argument("--no-validate", action="store_true", help="Skip automatic validation after benchmark execution.")
    parser.add_argument("--validation-output", default=None, help="Optional validation report path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    suite = CPTBenchSuite()
    if args.output:
        output_path = Path(args.output)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        output_path = Path("reports/benchmarks") / f"cpt_bench_v2.6.0_{stamp}.json"
    result = suite.write_report(output_path)
    print(json.dumps({
        "report_path": str(result.report_path),
        "version": result.version,
        "cases": result.metrics["cases"],
        "pass_rate": result.metrics["pass_rate"],
        "fingerprint": result.fingerprint,
        "metrics": result.metrics,
    }, indent=2, sort_keys=True))

    if not args.no_validate:
        validation = validate_benchmark_file(result.report_path, validation_path=args.validation_output)
        print(json.dumps(validation["report"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
