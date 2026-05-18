#!/usr/bin/env python3
"""Run CPT v2.8 circuit oracle core benchmarks and output reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.circuits.benchmarks import run_all_benchmarks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CPT v2.8 circuit oracle benchmarks.")
    parser.add_argument("--output-dir", default="workspace/reports/circuits", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_all_benchmarks()

    # JSON report
    json_path = output_dir / "benchmark_report.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"JSON report → {json_path}")

    # Markdown report
    md_path = output_dir / "benchmark_report.md"
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    print(f"Markdown report → {md_path}")

    print(f"\nResults: {report.passed}/{report.total} passed")
    print(f"Invariant pass rate: {report.invariant_pass_rate:.1%}")
    print(f"Trace determinism: {report.trace_determinism_rate:.1%}")
    print(f"Avg solve time: {report.avg_solve_time_ms:.3f} ms")

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
