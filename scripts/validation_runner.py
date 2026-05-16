#!/usr/bin/env python3
"""Run the validation pipeline against oracle datasets or benchmark reports."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from backend.datasets.oracle_generator import _module_executable_rule  # noqa: F401
from backend.validation.pipeline import ValidationPipeline
from backend.validation.thresholds import InvariantThresholds

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_modules(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"modules": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_module_row(row: dict[str, Any], modules_path: Path) -> dict[str, Any]:
    module_source = str(row.get("module_source", ""))
    module_key = row.get("module_key")
    if "::" in module_source and not module_key:
        _, module_key = module_source.split("::", 1)
    modules = _load_modules(modules_path).get("modules", {})
    module = modules.get(module_key or "", {})
    if not module:
        return {
            "rule": "\n".join(row.get("equations_used", [])),
            "initial_state": row.get("structured_state", {}).get("initial_state", {}),
            "frames": 1,
            "invariants": row.get("invariants_checked", []) or ["logic_basic"],
            "expected_state": row.get("final_answer", {}),
        }

    rule = _module_executable_rule(module) or "\n".join(row.get("equations_used", []))
    return {
        "rule": rule,
        "initial_state": row.get("structured_state", {}).get("initial_state", {}),
        "frames": int(module.get("simulation_frames", 1)),
        "invariants": row.get("invariants_checked", []) or list(module.get("invariants", [])) or ["logic_basic"],
        "expected_state": row.get("final_answer", {}),
    }


def _cases_from_dataset(rows: list[dict[str, Any]], modules_path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in rows:
        resolved = _resolve_module_row(row, modules_path)
        cases.append(
            {
                "rule": resolved["rule"],
                "initial_state": resolved["initial_state"],
                "frames": resolved["frames"],
                "invariants": resolved["invariants"],
                "expected_state": resolved["expected_state"],
            }
        )
    return cases


def _cases_from_benchmark_report(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for case in payload.get("cases", []):
        cases.append(
            {
                "rule": case.get("rule", case.get("structured_trace", {}).get("steps", [{}])[0].get("equation", "")),
                "initial_state": case.get("initial_state", {}),
                "frames": case.get("frames", 1),
                "invariants": case.get("invariants", []),
                "expected_state": case.get("expected_state", {}),
            }
        )
    return cases


def run_validation_dataset(
    dataset_path: str | Path,
    *,
    modules_path: str | Path = "backend/core_truth/modules.json",
    output_path: str | Path | None = None,
    thresholds: InvariantThresholds | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    modules_path = Path(modules_path)
    output_path = Path(output_path) if output_path is not None else dataset_path.with_suffix(".validation.json")
    rows = _load_jsonl(dataset_path)
    cases = _cases_from_dataset(rows, modules_path)
    pipeline = ValidationPipeline(thresholds=thresholds)
    report = pipeline.evaluate(cases, model_predictor=None)
    payload = {
        "input": str(dataset_path),
        "mode": "dataset",
        "passed": report.passed,
        "rejected": report.rejected,
        "metrics": report.metrics,
        "violations": report.violations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"report": payload, "output_path": str(output_path)}


def run_validation_benchmark(
    benchmark_path: str | Path,
    *,
    output_path: str | Path | None = None,
    thresholds: InvariantThresholds | None = None,
) -> dict[str, Any]:
    benchmark_path = Path(benchmark_path)
    output_path = Path(output_path) if output_path is not None else benchmark_path.with_suffix(".validation.json")
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    cases = _cases_from_benchmark_report(payload)
    pipeline = ValidationPipeline(thresholds=thresholds)
    report = pipeline.evaluate(cases, model_predictor=None)
    output = {
        "input": str(benchmark_path),
        "mode": "benchmark",
        "passed": report.passed,
        "rejected": report.rejected,
        "metrics": report.metrics,
        "violations": report.violations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    return {"report": output, "output_path": str(output_path)}


def validate_dataset_file(dataset_path: str | Path, validation_path: str | Path | None = None, modules_path: str | Path = "backend/core_truth/modules.json") -> dict[str, Any]:
    return run_validation_dataset(dataset_path, modules_path=modules_path, output_path=validation_path)


def validate_benchmark_file(benchmark_path: str | Path, validation_path: str | Path | None = None) -> dict[str, Any]:
    return run_validation_benchmark(benchmark_path, output_path=validation_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CPT validation pipeline.")
    parser.add_argument("--input", required=True, help="JSONL dataset or benchmark report JSON.")
    parser.add_argument("--mode", choices=["dataset", "benchmark"], default="dataset")
    parser.add_argument("--modules", default="backend/core_truth/modules.json", help="Module registry path for dataset validation.")
    parser.add_argument("--output", default=None, help="Validation output path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.mode == "dataset":
        result = validate_dataset_file(args.input, validation_path=args.output, modules_path=args.modules)
    else:
        result = validate_benchmark_file(args.input, validation_path=args.output)

    print(json.dumps(result["report"], indent=2, sort_keys=True))
    logger.info("Validation report written to %s", result["output_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
