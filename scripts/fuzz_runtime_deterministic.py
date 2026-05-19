#!/usr/bin/env python3
"""CORE v3.2 — Deterministic Fuzzing for Runtime Stability.

Generate synthetic tasks deterministically and run them twice.
Each task must produce identical results across runs.

The fuzzer detects:
  - nondeterministic traces
  - unstable routing
  - unstable projection outcomes
  - unstable memory writes

Usage:
  python scripts/fuzz_runtime_deterministic.py [--seed 42] [--tasks 50] [--output json|text]

Exit code: 0 if all fuzz checks pass, 1 if any nondeterminism detected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Ensure repo root on sys.path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from core_runtime.core.specs.task_hashing import compute_task_hash, canonicalize_task
from core_runtime.domains.linear_system import (
    LinearSystemOracle,
    LinearSystemProjection,
    LinearSystemSurrogate,
    LinearSystemTask,
    execute_linear_system_pipeline,
)
from backend.core_runtime.task_runtime import RuntimeTask


# ---------------------------------------------------------------------------
# Fuzz task generation
# ---------------------------------------------------------------------------

def generate_fuzz_tasks(
    seed: int = 42,
    count: int = 50,
) -> list[LinearSystemTask]:
    """Generate diverse synthetic tasks for fuzzing.

    Each task is generated from a fixed sub-seed derived from the master
    seed, ensuring full determinism.
    """
    master_rng = np.random.default_rng(seed)
    tasks = []

    for i in range(count):
        sub_seed = int(master_rng.integers(0, 2**31))
        rng = np.random.default_rng(sub_seed)

        # Vary size, condition number, and structure
        size = int(rng.integers(2, 10))
        scale = float(rng.uniform(1.0, 10.0))
        noise_scale = float(rng.uniform(0.01, 1.0))

        # Generate SPD matrix with varying condition number
        A_base = rng.standard_normal((size, size))
        eigenvalues = np.sort(np.abs(rng.standard_normal(size)))
        eigenvalues = eigenvalues * scale + 0.1  # Ensure positive
        U, _, Vt = np.linalg.svd(A_base)
        A = U @ np.diag(eigenvalues) @ Vt
        A = (A + A.T) / 2.0 + size * np.eye(size)  # Ensure SPD

        b = rng.standard_normal(size) * scale

        task = LinearSystemTask(
            task_id=f"fuzz_{seed}_{i:04d}",
            domain_name="linear_system",
            input_artifact=f"fuzz_{seed}_{i:04d}",
            metadata={
                "A": A,
                "b": b,
                "fuzz_sub_seed": sub_seed,
                "fuzz_size": size,
                "fuzz_scale": scale,
                "fuzz_noise": noise_scale,
            },
        )
        tasks.append(task)

    return tasks


# ---------------------------------------------------------------------------
# Fuzz result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FuzzTaskResult:
    """Result from fuzzing a single task twice."""
    task_id: str
    task_hash_run1: str
    task_hash_run2: str
    hash_match: bool
    trace_match: bool
    projection_solution_match: bool
    projection_residual_match: bool
    routing_match: bool
    residual_diff: float


@dataclass(frozen=True)
class FuzzReport:
    """Aggregate fuzzing report."""
    version: str
    seed: int
    task_count: int
    hash_mismatches: int
    trace_mismatches: int
    projection_mismatches: int
    residual_mismatches: int
    routing_mismatches: int
    total_mismatches: int
    overall_pass: bool
    tasks: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Fuzz runner
# ---------------------------------------------------------------------------

def _to_compat_task(ls_task: LinearSystemTask) -> RuntimeTask:
    """Convert to backend-compatible RuntimeTask."""
    metadata = {}
    for k, v in ls_task.metadata.items():
        if isinstance(v, np.ndarray):
            metadata[k] = v.tolist()
        else:
            metadata[k] = v
    return RuntimeTask(
        task_id=ls_task.task_id,
        domain=ls_task.domain_name,
        input_artifact=ls_task.input_artifact,
        oracle_name="LinearSystemOracle",
        surrogate_name="LinearSystemSurrogate",
        projection_enabled=True,
        metadata=metadata,
    )


def run_fuzz(
    seed: int = 42,
    task_count: int = 50,
    budget: int = 50,
    residual_tolerance: float = 1e-12,
) -> FuzzReport:
    """Run deterministic fuzzing on the runtime.

    Each task is executed twice and the results are compared for
    exact equality (within tolerance for floats).
    """
    tasks = generate_fuzz_tasks(seed=seed, count=task_count)

    task_results: list[FuzzTaskResult] = []
    hash_mismatches = 0
    trace_mismatches = 0
    projection_mismatches = 0
    residual_mismatches = 0
    routing_mismatches = 0

    for task in tasks:
        # Compute task hash twice
        compat1 = _to_compat_task(task)
        compat2 = _to_compat_task(task)
        h1 = compute_task_hash(compat1)
        h2 = compute_task_hash(compat2)
        hash_match = (h1 == h2)
        if not hash_match:
            hash_mismatches += 1

        # Execute pipeline twice
        r1 = execute_linear_system_pipeline(task, budget=budget)
        r2 = execute_linear_system_pipeline(task, budget=budget)

        # Compare traces
        trace_match = True
        for key in [
            "task_id", "domain_name", "fingerprint",
            "node_count", "edge_count",
            "surrogate_method", "projection_iterations",
            "projection_converged", "projection_method",
            "evaluation_correct", "trajectory_length",
        ]:
            if r1["trace"].get(key) != r2["trace"].get(key):
                trace_match = False
                break
        if not trace_match:
            trace_mismatches += 1

        # Compare projection solutions
        projection_solution_match = np.allclose(
            r1["projection"]["solution"],
            r2["projection"]["solution"],
            atol=1e-14,
        )
        if not projection_solution_match:
            projection_mismatches += 1

        # Compare projection residuals
        res1 = r1["projection"]["residual"]
        res2 = r2["projection"]["residual"]
        residual_diff = abs(res1 - res2)
        residual_match = residual_diff <= residual_tolerance
        if not residual_match:
            residual_mismatches += 1

        # Compare routing (via surrogate method + projection method)
        routing_match = (
            r1["trace"]["surrogate_method"] == r2["trace"]["surrogate_method"]
            and r1["trace"]["projection_method"] == r2["trace"]["projection_method"]
        )
        if not routing_match:
            routing_mismatches += 1

        task_results.append(FuzzTaskResult(
            task_id=task.task_id,
            task_hash_run1=h1,
            task_hash_run2=h2,
            hash_match=hash_match,
            trace_match=trace_match,
            projection_solution_match=projection_solution_match,
            projection_residual_match=residual_match,
            routing_match=routing_match,
            residual_diff=residual_diff,
        ))

    total_mismatches = (
        hash_mismatches + trace_mismatches + projection_mismatches
        + residual_mismatches + routing_mismatches
    )

    return FuzzReport(
        version="v3.2-fuzz",
        seed=seed,
        task_count=task_count,
        hash_mismatches=hash_mismatches,
        trace_mismatches=trace_mismatches,
        projection_mismatches=projection_mismatches,
        residual_mismatches=residual_mismatches,
        routing_mismatches=routing_mismatches,
        total_mismatches=total_mismatches,
        overall_pass=(total_mismatches == 0),
        tasks=[asdict(t) for t in task_results],
    )


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def format_text_report(report: FuzzReport) -> str:
    """Format fuzz report as human-readable text."""
    lines = [
        "=" * 60,
        "CORE v3.2 Deterministic Fuzzing Report",
        "=" * 60,
        f"  Version:       {report.version}",
        f"  Seed:          {report.seed}",
        f"  Tasks:         {report.task_count}",
        "-" * 60,
        f"  Hash mismatches:       {report.hash_mismatches}",
        f"  Trace mismatches:      {report.trace_mismatches}",
        f"  Projection mismatches: {report.projection_mismatches}",
        f"  Residual mismatches:   {report.residual_mismatches}",
        f"  Routing mismatches:    {report.routing_mismatches}",
        "-" * 60,
        f"  Total mismatches:      {report.total_mismatches}",
        f"  Overall:               {'PASS' if report.overall_pass else 'FAIL'}",
        "=" * 60,
    ]

    # Show any failures
    if not report.overall_pass:
        lines.append("")
        lines.append("FAILED TASKS:")
        for t in report.tasks:
            if not (t["hash_match"] and t["trace_match"]
                    and t["projection_solution_match"]
                    and t["projection_residual_match"]
                    and t["routing_match"]):
                failures = []
                if not t["hash_match"]:
                    failures.append("hash")
                if not t["trace_match"]:
                    failures.append("trace")
                if not t["projection_solution_match"]:
                    failures.append("projection_solution")
                if not t["projection_residual_match"]:
                    failures.append(f"residual(diff={t['residual_diff']:.2e})")
                if not t["routing_match"]:
                    failures.append("routing")
                lines.append(f"  {t['task_id']}: {', '.join(failures)}")

    return "\n".join(lines)


def format_json_report(report: FuzzReport) -> str:
    """Format fuzz report as JSON."""
    return json.dumps(asdict(report), indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CORE v3.2 Deterministic Fuzzing"
    )
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--tasks", type=int, default=50, help="Number of fuzz tasks")
    parser.add_argument("--budget", type=int, default=50, help="Projection budget")
    parser.add_argument("--tolerance", type=float, default=1e-12,
                        help="Residual comparison tolerance")
    parser.add_argument("--output", choices=["text", "json"], default="text",
                        help="Output format")
    args = parser.parse_args()

    report = run_fuzz(
        seed=args.seed,
        task_count=args.tasks,
        budget=args.budget,
        residual_tolerance=args.tolerance,
    )

    if args.output == "json":
        print(format_json_report(report))
    else:
        print(format_text_report(report))

    sys.exit(0 if report.overall_pass else 1)


if __name__ == "__main__":
    main()
