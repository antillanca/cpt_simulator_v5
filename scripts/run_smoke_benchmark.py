#!/usr/bin/env python3
"""CORE v3.2 — CI Smoke Benchmark.

Fast, deterministic benchmark suitable for pull request CI.
Uses the linear_system domain (no GPU, no circuit dependencies).

Reports:
  - runtime_ms
  - scheduler_overhead_ms
  - projection_iterations
  - residuals (surrogate, projected)
  - cache_hit_rate
  - retrieval_hit_rate
  - degraded_rate

Usage:
  python scripts/run_smoke_benchmark.py [--seed 42] [--samples 10] [--output json|text]

Exit code: 0 if all checks pass, 1 if any regression detected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure repo root on sys.path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from core_runtime.core.memory.exact_cache import ExactMatchCache
from core_runtime.core.routing.capability_router import CapabilityRouter
from core_runtime.core.specs.task_hashing import compute_task_hash
from core_runtime.domains.linear_system import (
    LinearSystemEvaluator,
    LinearSystemOracle,
    LinearSystemProjection,
    LinearSystemSurrogate,
    LinearSystemTask,
    execute_linear_system_pipeline,
)
from backend.core_runtime.task_runtime import RuntimeTask


# ---------------------------------------------------------------------------
# Data classes for the report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmokeSampleResult:
    """Result from a single smoke sample."""
    task_id: str
    runtime_ms: float
    scheduler_overhead_ms: float
    projection_iterations: int
    surrogate_residual: float
    projected_residual: float
    cache_hit: bool
    degraded: bool
    trajectory_class: str


@dataclass(frozen=True)
class SmokeReport:
    """Aggregate smoke benchmark report."""
    version: str
    seed: int
    sample_count: int
    total_runtime_ms: float
    avg_runtime_ms: float
    avg_scheduler_overhead_ms: float
    avg_projection_iterations: float
    avg_surrogate_residual: float
    avg_projected_residual: float
    cache_hit_rate: float
    retrieval_hit_rate: float
    degraded_rate: float
    correctness_pass: bool
    determinism_pass: bool
    overall_pass: bool
    samples: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Smoke benchmark core
# ---------------------------------------------------------------------------

def generate_smoke_tasks(seed: int = 42, count: int = 10) -> list[LinearSystemTask]:
    """Generate deterministic linear system tasks for smoke testing."""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(count):
        size = rng.integers(3, 8)  # Small systems: 3-7 dimensions
        A = rng.standard_normal((size, size))
        A = A @ A.T + 5.0 * np.eye(size)  # Ensure SPD
        b = rng.standard_normal(size)
        task = LinearSystemTask(
            task_id=f"smoke_{seed}_{i:04d}",
            domain_name="linear_system",
            input_artifact=f"smoke_{seed}_{i:04d}",
            metadata={"A": A, "b": b},
        )
        tasks.append(task)
    return tasks


def run_smoke_benchmark(
    seed: int = 42,
    sample_count: int = 10,
    budget: int = 50,
    output_dir: str | None = None,
) -> SmokeReport:
    """Run the CI smoke benchmark.

    Args:
        seed: Random seed for deterministic task generation.
        sample_count: Number of tasks to run.
        budget: Projection budget per task.
        output_dir: Optional directory for artifacts.

    Returns:
        SmokeReport with aggregate metrics.
    """
    np.random.seed(seed)

    # Generate tasks
    tasks = generate_smoke_tasks(seed=seed, count=sample_count)

    # Setup cache and router
    cache_dir = output_dir or "/tmp/cpt_smoke_cache"
    cache = ExactMatchCache(base_dir=cache_dir)
    from backend.core_runtime.confidence_runtime import ConfidenceEstimate
    router = CapabilityRouter()

    # Pre-populate cache with first run (for cache-hit testing)
    prepopulate = sample_count // 2  # Half will be cache hits
    for i in range(prepopulate):
        task = tasks[i]
        compat_task = _to_compat_runtime_task(task)
        task_hash = compute_task_hash(compat_task)
        if not cache.contains(task_hash):
            result = execute_linear_system_pipeline(task, budget=budget)
            rt_result = _pipeline_result_to_runtime_result(result, task)
            cache.put(task_hash, rt_result)

    # Run benchmark
    sample_results: list[SmokeSampleResult] = []
    total_runtime_ms = 0.0
    cache_hits = 0
    degraded_count = 0
    projection_iterations_list: list[int] = []
    surrogate_residuals: list[float] = []
    projected_residuals: list[float] = []
    scheduler_overhead_ms_list: list[float] = []

    for i, task in enumerate(tasks):
        t_start = time.perf_counter()

        # Scheduler overhead measurement
        t_sched_start = time.perf_counter()
        compat_task = _to_compat_runtime_task(task)
        task_hash = compute_task_hash(compat_task)
        is_cached = cache.contains(task_hash)
        confidence = ConfidenceEstimate(
            confidence_score=0.9,
            estimated_projection_iterations=budget,
            likely_ood=False,
        )
        _decision = router.route(
            compat_task,
            confidence=confidence,
            cache_hit=is_cached,
            retrieval_similarity=0.0,
        )
        t_sched_end = time.perf_counter()
        scheduler_overhead_ms = (t_sched_end - t_sched_start) * 1000.0

        # Check cache
        cached = cache.get(task_hash)
        cache_hit = cached is not None
        if cache_hit:
            cache_hits += 1
            runtime_ms = (time.perf_counter() - t_start) * 1000.0
            sample_results.append(SmokeSampleResult(
                task_id=task.task_id,
                runtime_ms=runtime_ms,
                scheduler_overhead_ms=scheduler_overhead_ms,
                projection_iterations=0,
                surrogate_residual=0.0,
                projected_residual=0.0,
                cache_hit=True,
                degraded=False,
                trajectory_class="exact_cache_hit",
            ))
            continue

        # Execute pipeline
        result = execute_linear_system_pipeline(task, budget=budget)
        runtime_ms = (time.perf_counter() - t_start) * 1000.0
        total_runtime_ms += runtime_ms

        proj_iters = result["trace"]["projection_iterations"]
        surr_res = result["surrogate"]["residual"]
        proj_res = result["projection"]["residual"]
        is_degraded = result["trace"].get("failure_type") is not None

        projection_iterations_list.append(proj_iters)
        surrogate_residuals.append(surr_res)
        projected_residuals.append(proj_res)
        scheduler_overhead_ms_list.append(scheduler_overhead_ms)

        if is_degraded:
            degraded_count += 1

        traj = result["trace"].get("trajectory_class", "standard")
        sample_results.append(SmokeSampleResult(
            task_id=task.task_id,
            runtime_ms=runtime_ms,
            scheduler_overhead_ms=scheduler_overhead_ms,
            projection_iterations=proj_iters,
            surrogate_residual=surr_res,
            projected_residual=proj_res,
            cache_hit=False,
            degraded=is_degraded,
            trajectory_class=traj,
        ))

        # Cache the result
        rt_result = _pipeline_result_to_runtime_result(result, task)
        cache.put(task_hash, rt_result)

    # Determinism check: re-run first non-cache-hit task and verify
    determinism_pass = True
    for task in tasks:
        r1 = execute_linear_system_pipeline(task, budget=budget)
        r2 = execute_linear_system_pipeline(task, budget=budget)
        if not np.allclose(
            r1["projection"]["solution"],
            r2["projection"]["solution"],
            atol=1e-14,
        ):
            determinism_pass = False
            break

    # Correctness check: projection must improve or match surrogate
    correctness_pass = all(
        p <= s + 1e-12
        for s, p in zip(surrogate_residuals, projected_residuals)
    )

    n = len(tasks)
    avg_proj_iters = float(np.mean(projection_iterations_list)) if projection_iterations_list else 0.0
    avg_surr_res = float(np.mean(surrogate_residuals)) if surrogate_residuals else 0.0
    avg_proj_res = float(np.mean(projected_residuals)) if projected_residuals else 0.0
    avg_sched_ms = float(np.mean(scheduler_overhead_ms_list)) if scheduler_overhead_ms_list else 0.0
    avg_runtime_ms = total_runtime_ms / max(n, 1)

    report = SmokeReport(
        version="v3.2-smoke",
        seed=seed,
        sample_count=n,
        total_runtime_ms=total_runtime_ms,
        avg_runtime_ms=avg_runtime_ms,
        avg_scheduler_overhead_ms=avg_sched_ms,
        avg_projection_iterations=avg_proj_iters,
        avg_surrogate_residual=avg_surr_res,
        avg_projected_residual=avg_proj_res,
        cache_hit_rate=cache_hits / max(n, 1),
        retrieval_hit_rate=0.0,  # No FAISS in smoke mode
        degraded_rate=degraded_count / max(n, 1),
        correctness_pass=correctness_pass,
        determinism_pass=determinism_pass,
        overall_pass=correctness_pass and determinism_pass,
        samples=[asdict(s) for s in sample_results],
    )

    return report


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def format_text_report(report: SmokeReport) -> str:
    """Format the smoke report as human-readable text."""
    lines = [
        "=" * 60,
        "CORE v3.2 CI Smoke Benchmark",
        "=" * 60,
        f"  Version:            {report.version}",
        f"  Seed:               {report.seed}",
        f"  Samples:            {report.sample_count}",
        "-" * 60,
        f"  Total runtime:      {report.total_runtime_ms:.2f} ms",
        f"  Avg runtime:        {report.avg_runtime_ms:.2f} ms",
        f"  Avg scheduler OH:   {report.avg_scheduler_overhead_ms:.3f} ms",
        f"  Avg proj iters:     {report.avg_projection_iterations:.1f}",
        f"  Avg surr residual:  {report.avg_surrogate_residual:.2e}",
        f"  Avg proj residual:  {report.avg_projected_residual:.2e}",
        "-" * 60,
        f"  Cache hit rate:     {report.cache_hit_rate:.1%}",
        f"  Retrieval hit rate: {report.retrieval_hit_rate:.1%}",
        f"  Degraded rate:      {report.degraded_rate:.1%}",
        "-" * 60,
        f"  Correctness:        {'PASS' if report.correctness_pass else 'FAIL'}",
        f"  Determinism:        {'PASS' if report.determinism_pass else 'FAIL'}",
        f"  Overall:            {'PASS' if report.overall_pass else 'FAIL'}",
        "=" * 60,
    ]
    return "\n".join(lines)


def format_json_report(report: SmokeReport) -> str:
    """Format the smoke report as JSON."""
    return json.dumps(asdict(report), indent=2, default=str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_compat_runtime_task(ls_task: LinearSystemTask) -> RuntimeTask:
    """Convert a LinearSystemTask to a backend-compatible RuntimeTask."""
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


def _pipeline_result_to_runtime_result(
    pipeline_result: dict,
    task: LinearSystemTask,
) -> "RuntimeResult":
    """Convert a pipeline result dict to a RuntimeResult for caching.

    The linear_system pipeline does not expose oracle_solution directly
    in its result dict (only via evaluation). We store the projected
    solution as the primary result, consistent with Principle 3.
    """
    import torch
    from backend.core_runtime.task_runtime import RuntimeResult

    proj_sol = pipeline_result["projection"]["solution"]
    surr_pred = pipeline_result["surrogate"]["prediction"]

    return RuntimeResult(
        task_id=task.task_id,
        task_fingerprint=pipeline_result["trace"]["fingerprint"],
        oracle_voltages=torch.tensor(proj_sol, dtype=torch.float32),
        surrogate_voltages=torch.tensor(surr_pred, dtype=torch.float32),
        projected_voltages=torch.tensor(proj_sol, dtype=torch.float32),
        projection_result=None,
        evaluation_report=None,
        memory_entry=None,
        total_runtime_ms=pipeline_result["trace"].get("projection_runtime_ms", 0.0)
                       + pipeline_result["trace"].get("surrogate_runtime_ms", 0.0),
        oracle_runtime_ms=0.0,
        surrogate_runtime_ms=pipeline_result["trace"].get("surrogate_runtime_ms", 0.0),
        projection_runtime_ms=pipeline_result["trace"].get("projection_runtime_ms", 0.0),
        failure_type=None,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CORE v3.2 CI Smoke Benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--samples", type=int, default=10, help="Number of tasks")
    parser.add_argument("--budget", type=int, default=50, help="Projection budget")
    parser.add_argument("--output", choices=["text", "json"], default="text",
                        help="Output format")
    parser.add_argument("--output-dir", default=None, help="Artifact directory")
    args = parser.parse_args()

    report = run_smoke_benchmark(
        seed=args.seed,
        sample_count=args.samples,
        budget=args.budget,
        output_dir=args.output_dir,
    )

    if args.output == "json":
        print(format_json_report(report))
    else:
        print(format_text_report(report))

    # Save report to file
    if args.output_dir:
        out_path = Path(args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "smoke_benchmark_report.json", "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"\nReport saved to: {out_path / 'smoke_benchmark_report.json'}")

    # Exit code
    sys.exit(0 if report.overall_pass else 1)


if __name__ == "__main__":
    main()
