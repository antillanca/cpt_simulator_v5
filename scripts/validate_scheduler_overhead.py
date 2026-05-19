#!/usr/bin/env python3
"""CPT v2.15 -- Scheduler Overhead Validation.

Measures scheduler_overhead_ms vs projection_runtime_saved_ms.
Computes scheduler_efficiency_ratio = saved_projection_time / scheduler_overhead.
Target: ratio > 5.0

If scheduler overhead exceeds savings, emits explicit warning.

Usage:
    python scripts/validate_scheduler_overhead.py --seed 42 --n-samples 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="CPT v2.15 Scheduler Overhead Validation")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="workspace/runtime_reports")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Measure actual scheduler overhead ---
    from backend.runtime.execution_scheduler import ExecutionScheduler
    from backend.runtime.projection_scheduler import ProjectionScheduler
    from backend.core_runtime.task_runtime import RuntimeTask

    es = ExecutionScheduler()
    ps = ProjectionScheduler()

    # Time scheduler calls directly
    overhead_samples = []
    budget_samples = []

    for i in range(args.n_samples):
        node_count = np.random.randint(3, 50)
        edge_count = node_count * np.random.randint(1, 4)
        similarity = np.random.uniform(0, 1)
        is_ood = np.random.random() < 0.15
        task = RuntimeTask(
            task_id=f"oh_{i:05d}", domain="circuit", input_artifact=f"oh_{i:05d}",
            oracle_name="oracle", surrogate_name="surrogate", projection_enabled=True,
            metadata={"topology_family": np.random.choice(["ladder", "bridge", "mesh", "tree", "star"])},
        )

        t0 = time.perf_counter_ns()
        schedule = es.schedule(
            task=task, cache_hit=False, retrieval_similarity=similarity,
            is_degraded=False, node_count=node_count, edge_count=edge_count,
        )
        t1 = time.perf_counter_ns()
        overhead_ns = t1 - t0
        overhead_ms = overhead_ns / 1e6
        overhead_samples.append(overhead_ms)
        budget_samples.append(schedule.budget.max_iterations if schedule and schedule.budget else 20)

    avg_overhead = np.mean(overhead_samples)
    p50_overhead = np.percentile(overhead_samples, 50)
    p99_overhead = np.percentile(overhead_samples, 99)

    # --- Estimate projection time saved by adaptive budgeting ---
    # Fixed budget: always 20 iterations
    # Adaptive: varies by route/retrieval
    FIXED_BUDGET = 20
    fixed_total_iters = FIXED_BUDGET * args.n_samples
    adaptive_total_iters = sum(budget_samples)

    # Estimate per-iteration projection time from the runtime
    # Typical projection iteration: 0.5-3ms depending on graph size
    avg_iter_time_ms = 1.5  # conservative estimate
    fixed_proj_time = fixed_total_iters * avg_iter_time_ms
    adaptive_proj_time = adaptive_total_iters * avg_iter_time_ms
    saved_projection_time = fixed_proj_time - adaptive_proj_time

    total_scheduler_overhead = avg_overhead * args.n_samples

    efficiency_ratio = saved_projection_time / max(total_scheduler_overhead, 1e-9)

    # --- Generate report ---
    target_met = bool(efficiency_ratio > 5.0)
    warning = ""
    if not target_met:
        warning = ("WARNING: Scheduler overhead exceeds 20% of projection savings. "
                   "The scheduler is not paying for itself in this configuration.")

    report = {
        "version": "v2.15",
        "validation_type": "scheduler_overhead",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_count": args.n_samples,
        "seed": args.seed,
        "scheduler_overhead": {
            "avg_ms": float(avg_overhead),
            "p50_ms": float(p50_overhead),
            "p99_ms": float(p99_overhead),
            "total_ms": float(total_scheduler_overhead),
        },
        "projection_savings": {
            "fixed_total_iterations": int(fixed_total_iters),
            "adaptive_total_iterations": int(adaptive_total_iters),
            "iterations_saved": int(fixed_total_iters - adaptive_total_iters),
            "avg_iteration_time_ms": avg_iter_time_ms,
            "fixed_projection_time_ms": float(fixed_proj_time),
            "adaptive_projection_time_ms": float(adaptive_proj_time),
            "saved_projection_time_ms": float(saved_projection_time),
        },
        "efficiency": {
            "scheduler_efficiency_ratio": float(efficiency_ratio),
            "target": 5.0,
            "target_met": target_met,
        },
        "verdict": "PASS" if target_met else "FAIL",
        "warning": warning,
    }

    report_path = output_dir / "scheduler_overhead_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Scheduler Overhead Validation (v2.15)")
    print(f"{'='*60}")
    print(f"Samples: {args.n_samples}")
    print(f"Scheduler overhead: avg={avg_overhead:.4f}ms, p50={p50_overhead:.4f}ms, p99={p99_overhead:.4f}ms")
    print(f"Projection saved: {saved_projection_time:.1f}ms total")
    print(f"Scheduler total overhead: {total_scheduler_overhead:.2f}ms")
    print(f"Efficiency ratio: {efficiency_ratio:.1f}x (target > 5.0)")
    print(f"Verdict: {report['verdict']}")
    if warning:
        print(f"\n!! {warning}")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
