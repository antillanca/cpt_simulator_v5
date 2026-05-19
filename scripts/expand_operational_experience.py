#!/usr/bin/env python3
"""CORE v3.2 — Expand Operational Experience Dataset.

Grow the operational experience from ~300 to a larger reproducible set
using existing stable runtime behavior. NO training — only data collection.

Outputs:
  data/operational_experience/manifest.json
  data/operational_experience/entries/<hash>.json
  data/runtime_traces/manifest.json
  data/runtime_traces/<task_id>.json
  data/benchmarks/smoke_report.json
  data/benchmarks/fuzz_report.json

Usage:
  python scripts/expand_operational_experience.py [--seed 42] [--tasks 500]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from core_runtime.core.specs.task_hashing import compute_task_hash
from core_runtime.domains.linear_system import (
    LinearSystemTask,
    execute_linear_system_pipeline,
)
from backend.core_runtime.task_runtime import RuntimeTask


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = REPO / "data"
OP_EXP_DIR = DATA_DIR / "operational_experience"
TRACES_DIR = DATA_DIR / "runtime_traces"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def generate_experience_tasks(
    seed: int = 42,
    count: int = 500,
) -> list[LinearSystemTask]:
    """Generate diverse tasks for operational experience collection."""
    master_rng = np.random.default_rng(seed)
    tasks = []

    for i in range(count):
        sub_seed = int(master_rng.integers(0, 2**31))
        rng = np.random.default_rng(sub_seed)

        size = int(rng.integers(2, 12))
        scale = float(rng.uniform(0.5, 20.0))

        # Generate SPD matrix
        A_base = rng.standard_normal((size, size))
        eigenvalues = np.sort(np.abs(rng.standard_normal(size)))
        eigenvalues = eigenvalues * scale + 0.1
        U, _, Vt = np.linalg.svd(A_base)
        A = U @ np.diag(eigenvalues) @ Vt
        A = (A + A.T) / 2.0 + size * np.eye(size)

        b = rng.standard_normal(size) * scale

        task = LinearSystemTask(
            task_id=f"opexp_{seed}_{i:05d}",
            domain_name="linear_system",
            input_artifact=f"opexp_{seed}_{i:05d}",
            metadata={
                "A": A,
                "b": b,
                "fuzz_sub_seed": sub_seed,
                "fuzz_size": size,
                "fuzz_scale": scale,
            },
        )
        tasks.append(task)

    return tasks


# ---------------------------------------------------------------------------
# SHA-256 anchoring
# ---------------------------------------------------------------------------

def sha256_anchor(data: bytes) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def anchor_json(obj: Any) -> str:
    """Anchor a JSON-serializable object with SHA-256."""
    serialized = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return sha256_anchor(serialized)


# ---------------------------------------------------------------------------
# Experience collection
# ---------------------------------------------------------------------------

@dataclass
class ExperienceStats:
    """Aggregate statistics from operational experience collection."""
    total_tasks: int = 0
    convergence_count: int = 0
    escalation_count: int = 0
    warmstart_effective_count: int = 0
    degraded_count: int = 0
    routing_distribution: dict[str, int] = field(default_factory=dict)
    avg_projection_iterations: float = 0.0
    avg_surrogate_residual: float = 0.0
    avg_projection_residual: float = 0.0
    avg_runtime_ms: float = 0.0
    runtime_distribution: dict[str, int] = field(default_factory=dict)
    family_convergence: dict[str, dict[str, Any]] = field(default_factory=dict)


def collect_experience(
    seed: int = 42,
    task_count: int = 500,
    budget: int = 100,
) -> ExperienceStats:
    """Run tasks and collect operational experience data."""
    tasks = generate_experience_tasks(seed=seed, count=task_count)

    # Create output directories
    OP_EXP_DIR.mkdir(parents=True, exist_ok=True)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    entries_dir = OP_EXP_DIR / "entries"
    entries_dir.mkdir(exist_ok=True)

    stats = ExperienceStats()
    all_proj_iters = []
    all_surr_res = []
    all_proj_res = []
    all_runtimes = []
    entry_hashes = []
    trace_hashes = []

    for task in tasks:
        t_start = time.perf_counter()
        result = execute_linear_system_pipeline(task, budget=budget)
        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        trace = result["trace"]
        stats.total_tasks += 1

        # Convergence tracking
        converged = trace["projection_converged"]
        if converged:
            stats.convergence_count += 1

        # Routing distribution
        method = trace.get("surrogate_method", "unknown")
        stats.routing_distribution[method] = stats.routing_distribution.get(method, 0) + 1

        # Family tracking by size
        size_key = str(task.metadata.get("fuzz_size", "unknown"))
        if size_key not in stats.family_convergence:
            stats.family_convergence[size_key] = {
                "total": 0, "converged": 0, "avg_iters": 0.0,
            }
        family = stats.family_convergence[size_key]
        family["total"] += 1
        if converged:
            family["converged"] += 1
        proj_iters = trace["projection_iterations"]
        family["avg_iters"] = (
            (family["avg_iters"] * (family["total"] - 1) + proj_iters)
            / family["total"]
        )

        # Runtime distribution
        rt_bucket = "fast" if t_elapsed < 1.0 else ("medium" if t_elapsed < 10.0 else "slow")
        stats.runtime_distribution[rt_bucket] = stats.runtime_distribution.get(rt_bucket, 0) + 1

        all_proj_iters.append(proj_iters)
        all_surr_res.append(trace["surrogate_residual"])
        all_proj_res.append(trace["projection_residual"])
        all_runtimes.append(t_elapsed)

        # Degraded tracking
        eval_correct = trace["evaluation_correct"]
        if not eval_correct and not converged:
            stats.degraded_count += 1

        # Save experience entry
        entry = {
            "task_id": task.task_id,
            "task_hash": trace["fingerprint"],
            "domain": "linear_system",
            "converged": converged,
            "projection_iterations": proj_iters,
            "surrogate_residual": float(trace["surrogate_residual"]),
            "projection_residual": float(trace["projection_residual"]),
            "evaluation_correct": eval_correct,
            "confidence_score": float(trace.get("confidence_score", 0.0)),
            "runtime_ms": t_elapsed,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        entry_sha = anchor_json(entry)
        entry_hashes.append(entry_sha)
        entry_path = entries_dir / f"{entry_sha[:16]}.json"
        if not entry_path.exists():
            with open(entry_path, "w") as f:
                json.dump(entry, f, indent=2, sort_keys=True, default=str)

        # Save trace
        trace_entry = {
            "task_id": task.task_id,
            "trace": trace,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        trace_sha = anchor_json(trace_entry)
        trace_hashes.append(trace_sha)
        trace_path = TRACES_DIR / f"{task.task_id}.json"
        if not trace_path.exists():
            with open(trace_path, "w") as f:
                json.dump(trace_entry, f, indent=2, sort_keys=True, default=str)

    # Compute averages
    if stats.total_tasks > 0:
        stats.avg_projection_iterations = float(np.mean(all_proj_iters))
        stats.avg_surrogate_residual = float(np.mean(all_surr_res))
        stats.avg_projection_residual = float(np.mean(all_proj_res))
        stats.avg_runtime_ms = float(np.mean(all_runtimes))

    # Write manifests
    op_manifest = {
        "version": "v3.2-opexp",
        "seed": seed,
        "task_count": task_count,
        "total_entries": len(entry_hashes),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entry_hashes": entry_hashes,
        "manifest_sha256": "",  # filled below
    }
    op_manifest["manifest_sha256"] = anchor_json(op_manifest)
    with open(OP_EXP_DIR / "manifest.json", "w") as f:
        json.dump(op_manifest, f, indent=2, default=str)

    trace_manifest = {
        "version": "v3.2-traces",
        "seed": seed,
        "task_count": task_count,
        "total_traces": len(trace_hashes),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trace_hashes": trace_hashes,
        "manifest_sha256": "",  # filled below
    }
    trace_manifest["manifest_sha256"] = anchor_json(trace_manifest)
    with open(TRACES_DIR / "manifest.json", "w") as f:
        json.dump(trace_manifest, f, indent=2, default=str)

    # Save stats
    stats_path = BENCHMARKS_DIR / "operational_stats.json"
    with open(stats_path, "w") as f:
        json.dump(asdict(stats), f, indent=2, default=str)

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CORE v3.2 — Expand Operational Experience"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", type=int, default=500)
    parser.add_argument("--budget", type=int, default=100)
    args = parser.parse_args()

    print(f"Collecting operational experience: {args.tasks} tasks, seed={args.seed}")
    stats = collect_experience(seed=args.seed, task_count=args.tasks, budget=args.budget)

    print(f"\nResults:")
    print(f"  Total tasks:        {stats.total_tasks}")
    print(f"  Converged:          {stats.convergence_count} ({100*stats.convergence_count/max(1,stats.total_tasks):.1f}%)")
    print(f"  Degraded:           {stats.degraded_count}")
    print(f"  Avg proj iters:     {stats.avg_projection_iterations:.1f}")
    print(f"  Avg surr residual:  {stats.avg_surrogate_residual:.2e}")
    print(f"  Avg proj residual:  {stats.avg_projection_residual:.2e}")
    print(f"  Avg runtime:        {stats.avg_runtime_ms:.2f} ms")
    print(f"\n  Routing distribution: {stats.routing_distribution}")
    print(f"  Runtime distribution: {stats.runtime_distribution}")
    print(f"\n  Output:")
    print(f"    {OP_EXP_DIR}/")
    print(f"    {TRACES_DIR}/")
    print(f"    {BENCHMARKS_DIR}/")


if __name__ == "__main__":
    main()
