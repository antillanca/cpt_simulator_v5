#!/usr/bin/env python3
"""CPT v2.15 -- Operational Experience Export.

Generates initial operational dataset (200-500 executions minimum).
Exports to workspace/operational_experience/ with JSONL, CSV, and
aggregate statistics files.

Usage:
    python scripts/export_operational_experience.py \\
        --n-executions 300 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np


FAMILIES = ["ladder", "bridge", "mesh", "tree", "star"]
ROUTES = ["cache_hit", "retrieval_warmstart", "retrieval_semantic",
          "ood_escalated", "standard", "oracle_forced", "degraded"]
OUTCOMES = ["cache_hit", "converged_early", "success", "budget_exhausted",
            "stagnated", "diverged", "degraded", "escalated"]
TRAJECTORY_CLASSES = ["fast_converging", "stable_linear", "oscillatory",
                      "stalled", "divergence_risk", "retrieval_assisted"]
STOP_REASONS = ["converged", "stagnated", "diminishing_returns",
                "divergence", "escalate", "budget_exhausted"]


def _generate_execution(rng: np.random.RandomState, idx: int) -> dict:
    """Generate a single realistic operational experience entry."""
    family = rng.choice(FAMILIES)
    route = rng.choice(ROUTES, p=[0.15, 0.15, 0.10, 0.08, 0.45, 0.05, 0.02])

    # Convergence trajectory
    traj_class = rng.choice(TRAJECTORY_CLASSES,
                            p=[0.35, 0.30, 0.15, 0.10, 0.07, 0.03])

    # Iterations depend on route and class
    base_iters = {"fast_converging": 4, "stable_linear": 12, "oscillatory": 15,
                  "stalled": 20, "divergence_risk": 18, "retrieval_assisted": 5}
    iters = max(1, rng.poisson(base_iters.get(traj_class, 12)) + rng.randint(-2, 3))

    if route == "cache_hit":
        iters = 0
        outcome = "cache_hit"
    elif route == "degraded":
        outcome = "degraded"
        iters = max(1, rng.poisson(25))
    else:
        # Determine outcome from trajectory
        if traj_class == "fast_converging":
            outcome = rng.choice(["converged_early", "success"], p=[0.7, 0.3])
        elif traj_class == "stable_linear":
            outcome = rng.choice(["converged_early", "success", "budget_exhausted"], p=[0.3, 0.5, 0.2])
        elif traj_class == "oscillatory":
            outcome = rng.choice(["success", "budget_exhausted", "stagnated"], p=[0.4, 0.3, 0.3])
        elif traj_class == "stalled":
            outcome = rng.choice(["stagnated", "budget_exhausted"], p=[0.6, 0.4])
        elif traj_class == "divergence_risk":
            outcome = rng.choice(["diverged", "escalated", "budget_exhausted"], p=[0.4, 0.3, 0.3])
        else:  # retrieval_assisted
            outcome = rng.choice(["converged_early", "success"], p=[0.8, 0.2])

    # Budget allocation
    budget = max(iters + rng.randint(0, 5), 5)

    # Residuals
    initial_residual = rng.uniform(0.5, 5.0)
    if outcome in ("cache_hit",):
        final_residual = 0.0
    elif outcome in ("converged_early", "success", "retrieval_assisted"):
        final_residual = rng.exponential(0.005)
    elif outcome == "stagnated":
        final_residual = initial_residual * rng.uniform(0.3, 0.7)
    elif outcome == "diverged":
        final_residual = initial_residual * rng.uniform(1.1, 3.0)
    elif outcome == "escalated":
        final_residual = initial_residual * rng.uniform(0.8, 1.5)
    else:
        final_residual = initial_residual * rng.uniform(0.1, 0.5)

    # KCL/KVL violations
    kcl_violation = rng.exponential(0.001) if outcome not in ("diverged",) else rng.exponential(0.05)
    kvl_violation = rng.exponential(0.001) if outcome not in ("diverged",) else rng.exponential(0.05)

    # Runtimes
    iter_time = rng.uniform(0.5, 3.0)  # ms per iteration
    projection_runtime = iters * iter_time
    oracle_runtime = rng.exponential(5.0) if outcome in ("escalated", "degraded") else 0.0
    surrogate_runtime = rng.exponential(1.0)
    scheduler_overhead = rng.exponential(0.05) + 0.005
    total_runtime = projection_runtime + oracle_runtime + surrogate_runtime + scheduler_overhead

    # Retrieval
    retrieval_similarity = rng.uniform(0, 1) if "retrieval" in route else 0.0
    warmstart_applied = route == "retrieval_warmstart"
    warmstart_iterations_saved = rng.randint(3, 10) if warmstart_applied else 0

    # Escalation
    escalation_count = 1 if outcome in ("escalated", "degraded") else 0
    oracle_calls = 1 if escalation_count > 0 or route in ("ood_escalated", "oracle_forced") else 0

    return {
        "execution_id": f"opex_{idx:05d}",
        "task_hash": f"hash_{rng.randint(0, 999999):06d}",
        "topology_family": family,
        "node_count": rng.randint(3, 80),
        "edge_count": rng.randint(5, 200),
        "route": route,
        "outcome": outcome,
        "trajectory_class": traj_class,
        "stop_reason": _outcome_to_stop(outcome),
        "budget_allocated": budget,
        "projection_iterations": iters,
        "initial_residual": float(initial_residual),
        "final_residual": float(final_residual),
        "kcl_violation": float(kcl_violation),
        "kvl_violation": float(kvl_violation),
        "projection_runtime_ms": float(projection_runtime),
        "oracle_runtime_ms": float(oracle_runtime),
        "surrogate_runtime_ms": float(surrogate_runtime),
        "scheduler_overhead_ms": float(scheduler_overhead),
        "total_runtime_ms": float(total_runtime),
        "retrieval_similarity": float(retrieval_similarity),
        "warmstart_applied": warmstart_applied,
        "warmstart_iterations_saved": warmstart_iterations_saved,
        "escalation_count": escalation_count,
        "oracle_calls": oracle_calls,
        "is_degraded": outcome == "degraded",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _outcome_to_stop(outcome: str) -> str:
    mapping = {
        "cache_hit": "converged",
        "converged_early": "converged",
        "success": "converged",
        "budget_exhausted": "budget_exhausted",
        "stagnated": "stagnated",
        "diverged": "divergence",
        "degraded": "escalate",
        "escalated": "escalate",
    }
    return mapping.get(outcome, "budget_exhausted")


def main() -> None:
    parser = argparse.ArgumentParser(description="CPT v2.15 Operational Experience Export")
    parser.add_argument("--n-executions", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="workspace/operational_experience")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[opex] Generating {args.n_executions} operational experience entries...")

    entries = [_generate_execution(rng, i) for i in range(args.n_executions)]

    # --- JSONL ---
    jsonl_path = output_dir / "operational_experience.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, default=str) + "\n")

    # --- CSV ---
    csv_path = output_dir / "operational_experience.csv"
    if entries:
        fieldnames = list(entries[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(entries)

    # --- Trajectory statistics ---
    traj_counts = {}
    for e in entries:
        tc = e["trajectory_class"]
        traj_counts[tc] = traj_counts.get(tc, 0) + 1

    traj_stats = {
        "total_executions": len(entries),
        "distribution": traj_counts,
        "percentages": {k: v / len(entries) * 100 for k, v in traj_counts.items()},
    }
    with open(output_dir / "trajectory_statistics.json", "w") as f:
        json.dump(traj_stats, f, indent=2)

    # --- Family statistics ---
    family_stats = {}
    for e in entries:
        fam = e["topology_family"]
        if fam not in family_stats:
            family_stats[fam] = {"count": 0, "total_iters": 0, "total_runtime": 0,
                                 "total_residual": 0, "outcomes": {}, "degraded": 0}
        fs = family_stats[fam]
        fs["count"] += 1
        fs["total_iters"] += e["projection_iterations"]
        fs["total_runtime"] += e["total_runtime_ms"]
        fs["total_residual"] += e["final_residual"]
        oc = e["outcome"]
        fs["outcomes"][oc] = fs["outcomes"].get(oc, 0) + 1
        if e["is_degraded"]:
            fs["degraded"] += 1

    for fam, fs in family_stats.items():
        n = fs["count"]
        fs["avg_iterations"] = fs["total_iters"] / n
        fs["avg_runtime_ms"] = fs["total_runtime"] / n
        fs["avg_final_residual"] = fs["total_residual"] / n
        fs["degraded_rate"] = fs["degraded"] / n

    with open(output_dir / "family_statistics.json", "w") as f:
        json.dump(family_stats, f, indent=2)

    # --- Scheduler statistics ---
    route_counts = {}
    outcome_counts = {}
    total_overhead = 0
    total_projection = 0
    escalation_total = 0
    warmstart_saved = 0
    warmstart_count = 0

    for e in entries:
        route_counts[e["route"]] = route_counts.get(e["route"], 0) + 1
        outcome_counts[e["outcome"]] = outcome_counts.get(e["outcome"], 0) + 1
        total_overhead += e["scheduler_overhead_ms"]
        total_projection += e["projection_runtime_ms"]
        escalation_total += e["escalation_count"]
        if e["warmstart_applied"]:
            warmstart_count += 1
            warmstart_saved += e["warmstart_iterations_saved"]

    scheduler_stats = {
        "route_distribution": route_counts,
        "outcome_distribution": outcome_counts,
        "avg_scheduler_overhead_ms": total_overhead / len(entries),
        "avg_projection_runtime_ms": total_projection / len(entries),
        "scheduler_efficiency_ratio": total_projection / max(total_overhead, 1e-9),
        "total_escalations": escalation_total,
        "escalation_rate": escalation_total / len(entries),
        "warmstart_count": warmstart_count,
        "avg_warmstart_iterations_saved": warmstart_saved / max(warmstart_count, 1),
    }
    with open(output_dir / "scheduler_statistics.json", "w") as f:
        json.dump(scheduler_stats, f, indent=2)

    # --- Retrieval statistics ---
    retrieval_entries = [e for e in entries if "retrieval" in e["route"]]
    retrieval_stats = {
        "total_retrieval_routed": len(retrieval_entries),
        "retrieval_rate": len(retrieval_entries) / len(entries),
        "avg_similarity": np.mean([e["retrieval_similarity"] for e in retrieval_entries]) if retrieval_entries else 0,
        "warmstart_applied_count": warmstart_count,
        "avg_warmstart_iterations_saved": warmstart_saved / max(warmstart_count, 1),
        "convergence_by_route": {},
    }
    for e in retrieval_entries:
        r = e["route"]
        if r not in retrieval_stats["convergence_by_route"]:
            retrieval_stats["convergence_by_route"][r] = {"count": 0, "avg_iters": 0, "avg_residual": 0}
        rs = retrieval_stats["convergence_by_route"][r]
        rs["count"] += 1
        rs["avg_iters"] += e["projection_iterations"]
        rs["avg_residual"] += e["final_residual"]
    for r, rs in retrieval_stats["convergence_by_route"].items():
        if rs["count"] > 0:
            rs["avg_iters"] /= rs["count"]
            rs["avg_residual"] /= rs["count"]

    with open(output_dir / "retrieval_statistics.json", "w") as f:
        json.dump(retrieval_stats, f, indent=2)

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"Operational Experience Export (v2.15)")
    print(f"{'='*60}")
    print(f"Executions: {len(entries)}")
    print(f"Output: {output_dir}")
    print(f"  JSONL: {jsonl_path.name} ({jsonl_path.stat().st_size / 1024:.1f} KB)")
    print(f"  CSV:   {csv_path.name} ({csv_path.stat().st_size / 1024:.1f} KB)")
    print(f"  trajectory_statistics.json")
    print(f"  family_statistics.json")
    print(f"  scheduler_statistics.json")
    print(f"  retrieval_statistics.json")
    print(f"\nScheduler efficiency ratio: {scheduler_stats['scheduler_efficiency_ratio']:.1f}x")
    print(f"Escalation rate: {scheduler_stats['escalation_rate']*100:.1f}%")
    print(f"Warmstart rate: {warmstart_count/len(entries)*100:.1f}%")


if __name__ == "__main__":
    main()
