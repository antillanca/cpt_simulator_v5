#!/usr/bin/env python3
"""CPT v2.15 — Operational Experience Export.

Exports runtime statistics for v2.16 consumption:
- Trajectory statistics
- Family statistics
- Convergence distributions
- Escalation distributions
- Warmstart effectiveness
- Retrieval effectiveness
- Budget allocation statistics

Output formats: JSONL + CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend.runtime.projection_experience import ProjectionExperienceMemory
from backend.runtime.retrieval_memory import RetrievalMemory


def export_family_stats(proj_mem: ProjectionExperienceMemory) -> dict:
    """Export topology family statistics."""
    return proj_mem.all_family_stats()


def export_trajectory_distribution(proj_mem: ProjectionExperienceMemory) -> dict:
    """Export trajectory class distribution."""
    # Note: trajectory_class was added to ProjectionExperienceEntry in v2.15
    # For entries created before v2.15, we classify from residual_slope
    distribution: dict[str, int] = {
        "fast_converging": 0,
        "stable_linear": 0,
        "oscillatory": 0,
        "stalled": 0,
        "divergence_risk": 0,
        "retrieval_assisted": 0,
    }

    for entry in proj_mem._entries:
        # Infer trajectory class from available data
        if not entry.converged and entry.iterations >= 20:
            distribution["divergence_risk"] += 1
        elif entry.used_warmstart and entry.converged and entry.final_residual < 0.01:
            distribution["retrieval_assisted"] += 1
        elif entry.residual_slope <= 0:
            distribution["stalled"] += 1
        elif entry.residual_slope > 0.01:
            distribution["fast_converging"] += 1
        else:
            distribution["stable_linear"] += 1

    return distribution


def export_convergence_distribution(proj_mem: ProjectionExperienceMemory) -> dict:
    """Export convergence rate distribution per topology family."""
    stats = proj_mem.all_family_stats()
    result = {}
    for family, data in stats.items():
        result[family] = {
            "convergence_rate": data.get("convergence_rate", 0),
            "avg_iterations": data.get("avg_iterations", 0),
            "avg_residual_slope": data.get("avg_residual_slope", 0),
            "warmstart_usage_rate": data.get("warmstart_usage_rate", 0),
            "count": data.get("count", 0),
        }
    return result


def export_warmstart_effectiveness(proj_mem: ProjectionExperienceMemory) -> dict:
    """Export warmstart effectiveness statistics."""
    ws_entries = [e for e in proj_mem._entries if e.used_warmstart]
    std_entries = [e for e in proj_mem._entries if not e.used_warmstart]

    ws_avg_iters = sum(e.iterations for e in ws_entries) / max(len(ws_entries), 1)
    std_avg_iters = sum(e.iterations for e in std_entries) / max(len(std_entries), 1)
    ws_conv = sum(1 for e in ws_entries if e.converged) / max(len(ws_entries), 1)
    std_conv = sum(1 for e in std_entries if e.converged) / max(len(std_entries), 1)

    return {
        "warmstart_sample_count": len(ws_entries),
        "standard_sample_count": len(std_entries),
        "warmstart_avg_iterations": ws_avg_iters,
        "standard_avg_iterations": std_avg_iters,
        "iterations_saved_avg": std_avg_iters - ws_avg_iters,
        "warmstart_convergence_rate": ws_conv,
        "standard_convergence_rate": std_conv,
    }


def export_retrieval_effectiveness(ret_mem: RetrievalMemory) -> dict:
    """Export retrieval memory effectiveness statistics."""
    stats = ret_mem.stats()
    return {
        "total_entries": stats.get("total_entries", 0),
        "unique_topologies": stats.get("unique_topologies", 0),
        "unique_task_hashes": stats.get("unique_task_hashes", 0),
        "avg_confidence": stats.get("avg_confidence", 0),
        "avg_projection_iterations": stats.get("avg_projection_iterations", 0),
    }


def export_budget_allocation(proj_mem: ProjectionExperienceMemory) -> dict:
    """Export budget allocation statistics (inferred from experience)."""
    iters = [e.iterations for e in proj_mem._entries]
    if not iters:
        return {"count": 0}

    iters_sorted = sorted(iters)
    n = len(iters_sorted)
    return {
        "count": n,
        "min_iterations": min(iters),
        "max_iterations": max(iters),
        "avg_iterations": sum(iters) / n,
        "median_iterations": iters_sorted[n // 2],
        "p25_iterations": iters_sorted[max(0, n // 4)],
        "p75_iterations": iters_sorted[min(n - 1, 3 * n // 4)],
        "buckets": {
            "1-5": sum(1 for i in iters if i <= 5),
            "6-10": sum(1 for i in iters if 5 < i <= 10),
            "11-20": sum(1 for i in iters if 10 < i <= 20),
            "21-50": sum(1 for i in iters if 20 < i <= 50),
            "50+": sum(1 for i in iters if i > 50),
        },
    }


def export_escalation_stats(proj_mem: ProjectionExperienceMemory) -> dict:
    """Export escalation-related statistics."""
    diverged = [e for e in proj_mem._entries if not e.converged]
    families = {}
    for e in diverged:
        families[e.topology_family] = families.get(e.topology_family, 0) + 1

    return {
        "total_diverged": len(diverged),
        "divergence_rate": len(diverged) / max(len(proj_mem._entries), 1),
        "divergence_by_family": families,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export operational experience (v2.15)")
    parser.add_argument("--proj-experience-dir", required=True, help="ProjectionExperienceMemory directory")
    parser.add_argument("--retrieval-dir", required=True, help="RetrievalMemory directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for exports")
    parser.add_argument("--format", choices=["jsonl", "csv", "both"], default="both")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    proj_mem = ProjectionExperienceMemory(args.proj_experience_dir)
    ret_mem = RetrievalMemory(args.retrieval_dir)

    # Compute exports
    exports = {
        "family_stats": export_family_stats(proj_mem),
        "trajectory_distribution": export_trajectory_distribution(proj_mem),
        "convergence_distribution": export_convergence_distribution(proj_mem),
        "warmstart_effectiveness": export_warmstart_effectiveness(proj_mem),
        "retrieval_effectiveness": export_retrieval_effectiveness(ret_mem),
        "budget_allocation": export_budget_allocation(proj_mem),
        "escalation_stats": export_escalation_stats(proj_mem),
    }

    # JSONL output
    if args.format in ("jsonl", "both"):
        jsonl_path = output_dir / "operational_experience.jsonl"
        with open(jsonl_path, "w") as f:
            for category, data in exports.items():
                record = {"category": category, "data": data}
                f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        print(f"JSONL exported to: {jsonl_path}")

    # CSV output (flattened)
    if args.format in ("csv", "both"):
        csv_path = output_dir / "operational_experience.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "key", "value"])
            for category, data in exports.items():
                _flatten_to_csv(writer, category, data)
        print(f"CSV exported to: {csv_path}")

    # Also write a single summary JSON
    summary_path = output_dir / "operational_experience_summary.json"
    with open(summary_path, "w") as f:
        json.dump(exports, f, indent=2, sort_keys=True, default=str)
    print(f"Summary exported to: {summary_path}")

    print(f"\nExport complete. {proj_mem.count} projection entries, {ret_mem.stats().get('total_entries', 0)} retrieval entries.")


def _flatten_to_csv(writer: csv.writer, category: str, data: dict, prefix: str = "") -> None:
    """Recursively flatten a dict to CSV rows."""
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _flatten_to_csv(writer, category, value, full_key)
        elif isinstance(value, (list, tuple)):
            writer.writerow([category, full_key, json.dumps(value)])
        else:
            writer.writerow([category, full_key, str(value)])


if __name__ == "__main__":
    main()
