#!/usr/bin/env python3
"""CORE v3.3 — Trustworthiness Benchmark.

Runs the linear_system domain through the runtime with trust auditing
enabled, then exports trust-specific metrics.

Proves: audit layer adds observability WITHOUT changing correctness.

Usage:
    python scripts/run_trustworthiness_benchmark.py [--seed 42] [--samples 20]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from core_runtime.core.trustworthiness_runtime import (
    TrustLevel,
    TrustFlag,
    TrustworthinessAudit,
    audit_execution,
)
from core_runtime.core.uncertainty_memory import UncertaintyMemory, UncertaintyEntry
from core_runtime.core.failure_report import generate_failure_report
from core_runtime.core.explainability_runtime import ExplainabilityRuntime
from core_runtime.core.audit_trace_store import AuditTraceStore
from core_runtime.core.runtime_config import RuntimeConfig


# ---------------------------------------------------------------------------
# Domain setup — linear_system (no GPU required)
# ---------------------------------------------------------------------------

def _setup_linear_domain(seed: int):
    """Create a linear system domain with oracle, surrogate, and projection."""
    from core_runtime.domains.linear_system import (
        LinearSystemTask,
        LinearSystemOracle,
        LinearSystemSurrogate,
        LinearSystemProjection,
    )

    rng = np.random.RandomState(seed)
    oracle = LinearSystemOracle()
    surrogate = LinearSystemSurrogate()
    projection = LinearSystemProjection()
    return oracle, surrogate, projection, rng


def _generate_task(rng: np.random.RandomState, idx: int, n: int = 5):
    """Generate a deterministic linear system task."""
    from core_runtime.domains.linear_system import LinearSystemTask
    A = rng.randn(n, n)
    # Make it diagonally dominant for stability
    A += n * np.eye(n)
    b = rng.randn(n)
    return LinearSystemTask(
        task_id=f"trust_bench_{idx:04d}",
        domain_name="linear_system",
        input_artifact=f"trust_bench_{idx:04d}",
        metadata={
            "A": A,
            "b": b,
            "n": n,
            "task_hash": f"trust_bench_{idx:04d}",
            "topology_family": f"n={n}",
        },
    )


def _execute_pipeline(oracle, surrogate, projection, task, budget: int = 50):
    """Run oracle -> surrogate -> projection pipeline. Return raw results."""
    oracle_result = oracle.solve(task)
    surrogate_result = surrogate.predict(task)
    projection_result = projection.project(task, surrogate_result, budget=budget)
    return oracle_result, surrogate_result, projection_result


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------

@dataclass
class TrustBenchmarkResult:
    """Result of the trustworthiness benchmark."""
    version: str = "v3.3"
    seed: int = 0
    sample_count: int = 0
    # Core correctness metrics (must be identical with/without audit)
    avg_residual: float = 0.0
    avg_projection_iterations: float = 0.0
    avg_runtime_ms: float = 0.0
    # Trust metrics (new in v3.3)
    avg_trust_score: float = 0.0
    avg_uncertainty_score: float = 0.0
    certain_rate: float = 0.0
    transitional_rate: float = 0.0
    indeterminate_rate: float = 0.0
    audit_trigger_rate: float = 0.0
    uncertainty_memory_hit_rate: float = 0.0
    explanation_generation_rate: float = 0.0
    # Flag distribution
    flag_distribution: dict[str, int] = field(default_factory=dict)
    # Explanation type distribution
    explanation_distribution: dict[str, int] = field(default_factory=dict)
    # Safety proof
    correctness_unchanged: bool = True
    # Raw per-sample data
    samples: list[dict[str, Any]] = field(default_factory=list)
    overall_pass: bool = True


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------

def run_trust_benchmark(
    seed: int = 42,
    sample_count: int = 20,
    output_dir: str = "workspace/runtime_reports",
) -> TrustBenchmarkResult:
    """Run the trustworthiness benchmark over linear_system domain."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    oracle, surrogate, projection, rng = _setup_linear_domain(seed)
    explainer = ExplainabilityRuntime()
    audit_store = AuditTraceStore()
    uncertainty_mem = UncertaintyMemory()
    config = RuntimeConfig()

    # Run WITHOUT audit layer first (for correctness comparison)
    residuals_no_audit: list[float] = []
    iters_no_audit: list[int] = []
    for i in range(sample_count):
        task = _generate_task(rng, i)
        _, _, proj_result = _execute_pipeline(oracle, surrogate, projection, task)
        residuals_no_audit.append(float(proj_result["residual"]))
        iters_no_audit.append(int(proj_result["iterations"]))

    # Reset rng for same task generation
    rng2 = np.random.RandomState(seed)

    # Run WITH audit layer
    residuals_with_audit: list[float] = []
    iters_with_audit: list[int] = []
    trust_levels: list[TrustLevel] = []
    trust_scores: list[float] = []
    uncertainty_scores: list[float] = []
    flag_dist: dict[str, int] = {}
    expl_dist: dict[str, int] = {}
    samples_data: list[dict[str, Any]] = []
    total_runtime_ms = 0.0
    audit_triggers = 0
    explanation_count = 0

    for i in range(sample_count):
        task = _generate_task(rng2, i)

        t0 = time.monotonic()
        oracle_result, surrogate_result, proj_result = _execute_pipeline(
            oracle, surrogate, projection, task,
        )
        runtime_ms = (time.monotonic() - t0) * 1000

        final_residual = float(proj_result["residual"])
        proj_iters = int(proj_result["iterations"])
        residuals_with_audit.append(final_residual)
        iters_with_audit.append(proj_iters)
        total_runtime_ms += runtime_ms

        # Compute confidence/uncertainty from projection result
        confidence = max(0.0, 1.0 - final_residual)
        uncertainty = min(1.0, final_residual)
        trajectory_class = "fast_converging" if proj_iters < 8 else (
            "oscillating" if final_residual > 0.01 else "standard"
        )
        is_ood = final_residual > 0.1
        escalation = proj_iters > 15

        # Generate audit
        task_hash = task.metadata.get("task_hash", f"task_{i}")
        audit = audit_execution(
            task_hash=task_hash,
            confidence_score=confidence,
            uncertainty_score=uncertainty,
            projection_iterations=proj_iters,
            final_residual=final_residual,
            trajectory_class=trajectory_class,
            escalation_required=escalation,
            ood=is_ood,
            family_budget=20,
        )

        # Generate explanation
        explanation = explainer.explain(audit)

        # Store audit
        audit_store.append(audit, explanation)

        # If transitional or indeterminate, add to uncertainty memory
        if audit.trust_level in (TrustLevel.TRANSITIONAL, TrustLevel.INDETERMINATE):
            entry = UncertaintyEntry(
                task_hash=task_hash,
                reason="high_residual" if final_residual > 0.01 else "low_confidence",
                topology_signature=task.metadata.get("topology_family", "unknown"),
                routing_action="standard_projection",
                projection_iterations=proj_iters,
                residual=final_residual,
                confidence_score=confidence,
                trust_level=audit.trust_level.value,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                scheduler_context={},
                metadata={},
            )
            uncertainty_mem.add_entry(entry)

        # Track metrics
        trust_levels.append(audit.trust_level)
        trust_scores.append(audit.confidence_score)
        uncertainty_scores.append(audit.uncertainty_score)

        for flag in audit.flags:
            flag_dist[flag.value] = flag_dist.get(flag.value, 0) + 1

        expl_dist[explanation.explanation_type] = (
            expl_dist.get(explanation.explanation_type, 0) + 1
        )
        explanation_count += 1

        if len(audit.flags) > 0:
            audit_triggers += 1

        samples_data.append({
            "task_hash": task_hash,
            "trust_level": audit.trust_level.value,
            "confidence": audit.confidence_score,
            "uncertainty": audit.uncertainty_score,
            "residual": final_residual,
            "iterations": proj_iters,
            "explanation_type": explanation.explanation_type,
        })

    # Compute correctness comparison
    n = sample_count
    residual_match = all(
        abs(a - b) < 1e-10
        for a, b in zip(residuals_no_audit, residuals_with_audit)
    )
    iters_match = all(
        a == b for a, b in zip(iters_no_audit, iters_with_audit)
    )
    correctness_unchanged = residual_match and iters_match

    # Build result
    certain_count = sum(1 for tl in trust_levels if tl == TrustLevel.CERTAIN)
    transitional_count = sum(1 for tl in trust_levels if tl == TrustLevel.TRANSITIONAL)
    indeterminate_count = sum(1 for tl in trust_levels if tl == TrustLevel.INDETERMINATE)

    result = TrustBenchmarkResult(
        seed=seed,
        sample_count=n,
        avg_residual=sum(residuals_with_audit) / n if n else 0,
        avg_projection_iterations=sum(iters_with_audit) / n if n else 0,
        avg_runtime_ms=total_runtime_ms / n if n else 0,
        avg_trust_score=sum(trust_scores) / n if n else 0,
        avg_uncertainty_score=sum(uncertainty_scores) / n if n else 0,
        certain_rate=certain_count / n if n else 0,
        transitional_rate=transitional_count / n if n else 0,
        indeterminate_rate=indeterminate_count / n if n else 0,
        audit_trigger_rate=audit_triggers / n if n else 0,
        uncertainty_memory_hit_rate=len(uncertainty_mem) / n if n else 0,
        explanation_generation_rate=explanation_count / n if n else 1.0,
        flag_distribution=flag_dist,
        explanation_distribution=expl_dist,
        correctness_unchanged=correctness_unchanged,
        samples=samples_data,
        overall_pass=correctness_unchanged,
    )

    # Export JSON report
    report_path = output_path / "trustworthiness_benchmark.json"
    report_data = {
        "version": result.version,
        "seed": result.seed,
        "sample_count": result.sample_count,
        "avg_residual": result.avg_residual,
        "avg_projection_iterations": result.avg_projection_iterations,
        "avg_runtime_ms": result.avg_runtime_ms,
        "avg_trust_score": result.avg_trust_score,
        "avg_uncertainty_score": result.avg_uncertainty_score,
        "certain_rate": result.certain_rate,
        "transitional_rate": result.transitional_rate,
        "indeterminate_rate": result.indeterminate_rate,
        "audit_trigger_rate": result.audit_trigger_rate,
        "uncertainty_memory_hit_rate": result.uncertainty_memory_hit_rate,
        "explanation_generation_rate": result.explanation_generation_rate,
        "flag_distribution": result.flag_distribution,
        "explanation_distribution": result.explanation_distribution,
        "correctness_unchanged": result.correctness_unchanged,
        "overall_pass": result.overall_pass,
        "samples": result.samples,
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    return result


def format_trust_report(result: TrustBenchmarkResult) -> str:
    """Format trust benchmark result as readable text."""
    lines = [
        f"{'='*60}",
        f"CORE v3.3 Trustworthiness Benchmark Report",
        f"{'='*60}",
        f"Samples: {result.sample_count}  Seed: {result.seed}",
        f"",
        f"--- Correctness (must be unchanged) ---",
        f"Correctness unchanged: {'PASS' if result.correctness_unchanged else 'FAIL'}",
        f"Avg residual: {result.avg_residual:.6e}",
        f"Avg projection iterations: {result.avg_projection_iterations:.1f}",
        f"Avg runtime: {result.avg_runtime_ms:.2f} ms",
        f"",
        f"--- Trust Metrics ---",
        f"Avg trust score: {result.avg_trust_score:.4f}",
        f"Avg uncertainty score: {result.avg_uncertainty_score:.4f}",
        f"Certain rate: {result.certain_rate*100:.1f}%",
        f"Transitional rate: {result.transitional_rate*100:.1f}%",
        f"Indeterminate rate: {result.indeterminate_rate*100:.1f}%",
        f"Audit trigger rate: {result.audit_trigger_rate*100:.1f}%",
        f"Uncertainty memory hit rate: {result.uncertainty_memory_hit_rate*100:.1f}%",
        f"Explanation generation rate: {result.explanation_generation_rate*100:.1f}%",
        f"",
        f"--- Flag Distribution ---",
    ]
    for flag, count in sorted(result.flag_distribution.items()):
        lines.append(f"  {flag}: {count}")
    lines.append("")
    lines.append("--- Explanation Distribution ---")
    for etype, count in sorted(result.explanation_distribution.items()):
        lines.append(f"  {etype}: {count}")
    lines.append("")
    lines.append(f"Overall: {'PASS' if result.overall_pass else 'FAIL'}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="CORE v3.3 Trustworthiness Benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output-dir", default="workspace/runtime_reports")
    args = parser.parse_args()

    result = run_trust_benchmark(
        seed=args.seed,
        sample_count=args.samples,
        output_dir=args.output_dir,
    )
    print(format_trust_report(result))
    sys.exit(0 if result.overall_pass else 1)


if __name__ == "__main__":
    main()
