#!/usr/bin/env python3
"""CPT v2.13 — Canonical Runtime Benchmark Runner.

The NEW standard benchmark runner. Executes the full runtime pipeline:
 task -> oracle -> surrogate -> projection -> evaluation -> memory

v2.13 adds:
 - ExactMatchCache integration (cache_hit_rate)
 - ConfidenceRuntime estimation (confidence_calibration)
 - CapabilityRouter decisions (routing distribution)
 - RecoveryHandler tracking (degraded_execution_rate)
 - Atomic memory persistence

Produces:
 - EvaluationReport per sample
 - MemoryEntry per sample
 - ExecutionTrace per sample
 - ExactCacheEntry per sample
 - Aggregate summary with v2.13 metrics

Usage:
 python scripts/run_runtime_benchmark.py \\
   --dataset workspace/datasets/circuit_v29f_10k.pt \\
   --checkpoint workspace/checkpoints/gnn_v29f.pt \\
   [--no-projection] \\
   [--seed 42] \\
   [--max-samples 100] \\
   [--output-dir workspace/runtime_benchmarks]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure repo root on sys.path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch

from backend.core_runtime.task_runtime import RuntimeTask, RuntimeExecutor
from backend.core_runtime.oracle_protocol import MNAOracleAdapter
from backend.core_runtime.surrogate_runtime import SurrogateRuntime
from backend.core_runtime.projection_runtime import ProjectionRuntime
from backend.core_runtime.memory_runtime import MemoryRuntime
from backend.core_runtime.execution_trace import ExecutionTrace, TraceStore, make_trace_id
from backend.core_runtime.dataset_registry import DatasetManifest, DatasetRegistry, compute_dataset_sha256
from backend.core_runtime.exact_cache import ExactMatchCache
from backend.core_runtime.task_hashing import compute_task_hash
from backend.core_runtime.execution_policy import ExecutionPolicy, RecoveryHandler
from backend.core_runtime.confidence_runtime import ConfidenceRuntime
from backend.core_runtime.capability_router import CapabilityRouter

# Failure taxonomy — handle import gracefully
try:
    from backend.circuits.failure_analysis import classify_failure
except ImportError:
    def classify_failure(mae, report, topology="unknown"):
        if mae > 1.0:
            return "large_error"
        return "acceptable"


def main() -> None:
    parser = argparse.ArgumentParser(description="CPT Runtime Benchmark Runner (v2.13)")
    parser.add_argument("--dataset", required=True, help="Path to dataset (.pt)")
    parser.add_argument("--checkpoint", default=None, help="Path to surrogate checkpoint")
    parser.add_argument("--no-projection", action="store_true", help="Disable projection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="workspace/runtime_benchmarks")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    print(f"[benchmark] Loading dataset: {args.dataset}")
    data_path = Path(args.dataset)
    if not data_path.exists():
        data_path = REPO / args.dataset

    samples = []
    if data_path.exists() and data_path.suffix == ".pt":
        loaded = torch.load(str(data_path), weights_only=False)
        if isinstance(loaded, dict):
            samples = loaded.get("samples", loaded.get("graphs", []))
        elif isinstance(loaded, (list, tuple)):
            samples = loaded
        else:
            samples = [loaded]
    else:
        # Try CircuitGraphDataset
        try:
            from backend.circuits.graph_dataset import CircuitGraphDataset
            dataset = CircuitGraphDataset(root=REPO / "workspace" / "datasets")
            samples = list(dataset)
        except Exception:
            print(f"[benchmark] Dataset not found at {data_path}, using empty")

    if args.max_samples and len(samples) > args.max_samples:
        samples = samples[:args.max_samples]

    print(f"[benchmark] Samples: {len(samples)}")

    # --- Register dataset manifest ---
    ds_sha256 = compute_dataset_sha256(data_path, seed=args.seed) if data_path.exists() else "no_dataset"
    manifest = DatasetManifest(
        dataset_id=f"ds_{ds_sha256[:12]}",
        sha256=ds_sha256,
        sample_count=len(samples),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        domain="circuit",
    )
    registry = DatasetRegistry(str(output_dir / "dataset_registry.jsonl"))
    registry.register(manifest)

    # --- Setup oracle ---
    oracle = MNAOracleAdapter()

    # --- Setup surrogate ---
    surrogate = SurrogateRuntime(name="circuit_gnn")
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            print(f"[benchmark] Loading checkpoint: {ckpt_path}")
            try:
                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                from backend.neural.models.circuit_gnn import CircuitGNN
                model = CircuitGNN()
                if "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                elif isinstance(ckpt, dict):
                    model.load_state_dict(ckpt)
                surrogate = SurrogateRuntime(model=model, name="circuit_gnn")
            except Exception as e:
                print(f"[benchmark] WARNING: Could not load checkpoint: {e}")

    # --- Setup projection ---
    projection = None
    if not args.no_projection:
        try:
            from backend.circuits.physics_projection import ProjectionConfig
            proj_config = ProjectionConfig()
            projection = ProjectionRuntime(proj_config)
        except ImportError:
            print("[benchmark] Projection not available, disabling")

    # --- Setup memory & trace ---
    memory = MemoryRuntime(str(output_dir / "memory"))
    trace_store = TraceStore(str(output_dir / "traces"))

    # --- v2.13: Setup cache, confidence, router, recovery ---
    cache = ExactMatchCache(str(output_dir / "exact_cache"))
    confidence_rt = ConfidenceRuntime()
    router = CapabilityRouter()
    policy = ExecutionPolicy()
    recovery = RecoveryHandler(policy)

    # --- Build executor ---
    executor = RuntimeExecutor(
        oracle=oracle,
        surrogate=surrogate,
        projection=projection,
        memory_sink=memory,
    )

    # --- Execute benchmark ---
    print(f"[benchmark] Running pipeline...")
    results = []
    total_mae_oracle = 0.0
    total_mae_projected = 0.0
    total_proj_iters = 0
    failure_counts: dict[str, int] = {}
    degraded_count = 0
    cache_hits = 0
    routing_actions: dict[str, int] = {}
    confidence_scores = []
    total_runtime_ms = 0.0

    for idx, sample in enumerate(samples):
        task_id = f"bench_{idx:05d}"

        # Resolve circuit + graph
        circuit = sample.circuit if hasattr(sample, "circuit") else None
        graph = sample

        if circuit is not None:
            oracle.register_circuit(task_id, circuit)

        topo_family = getattr(graph, "topology_family", "unknown")

        task = RuntimeTask(
            task_id=task_id,
            domain="circuit",
            input_artifact=task_id,
            oracle_name=oracle.name(),
            surrogate_name=surrogate.name,
            projection_enabled=not args.no_projection,
            metadata={"sample_idx": idx, "dataset_id": manifest.dataset_id, "topology_family": topo_family},
        )

        # v2.13: Check exact cache first
        task_hash = compute_task_hash(task)
        cached = cache.get(task_hash)
        if cached is not None:
            result = cached
            cache_hits += 1
            routing_actions["cache_hit"] = routing_actions.get("cache_hit", 0) + 1
        else:
            # v2.13: Estimate confidence & route
            graph_size = getattr(graph, "num_nodes", 0) if hasattr(graph, "num_nodes") else 0
            confidence = confidence_rt.estimate(
                task, graph_size=graph_size, topology_family=topo_family,
            )
            confidence_scores.append(confidence.confidence_score)

            decision = router.route(task, confidence, cache_hit=False)
            routing_actions[decision.action] = routing_actions.get(decision.action, 0) + 1

            # Execute pipeline
            result = executor.execute(task)

            # v2.13: Check for degradation
            if result.failure_type is not None:
                degraded_count += 1

            # v2.13: Cache result
            cache.put(task_hash, result)

            # v2.13: Record outcome for router
            if result.failure_type is not None:
                router.record_failure(topo_family)
            else:
                router.record_success(topo_family)

        # Compute MAE if we have oracle voltages
        mae_oracle = 0.0
        mae_projected = 0.0
        surr_v = result.surrogate_voltages
        if hasattr(surr_v, "prediction"):
            surr_v = surr_v.prediction
        if result.oracle_voltages is not None and surr_v is not None:
            if isinstance(surr_v, torch.Tensor) and isinstance(result.oracle_voltages, torch.Tensor):
                mae_oracle = float((surr_v - result.oracle_voltages).abs().mean())
                total_mae_oracle += mae_oracle

        proj_v = result.projected_voltages
        if proj_v is not None and result.oracle_voltages is not None:
            if isinstance(proj_v, torch.Tensor):
                mae_projected = float((proj_v - result.oracle_voltages).abs().mean())
                total_mae_projected += mae_projected

        proj_iters = result.projection_result.iterations if result.projection_result else 0
        total_proj_iters += proj_iters

        # Classify failure
        failure = None
        if mae_oracle > 0.5:
            failure = classify_failure(mae_oracle, {}, topology=topo_family)
            failure_counts[failure] = failure_counts.get(failure, 0) + 1

        # Save trace
        trace = ExecutionTrace(
            trace_id=make_trace_id(),
            task_id=task_id,
            runtime_ms=result.total_runtime_ms,
            oracle_runtime_ms=result.oracle_runtime_ms,
            surrogate_runtime_ms=result.surrogate_runtime_ms,
            projection_runtime_ms=result.projection_runtime_ms,
            projection_iterations=proj_iters,
            topology_family=topo_family,
            failure_type=failure,
        )
        trace_store.save(trace)

        total_runtime_ms += result.total_runtime_ms

        results.append({
            "task_id": task_id,
            "mae_oracle": mae_oracle,
            "mae_projected": mae_projected,
            "proj_iters": proj_iters,
            "runtime_ms": result.total_runtime_ms,
            "degraded": result.failure_type is not None,
            "cache_hit": cached is not None,
        })

    # --- Print summary ---
    n = len(results)
    print(f"\n{'='*60}")
    print(f"CPT Runtime Benchmark Summary (v2.13)")
    print(f"{'='*60}")
    print(f"Dataset: {manifest.dataset_id} ({n} samples)")
    print(f"Oracle: {oracle.name()}")
    print(f"Surrogate: {surrogate.name}")
    print(f"Projection: {'enabled' if projection else 'disabled'}")
    print(f"")
    if n > 0:
        print(f"Avg Oracle MAE: {total_mae_oracle/n:.6f} V")
        print(f"Avg Projected MAE: {total_mae_projected/n:.6f} V")
        print(f"Avg Projection Iters: {total_proj_iters/n:.1f}")
        print(f"Avg Runtime: {total_runtime_ms/n:.2f} ms")

        # v2.13 metrics
        print(f"\n--- v2.13 Metrics ---")
        print(f"Cache Hit Rate: {cache_hits/n*100:.1f}%")
        print(f"Degraded Execution Rate: {degraded_count/n*100:.1f}%")
        if confidence_scores:
            print(f"Avg Confidence: {sum(confidence_scores)/len(confidence_scores):.4f}")
        print(f"\nRouting Distribution:")
        for action, cnt in sorted(routing_actions.items()):
            print(f"  {action}: {cnt} ({cnt/n*100:.1f}%)")
        if failure_counts:
            print(f"\nFailure Distribution:")
            for ft, cnt in sorted(failure_counts.items()):
                print(f"  {ft}: {cnt}")

    # --- Save aggregate results ---
    summary_path = output_dir / "benchmark_summary.json"
    summary = {
        "version": "v2.13",
        "dataset_id": manifest.dataset_id,
        "dataset_sha256": manifest.sha256,
        "sample_count": n,
        "seed": args.seed,
        "projection_enabled": not args.no_projection,
        "avg_oracle_mae": total_mae_oracle / n if n else 0,
        "avg_projected_mae": total_mae_projected / n if n else 0,
        "avg_projection_iters": total_proj_iters / n if n else 0,
        "avg_runtime_ms": total_runtime_ms / n if n else 0,
        "cache_hit_rate": cache_hits / n if n else 0,
        "degraded_execution_rate": degraded_count / n if n else 0,
        "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
        "routing_distribution": routing_actions,
        "failure_counts": failure_counts,
        "recovery_events": len(recovery.events),
        "results": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")
    print(f"Cache saved to: {output_dir / 'exact_cache'}")
    print(f"Traces saved to: {output_dir / 'traces'}")
    print(f"Memory saved to: {output_dir / 'memory'}")


if __name__ == "__main__":
    main()
