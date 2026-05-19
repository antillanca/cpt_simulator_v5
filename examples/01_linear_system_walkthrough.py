#!/usr/bin/env python3
"""CORE v3.2 — Linear System Walkthrough Tutorial.

A minimal end-to-end tutorial for new developers showing how to:
  1. Define a DomainTask
  2. Implement an OracleProtocol
  3. Run the runtime
  4. Inspect the trace
  5. Read the evaluation report
  6. See memory registration

This example runs without GPU and without circuit dependencies.
It is fully deterministic: same input always produces same output.

Usage:
  python examples/01_linear_system_walkthrough.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on the path so core_runtime is importable
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from core_runtime.domains.linear_system import (
    LinearSystemEvaluator,
    LinearSystemOracle,
    LinearSystemProjection,
    LinearSystemSurrogate,
    LinearSystemTask,
    execute_linear_system_pipeline,
)
from core_runtime.core.specs.task_hashing import compute_task_hash
from backend.core_runtime.task_runtime import RuntimeTask


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Step 1: Define a DomainTask
# ---------------------------------------------------------------------------

def step1_define_task() -> LinearSystemTask:
    """Show how to create a linear system task.

    A DomainTask carries:
      - task_id: unique identifier
      - domain_name: which domain handles this task
      - input_artifact: a string key or path for the input
      - metadata: domain-specific data (here: matrix A, vector b)
    """
    section("Step 1: Define a DomainTask")

    # Create a 3x3 symmetric positive-definite system Ax = b
    A = np.array([
        [4.0, 1.0, 0.5],
        [1.0, 3.0, 0.8],
        [0.5, 0.8, 2.0],
    ])

    b = np.array([5.5, 4.8, 3.3])

    task = LinearSystemTask(
        task_id="tutorial_task_001",
        domain_name="linear_system",
        input_artifact="tutorial_task_001",
        metadata={"A": A, "b": b},
    )

    print(f"Created task: {task.task_id}")
    print(f"  Domain:     {task.domain_name}")
    print(f"  Matrix A:\n{A}")
    print(f"  Vector b:   {b}")
    print(f"  System size: {A.shape[0]}x{A.shape[1]}")

    return task


# ---------------------------------------------------------------------------
# Step 2: Implement an OracleProtocol
# ---------------------------------------------------------------------------

def step2_oracle(task: LinearSystemTask) -> None:
    """Show how the oracle produces the ground-truth solution.

    The oracle is the authoritative solver. For linear systems,
    it uses numpy.linalg.solve (direct LU decomposition).
    """
    section("Step 2: The Oracle Protocol")

    oracle = LinearSystemOracle()
    result = oracle.solve(task)
    solution = result["solution"]

    print(f"Oracle solution (ground truth):")
    print(f"  x = {solution}")
    print(f"  Residual ||Ax - b|| = {np.linalg.norm(task.metadata['A'] @ solution - task.metadata['b']):.2e}")
    print(f"  Oracle method: {result['method']}")

    # Verify: the oracle's solution should satisfy Ax = b
    residual = np.linalg.norm(task.metadata["A"] @ solution - task.metadata["b"])
    print(f"  Oracle residual (should be ~0): {residual:.2e}")


# ---------------------------------------------------------------------------
# Step 3: Run the Runtime
# ---------------------------------------------------------------------------

def step3_run_runtime(task: LinearSystemTask) -> dict:
    """Run the full pipeline: surrogate -> projection -> evaluation.

    The runtime orchestrates:
      1. Surrogate: fast approximate solution
      2. Projection: iterative refinement toward truth
      3. Evaluation: compare projected vs oracle
      4. Confidence: assess result quality
    """
    section("Step 3: Run the Runtime Pipeline")

    result = execute_linear_system_pipeline(task, budget=100)

    print("Pipeline executed successfully!")
    print(f"  Surrogate method:  {result['surrogate']['method']}")
    print(f"  Surrogate residual: {result['surrogate']['residual']:.6e}")
    print(f"  Projection method: {result['projection']['method']}")
    print(f"  Projection iters:  {result['projection']['iterations']}")
    print(f"  Projection residual: {result['projection']['residual']:.6e}")
    print(f"  Converged:         {result['projection']['converged']}")

    return result


# ---------------------------------------------------------------------------
# Step 4: Inspect the Trace
# ---------------------------------------------------------------------------

def step4_inspect_trace(result: dict) -> None:
    """Show how to read the execution trace.

    Every runtime execution produces a trace — an immutable record
    of what happened, how, and with what results.
    """
    section("Step 4: Inspect the Execution Trace")

    trace = result["trace"]

    print("Trace fields:")
    for key, value in sorted(trace.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.6e}")
        else:
            print(f"  {key}: {value}")

    print(f"\nKey trace properties:")
    print(f"  Fingerprint (task hash):  {trace['fingerprint'][:16]}...")
    print(f"  Surrogate method:        {trace['surrogate_method']}")
    print(f"  Projection iterations:   {trace['projection_iterations']}")
    print(f"  Evaluation correct:      {trace['evaluation_correct']}")
    print(f"  Trajectory length:       {trace['trajectory_length']}")


# ---------------------------------------------------------------------------
# Step 5: Read the Evaluation Report
# ---------------------------------------------------------------------------

def step5_evaluation_report(result: dict, task: LinearSystemTask) -> None:
    """Show how to read the evaluation report.

    The evaluator compares the projected solution against the oracle
    and reports correctness, residual, and accuracy metrics.
    """
    section("Step 5: Read the Evaluation Report")

    eval_result = result["evaluation"]

    print(f"Evaluation results:")
    print(f"  Correct:     {eval_result.get('correct', 'N/A')}")
    print(f"  Residual:    {eval_result.get('residual', 'N/A'):.6e}")
    print(f"  Error norm:  {np.linalg.norm(eval_result.get('solution', np.array([])) - eval_result.get('oracle_solution', np.array([]))):.6e}" if 'solution' in eval_result and 'oracle_solution' in eval_result else "")

    # Direct comparison
    proj_sol = result["projection"]["solution"]
    oracle_result = LinearSystemOracle().solve(task)
    oracle_sol = oracle_result["solution"]
    error = np.linalg.norm(proj_sol - oracle_sol)

    print(f"\nDirect comparison:")
    print(f"  Oracle solution:   {oracle_sol}")
    print(f"  Projected solution: {proj_sol}")
    print(f"  ||projected - oracle||: {error:.6e}")
    print(f"  Relative error:     {error / np.linalg.norm(oracle_sol):.6e}")


# ---------------------------------------------------------------------------
# Step 6: See Memory Registration
# ---------------------------------------------------------------------------

def step6_memory_registration(task: LinearSystemTask) -> None:
    """Show how tasks interact with the memory system.

    The runtime has three memory tiers:
      1. ExactMatchCache (Tier 1): hash-based exact replay
      2. RetrievalMemory (Tier 2): similarity-based warm-start
      3. ExperienceMemory (Tier 3): embedding-based retrieval

    Here we demonstrate exact cache behavior.
    """
    section("Step 6: Memory Registration (Exact Cache)")

    from core_runtime.core.memory.exact_cache import ExactMatchCache
    from backend.core_runtime.task_runtime import RuntimeResult
    import torch

    # Build a compatible RuntimeTask for hashing
    metadata = {}
    for k, v in task.metadata.items():
        if isinstance(v, np.ndarray):
            metadata[k] = v.tolist()
        else:
            metadata[k] = v

    compat_task = RuntimeTask(
        task_id=task.task_id,
        domain=task.domain_name,
        input_artifact=task.input_artifact,
        oracle_name="LinearSystemOracle",
        surrogate_name="LinearSystemSurrogate",
        projection_enabled=True,
        metadata=metadata,
    )

    # Compute the task hash
    task_hash = compute_task_hash(compat_task)
    print(f"Task hash: {task_hash[:24]}...")

    # Create a temporary exact cache
    cache = ExactMatchCache(base_dir="/tmp/cpt_tutorial_cache")

    # Check: not cached yet
    hit_before = cache.contains(task_hash)
    print(f"Cache hit before registration: {hit_before}")

    # Run the pipeline and get results
    result = execute_linear_system_pipeline(task, budget=100)
    proj_sol = result["projection"]["solution"]

    # Register in the exact cache
    rt_result = RuntimeResult(
        task_id=task.task_id,
        task_fingerprint=result["trace"]["fingerprint"],
        oracle_voltages=torch.tensor(proj_sol, dtype=torch.float32),
        surrogate_voltages=torch.tensor(result["surrogate"]["prediction"], dtype=torch.float32),
        projected_voltages=torch.tensor(proj_sol, dtype=torch.float32),
        projection_result=None,
        evaluation_report=None,
        memory_entry=None,
        total_runtime_ms=1.0,
        oracle_runtime_ms=0.0,
        surrogate_runtime_ms=0.5,
        projection_runtime_ms=0.5,
        failure_type=None,
    )
    # Register in the exact cache
    entry = cache.put(task_hash, rt_result)
    print(f"Registered entry: {entry.task_hash[:24]}...")
    print(f"  Projection iterations: {entry.projection_iterations}")

    # Check: now it should be cached
    hit_after = cache.contains(task_hash)
    print(f"Cache hit after registration: {hit_after}")

    # Retrieve the cached result
    cached = cache.get(task_hash)
    if cached is not None:
        print(f"Retrieved from cache: task_id={cached.task_id}")
    else:
        print("  Cache retrieval returned None (unexpected)")

    print(f"\n  Key principle: Exact cache always first (Principle 2)")
    print(f"  Same task hash -> same cached result -> no re-computation")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full tutorial walkthrough."""
    print("=" * 60)
    print("  CORE v3.2 — Linear System Walkthrough Tutorial")
    print("=" * 60)
    print()
    print("This tutorial demonstrates the core runtime engine")
    print("using a simple linear system domain.")
    print()
    print("No GPU required. No circuit dependencies.")
    print("Fully deterministic: same input -> same output.")

    # Step 1: Define a task
    task = step1_define_task()

    # Step 2: Oracle
    step2_oracle(task)

    # Step 3: Run the runtime
    result = step3_run_runtime(task)

    # Step 4: Inspect the trace
    step4_inspect_trace(result)

    # Step 5: Evaluation report
    step5_evaluation_report(result, task)

    # Step 6: Memory registration
    step6_memory_registration(task)

    print(f"\n{'=' * 60}")
    print("  Tutorial complete!")
    print(f"{'=' * 60}")
    print()
    print("Key takeaways:")
    print("  1. DomainTask carries domain-specific data in metadata")
    print("  2. Oracle provides ground-truth solutions")
    print("  3. Runtime orchestrates surrogate -> projection -> evaluation")
    print("  4. Every execution produces an immutable trace")
    print("  5. Evaluation compares projected vs oracle solutions")
    print("  6. Exact cache enables zero-cost replay for known tasks")
    print()
    print("Next steps:")
    print("  - Read docs/TUTORIAL_LINEAR_SYSTEM.md for detailed explanation")
    print("  - Run the smoke benchmark: python scripts/run_smoke_benchmark.py")
    print("  - Explore the fuzzing script: python scripts/fuzz_runtime_deterministic.py")


if __name__ == "__main__":
    main()
