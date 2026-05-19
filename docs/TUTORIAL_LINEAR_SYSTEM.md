# Linear System Walkthrough Tutorial

This tutorial walks you through the CORE runtime engine using the
`linear_system` domain — a simple, self-contained domain that requires
no GPU and no circuit dependencies.

## Prerequisites

```bash
pip install core-runtime-engine[linear-system]
```

Or from source:

```bash
pip install -e ".[linear-system]"
```

## Running the Tutorial

```bash
python examples/01_linear_system_walkthrough.py
```

Expected output: deterministic, no errors, all 6 steps complete.

## What You Will Learn

### Step 1: Define a DomainTask

A `DomainTask` is the basic unit of work in CORE. It carries:

- `task_id`: unique identifier
- `domain_name`: which domain handles this task
- `input_artifact`: a string key or path for the input
- `metadata`: domain-specific data (for linear_system: matrix A, vector b)

```python
from core_runtime.domains.linear_system import LinearSystemTask
import numpy as np

A = np.array([[4.0, 1.0, 0.5],
              [1.0, 3.0, 0.8],
              [0.5, 0.8, 2.0]])
b = np.array([5.5, 4.8, 3.3])

task = LinearSystemTask(
    task_id="my_task_001",
    domain_name="linear_system",
    input_artifact="my_task_001",
    metadata={"A": A, "b": b},
)
```

### Step 2: The Oracle Protocol

The oracle is the authoritative solver. For linear systems, it uses
`numpy.linalg.solve` (direct LU decomposition). The oracle provides
ground truth for evaluation.

```python
from core_runtime.domains.linear_system import LinearSystemOracle

oracle = LinearSystemOracle()
result = oracle.solve(task)
solution = result["solution"]  # The exact answer
```

### Step 3: Run the Runtime Pipeline

The full pipeline orchestrates:

1. **Surrogate**: fast approximate solution
2. **Projection**: iterative refinement toward truth
3. **Evaluation**: compare projected vs oracle
4. **Confidence**: assess result quality

```python
from core_runtime.domains.linear_system import execute_linear_system_pipeline

result = execute_linear_system_pipeline(task, budget=100)
# result contains: surrogate, projection, evaluation, confidence, trace
```

### Step 4: Inspect the Execution Trace

Every runtime execution produces an **immutable trace** — a record of
what happened, how, and with what results.

```python
trace = result["trace"]
print(trace["fingerprint"])        # Task hash
print(trace["surrogate_method"])   # Which surrogate was used
print(trace["projection_iterations"])  # How many iterations
print(trace["evaluation_correct"])     # Did it match oracle?
```

### Step 5: Read the Evaluation Report

The evaluator compares the projected solution against the oracle:

```python
eval_result = result["evaluation"]
print(eval_result["correct"])     # Boolean: match oracle?
print(eval_result["residual"])    # ||Ax_projected - b||
```

### Step 6: Memory Registration

CORE has three memory tiers:

| Tier | Name | Purpose |
|------|------|---------|
| 1 | ExactMatchCache | Hash-based exact replay |
| 2 | RetrievalMemory | Similarity-based warm-start |
| 3 | ExperienceMemory | Embedding-based retrieval |

```python
from core_runtime.core.memory.exact_cache import ExactMatchCache
from core_runtime.core.specs.task_hashing import compute_task_hash

cache = ExactMatchCache(base_dir="/tmp/my_cache")
task_hash = compute_task_hash(compat_task)

# Check cache
if cache.contains(task_hash):
    cached_result = cache.get(task_hash)  # Instant replay!

# Register result
entry = cache.put(task_hash, runtime_result)
```

## Key Principles

1. **Determinism**: Same input + same seed → same output, always
2. **Exact cache first**: Known tasks are replayed instantly
3. **Projection is final authority**: The projected solution is the answer
4. **Warmstart never bypasses projection**: Warm-start only provides a better initial point
5. **Traces are immutable**: Every execution is fully auditable

## Architecture Overview

```
DomainTask → [Surrogate] → approximate solution
                   ↓
            [Projection] → refined solution (FINAL)
                   ↓
           [Evaluation] → compare vs oracle
                   ↓
           [Confidence] → quality assessment
                   ↓
              [Trace] → immutable execution record
                   ↓
             [Memory] → cache for future use
```

## Next Steps

- Run the smoke benchmark: `python scripts/run_smoke_benchmark.py`
- Explore deterministic fuzzing: `python scripts/fuzz_runtime_deterministic.py`
- Read the principles document: `docs/CORE_PRINCIPLES.md`
