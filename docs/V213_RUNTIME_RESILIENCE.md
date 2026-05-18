# CPT v2.13 — Runtime Resilience, Exact Cache & Confidence-Aware Routing

## Purpose

v2.13 hardens the runtime into a production-grade deterministic execution
system. The runtime now survives failures safely, caches exact matches,
routes tasks by confidence, and persists memory atomically.

This is the **stable foundation** before: retrieval memory, LoRA experts,
replay learning, KiCad integration.

## Architecture

```
                    ┌──────────────────────┐
                    │     RuntimeTask      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  compute_task_hash()  │  ← Canonical SHA-256
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   ExactMatchCache    │  ← Hit? Return immediately
                    └──────────┬───────────┘
                          Miss │
                    ┌──────────▼───────────┐
                    │  ConfidenceRuntime   │  ← Deterministic heuristics
                    │  .estimate()         │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  CapabilityRouter    │  ← Rule-based routing
                    │  .route()            │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
   cache_hit            standard          ood_escalation
   (return)         (small budget)      (large budget)
          │                    │                    │
          │           ┌────────▼────────┐           │
          │           │ RuntimeExecutor │           │
          │           │  oracle.solve() │           │
          │           │  surr.predict() │           │
          │           │  proj.project() │           │
          │           └────────┬────────┘           │
          │                    │                    │
          │           ┌────────▼────────┐           │
          │           │ RecoveryHandler │  ← Check for degradation
          │           │  .check_nan()   │
          │           │  .check_timeout │
          │           │  .check_instab. │
          │           │  .check_diverg. │
          │           └────────┬────────┘
          │                    │
          │           ┌────────▼────────┐
          │           │  MemoryRuntime  │  ← Atomic JSONL append
          │           │  .register()    │
          │           └────────┬────────┘
          │                    │
          │           ┌────────▼────────┐
          │           │  ExactMatchCache│  ← Store for future hits
          │           │  .put()         │
          │           └────────┬────────┘
          │                    │
          └────────────────────┼
                    ┌──────────▼───────────┐
                    │    RuntimeResult     │
                    │  + ExecutionTrace    │
                    └──────────────────────┘
```

## New Modules

### 1. Exact Match Cache (`exact_cache.py`)

Deterministic SHA-256 cache. Equivalent circuits produce identical hashes.

- `ExactCacheEntry`: frozen dataclass with task_hash, result_hash, topology
- `ExactMatchCache.get(task_hash)`: instant result reuse
- `ExactMatchCache.put(task_hash, result)`: persist to JSONL + JSON
- Persistence: `workspace/exact_cache/`

### 2. Canonical Task Hashing (`task_hashing.py`)

Normalizes task attributes for deterministic hashing:

- **Node ordering**: sorted alphabetically
- **Edge ordering**: sorted by (source, target)
- **Component values**: float normalization (round to 8 sig figs)
- **Configuration**: oracle/projection config included in hash
- **Schema versioning**: `HASH_SCHEMA_VERSION = "v1"`

Key functions:
- `compute_task_hash(task)`: SHA-256 of canonicalized RuntimeTask
- `compute_circuit_hash(circuit)`: SHA-256 of canonicalized Circuit
- `canonicalize_task(task)`: normalized dict (for inspection)

### 3. Execution Policy (`execution_policy.py`)

Defines runtime safety boundaries:

```python
@dataclass(frozen=True)
class ExecutionPolicy:
    oracle_timeout_s: float = 30.0
    max_retries: int = 2
    fallback_to_cache: bool = True
    projection_budget_high: int = 20
    projection_budget_low: int = 5
    surrogate_instability_threshold: float = 10.0
```

`RecoveryHandler` checks:
- **Oracle timeout**: exceeded policy limit → degraded
- **NaN output**: surrogate or projection produced NaN → degraded
- **Surrogate instability**: prediction >> oracle → degraded
- **Projection divergence**: didn't converge → degraded
- **Cache fallback**: no oracle available → degraded from cache

All degradation events are logged — **NO silent failures**.

### 4. Confidence Estimation (`confidence_runtime.py`)

Deterministic heuristic confidence. No stochastic methods.

Inputs:
- Topology family (known vs unknown)
- Graph size (small = easier)
- Resistance dynamic range
- Projection effort history
- Nearest historical failure rate
- Raw KCL residual

Output:
```python
@dataclass(frozen=True)
class ConfidenceEstimate:
    confidence_score: float        # [0, 1]
    estimated_projection_iterations: int
    likely_ood: bool
```

### 5. Capability Router (`capability_router.py`)

Rule-based routing decisions:

| Condition | Action | Budget | Force Oracle |
|-----------|--------|--------|-------------|
| Cache hit | `cache_hit` | 0 | No |
| confidence >= 0.8 | `standard` | low (5) | No |
| confidence >= 0.5 | `increased_budget` | mid (10) | No |
| likely_ood | `ood_escalation` | high (20) | Yes |
| repeated failures | `oracle_verification` | high (20) | Yes |

### 6. Atomic Memory Persistence (`memory_runtime.py` v2)

Crash-safe writes:
1. Write to temp file
2. `os.fsync()` to flush to disk
3. `os.replace()` for atomic rename

Memory compaction utility: `scripts/compact_memory_store.py`

## Degradation Reasons

| Constant | Meaning |
|----------|---------|
| `DEGRADED_ORACLE_TIMEOUT` | Oracle exceeded time limit |
| `DEGRADED_NAN_OUTPUT` | NaN detected in output tensor |
| `DEGRADED_SURROGATE_INSTABILITY` | Surrogate wildly off from oracle |
| `DEGRADED_PROJECTION_DIVERGENCE` | Projection didn't converge |
| `DEGRADED_CACHE_FALLBACK` | Using cached result (no fresh oracle) |

## Benchmark Metrics (v2.13)

| Metric | Meaning |
|--------|---------|
| `cache_hit_rate` | % exact matches from cache |
| `avg_projection_iterations` | projection effort |
| `degraded_execution_rate` | % runtime failures |
| `avg_runtime_ms` | end-to-end latency |
| `avg_confidence` | confidence calibration quality |
| `routing_distribution` | decision category counts |

Export: `workspace/runtime_benchmarks/`

## Oracle SDK

External contributors can add new domains by implementing `OracleProtocol`:

```python
class MyOracle:
    def solve(self, task_or_graph) -> dict:
        return {"voltages": tensor, "oracle_name": self.name()}

    def name(self) -> str:
        return "my_oracle"
```

See `examples/oracle_template.py` and `docs/ORACLE_SDK_GUIDE.md`.

## Intentionally Delayed

- FAISS / vector retrieval
- LoRA experts
- Replay learning / online adaptation
- KiCad plugin
- FreeCAD integration

The goal is to **harden the runtime BEFORE adding adaptive intelligence**.

## Files Created

| File | Purpose |
|------|---------|
| `backend/core_runtime/exact_cache.py` | ExactMatchCache + ExactCacheEntry |
| `backend/core_runtime/task_hashing.py` | Canonical task hashing |
| `backend/core_runtime/execution_policy.py` | ExecutionPolicy + RecoveryHandler |
| `backend/core_runtime/confidence_runtime.py` | ConfidenceEstimate + ConfidenceRuntime |
| `backend/core_runtime/capability_router.py` | CapabilityRouter + RoutingDecision |
| `backend/core_runtime/memory_runtime.py` | Atomic persistence (updated) |
| `scripts/compact_memory_store.py` | Memory compaction utility |
| `scripts/run_runtime_benchmark.py` | Extended with v2.13 metrics |
| `examples/oracle_template.py` | Working Oracle SDK example |
| `docs/ORACLE_SDK_GUIDE.md` | Oracle integration guide |
| `tests/test_v213_resilient_runtime.py` | 52 tests (E2E + unit) |

## Test Results

- v2.13 tests: 52/52 PASSED
- v2.11+v2.12+v2.13: 141/141 PASSED (no regression)
