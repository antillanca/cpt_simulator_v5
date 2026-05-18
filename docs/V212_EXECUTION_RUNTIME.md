# CPT v2.12 — Execution Runtime & Task Standardization

## Purpose

v2.12 builds the canonical execution engine that ALL future CPT domains
(circuits, KiCad, FreeCAD, math, logic, language) will use. It does NOT
improve MAE or add AI — it **standardizes the execution pipeline**.

## Architecture

```
RuntimeTask
    │
    ▼
┌─────────────────────────────────────┐
│         RuntimeExecutor             │
│                                     │
│  1. OracleProtocol.solve()          │
│  2. SurrogateProtocol.predict()     │
│  3. ProjectionProtocol.project()    │  ← optional
│  4. EvaluatorProtocol.evaluate()    │  ← optional
│  5. MemoryRuntime.register()        │  ← optional
│                                     │
│  → RuntimeResult                    │
│  → ExecutionTrace (persisted)       │
└─────────────────────────────────────┘
```

## Modules

| Module | Purpose |
|--------|---------|
| `task_runtime.py` | RuntimeTask, RuntimeResult, RuntimeExecutor, Protocols |
| `oracle_protocol.py` | OracleProtocol, MNAOracleAdapter |
| `surrogate_runtime.py` | SurrogateRuntime, SurrogatePrediction |
| `projection_runtime.py` | ProjectionRuntime, ProjectionExecution |
| `memory_runtime.py` | MemoryRuntime (JSONL persistence) |
| `execution_trace.py` | ExecutionTrace, TraceStore |
| `dataset_registry.py` | DatasetManifest, DatasetRegistry, SHA-256 hashing |

## Key Design Decisions

### 1. Domain-Agnostic Protocols
All protocols (Oracle, Surrogate, Projection, Evaluator) use `Any` for
domain-specific types. The runtime doesn't know about circuits — only
the adapters do.

### 2. Frozen Dataclasses
Every result type is `frozen=True`. No mutation after construction.
This guarantees deterministic fingerprints.

### 3. JSONL Persistence
Memory and traces use line-delimited JSON. No database dependency.
FAISS can be added later without schema changes.

### 4. Deterministic Fingerprinting
Every contract (RuntimeTask, SurrogatePrediction, ExecutionTrace,
DatasetManifest) produces a SHA-256 fingerprint from sorted JSON.
Same input → same hash, always.

### 5. Adapter Pattern
Existing systems (MNA solver, CircuitGNN, PhysicsProjection) are
wrapped via adapters, not modified. Zero regression risk.

## Pipeline Execution Order

```
RuntimeTask → Oracle.solve() → Surrogate.predict() → Projection.project()
→ Evaluator.evaluate() → Memory.register() → RuntimeResult
```

Each step is timed. ExecutionTrace records all latencies.

## Future Compatibility

| Future Feature | How v2.12 Supports It |
|---------------|----------------------|
| KiCad plugin | New OracleProtocol impl for KiCad netlists |
| FreeCAD | New domain="freecad" RuntimeTask |
| LoRA experts | SurrogateRuntime selects expert by topology_family |
| FAISS memory | MemoryEntry schema is FAISS-ready (just add vectors) |
| Distributed exec | RuntimeExecutor is stateless — farm out tasks |
| Continual learning | MemoryRuntime provides replay data |

## Files Created

- `backend/core_runtime/` — 8 modules + __init__
- `scripts/run_runtime_benchmark.py` — canonical benchmark runner
- `tests/test_v212_runtime.py` — 43 tests
- `docs/V212_EXECUTION_RUNTIME.md` — this file

## Tests

```
43 tests PASSED:
  - RuntimeTask: construction, frozen, fingerprint, determinism
  - MNAOracleAdapter: protocol conformance, deterministic solve
  - SurrogateRuntime: zero baseline, fingerprint, latency
  - ProjectionExecution: construction, frozen, to_projection_result
  - MemoryRuntime: register, load, roundtrip, count, clear
  - ExecutionTrace: construction, frozen, fingerprint, json roundtrip
  - TraceStore: save, load, load_all, clear
  - DatasetManifest: construction, frozen, fingerprint, roundtrip
  - DatasetRegistry: register, find_by_id, find_by_sha256, clear
  - compute_dataset_sha256: deterministic, seed-dependent
  - Determinism: task + trace fingerprint stability
  - Failure taxonomy: category consistency
  - Integration: executor with mock components
```

## No-Regression

- v2.11 tests: 46/46 PASSED
- v2.12 tests: 43/43 PASSED
- Combined: 89/89 PASSED
