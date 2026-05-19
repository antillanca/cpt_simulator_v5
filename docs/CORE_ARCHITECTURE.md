# CORE Architecture

## What CORE Is

CORE = **Cognitive Operational Runtime Engine**

CORE is not a simulator, nor a GNN framework, nor a physics solver.
It is a deterministic hybrid runtime that orchestrates oracle
computation, surrogate inference, constraint projection, memory
retrieval, and adaptive scheduling to execute verifiable cognitive
tasks.

CORE schedules, routes, caches, and traces execution -- it never
performs domain-specific computation itself.

CORE is the canonical runtime. CPT (Circuit Projection Tool) is the
first-domain validation lineage, originating from cpt_simulator_v5.
Circuits were the historical first domain that validated the runtime
architecture. All v2.15 guarantees carry forward unchanged.

## Package Structure

```
core_runtime/
  core/
    runtime/        -- generic task execution
    memory/         -- exact cache + retrieval (FAISS)
    routing/        -- capability router + execution scheduler
    scheduling/     -- projection budget, trajectory analysis, warmstart
    specs/          -- task hashing, dataset registry, frozen specs
    experience/     -- operational experience schemas + datasets
    tracing/        -- execution traces
    domain_sdk/     -- Protocol interfaces for domain integration
    projection/     -- generic projection interface (future)
    surrogate/      -- generic surrogate interface (future)
    oracle/         -- generic oracle interface (future)
  domains/
    circuits/       -- first validated domain (v2.15.0 lineage)
 linear_system/ -- proof-of-concept second domain (v0.2.0)
  benchmarks/
  data/             -- migrated operational artifacts
  tests/
```

## Architecture Diagram

```
                 +------------------+
                 |   Domain Code    |
                 | (circuits,       |
                 |  linear_system,  |
                 |  future domains) |
                 +--------+---------+
                          |
                   Domain SDK Protocols
                  (DomainTask, DomainOracle,
                   DomainSurrogate, DomainProjection,
                   DomainEvaluator, DomainConfidence)
                          |
                 +--------+---------+
                 |   CORE Runtime   |
                 |                  |
                 |  +------------+  |
                 |  |  Routing   |  |  capability_router
                 |  |            |  |  execution_scheduler
                 |  +------+-----+  |  execution_policy
                 |         |        |
                 |  +------v-----+  |
                 |  | Scheduling |  |  projection_scheduler
                 |  |            |  |  trajectory_analysis
                 |  |            |  |  cost_estimator
                 |  |            |  |  warmstart_runtime
                 |  +------+-----+  |
                 |         |        |
                 |  +------v-----+  |
                 |  |   Memory   |  |  exact_cache
                 |  |            |  |  retrieval_memory
                 |  |            |  |  faiss_runtime
                 |  +------+-----+  |  memory_runtime
                 |         |        |
                 |  +------v-----+  |
                 |  | Experience |  |  operational_experience_schema
                 |  |            |  |  experience_dataset_schema
                 |  +------+-----+  |
                 |         |        |
                 |  +------v-----+  |
                 |  |  Tracing   |  |  execution_trace
                 |  +------+-----+  |
                 |         |        |
                 |  +------v-----+  |
                 |  |   Specs    |  |  task_hashing
                 |  |            |  |  dataset_registry
                 |  +------------+  |  frozen spec modules
                 +------------------+
```

## Relationship: Old Code vs New Code

| Old Location | New Location | Status |
|---|---|---|
| backend/runtime/ | core_runtime/core/scheduling/ + routing/ + memory/ | Copied |
| backend/core_runtime/ | core_runtime/core/routing/ + tracing/ + specs/ | Copied |
| backend/core_spec/ | core_runtime/core/specs/ | Copied |
| backend/circuits/ | core_runtime/domains/circuits/ | Copied |
| workspace/operational_experience/ | core_runtime/data/operational_experience/ | Migrated |
| workspace/paper_figures/ | core_runtime/data/paper_figures/ | Migrated |

Old code paths remain functional through compatibility shims
in backend/runtime/_compat.py with deprecation warnings.

## Key Constraint

Projection remains the final authority. CORE schedules and routes,
but never modifies domain-specific physics equations or projection
logic.

## v3.1 Observability Additions

- linear_system domain now exercises FULL SDK pipeline (Oracle, Surrogate, Projection, Evaluator, Confidence, Trace)
- execute_linear_system_pipeline() provides canonical E2E SDK validation
- Operational experience dataset (300 executions) verified with SHA-256 hashes in manifest
- Paper figures (7 PNG + CSV) and runtime reports (4 files) preserved in core_runtime/data/
