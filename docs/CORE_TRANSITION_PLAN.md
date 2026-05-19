# CORE Transition Plan

## Why CORE Exists

The CPT Simulator was built as a monolithic circuit-physics runtime.
As the platform matured through v2.15, it became clear that the
scheduling, tracing, caching, and retrieval infrastructure is
domain-agnostic. The projection loop, trajectory analysis, and
scheduler have no inherent circuit dependency.

CORE (Cognitive Operational Runtime Engine) extracts this
domain-agnostic core so that future domains (KiCad, FreeCAD,
mathematics, logic, programming) can plug into the same validated
runtime without forking or modifying the engine.

## What Stays Frozen in the Old Runtime

The v2.15-runtime-stable tag is the historical source of truth.
These frozen artifacts are never modified:

- Projection mathematics
- GNN architecture
- Exact cache semantics
- Retrieval semantics
- Deterministic hashing
- Benchmark semantics
- Failure taxonomy semantics
- Operational experience data (JSONL/CSV)
- Paper figures
- Runtime release manifest

## What Moves Into CORE

All domain-agnostic runtime infrastructure moves to core_runtime/core/:

| Old Location | New Location | Notes |
|---|---|---|
| backend/runtime/projection_scheduler.py | core/runtime/scheduling/ | Budget + stop decisions |
| backend/runtime/trajectory_analysis.py | core/runtime/scheduling/ | Convergence classification |
| backend/runtime/execution_scheduler.py | core/runtime/routing/ | Capability routing |
| backend/runtime/cost_estimator.py | core/runtime/scheduling/ | Cost prediction |
| backend/runtime/exact_cache.py -> moved to core_runtime/core_runtime/ | core/runtime/memory/ | Cache semantics unchanged |
| backend/runtime/retrieval_memory.py | core/runtime/memory/ | Topology-families ref becomes generic |
| backend/runtime/warmstart_runtime.py | core/runtime/scheduling/ | Warmstart logic |
| backend/runtime/operational_experience_schema.py | core/runtime/experience/ | Immutable entries |
| backend/runtime/experience_dataset_schema.py | core/runtime/experience/ | Frozen v2.16 schemas |
| backend/runtime/faiss_runtime.py | core/runtime/memory/ | FAISS index ops |
| backend/runtime/embedding_runtime.py | core/runtime/memory/ | GNN embeddings |
| backend/runtime/projection_experience.py | core/runtime/experience/ | Projection traces |
| backend/core_runtime/capability_router.py | core/runtime/routing/ | 7-action routing |
| backend/core_runtime/execution_policy.py | core/runtime/routing/ | Policy decisions |
| backend/core_runtime/execution_trace.py | core/runtime/tracing/ | Execution traces |
| backend/core_runtime/memory_runtime.py | core/runtime/memory/ | Memory orchestrator |
| backend/core_runtime/task_hashing.py | core/runtime/specs/ | Deterministic hashing |
| backend/core_runtime/dataset_registry.py | core/runtime/specs/ | Dataset metadata |

## What Remains Domain-Specific (moves to domains/circuits/)

| Old Location | New Location | Notes |
|---|---|---|
| backend/circuits/* | domains/circuits/ | All circuit modules |
| backend/core_runtime/oracle_protocol.py | domains/circuits/ | MNA oracle adapter |
| backend/core_runtime/projection_runtime.py | domains/circuits/ | KCL/KVL projection |
| backend/core_runtime/surrogate_runtime.py | domains/circuits/ | CircuitGraph surrogate |
| backend/core_runtime/confidence_runtime.py | domains/circuits/ | KCL-residual confidence |
| backend/core_runtime/task_runtime.py | core/runtime/ (refactored) | Becomes domain-agnostic |
| backend/core_spec/graph_spec.py | domains/circuits/ | CanonicalCircuitGraph |
| backend/core_spec/failure_taxonomy.py | core/runtime/specs/ | Generic if no circuit deps |
| backend/core_spec/projection_spec.py | core/runtime/specs/ | Generic projection specs |
| backend/core_spec/experiment_spec.py | core/runtime/specs/ | Generic experiment specs |
| backend/core_spec/memory_spec.py | core/runtime/specs/ | Generic memory specs |
| backend/core_spec/model_spec.py | core/runtime/specs/ | Generic model specs |
| backend/core_spec/report_spec.py | core/runtime/specs/ | Generic report specs |

## Migration Principles

1. **No scientific regression**: All v2.15 test outcomes preserved
2. **Adapters over rewrites**: Thin wrappers, not new implementations
3. **Import compatibility**: Old paths re-export from new locations
4. **Deterministic hashing unchanged**: Same inputs produce same hashes
5. **Domain SDK first**: Core only knows DomainTask, not CircuitTask
6. **Operational data carries over**: All JSONL/CSV migrated verbatim

## Compatibility Guarantees

- `from backend.runtime.X` still works (compatibility shims)
- `from backend.core_runtime.X` still works (compatibility shims)
- All v2.15 tests continue to pass without modification
- Deprecation warnings added but no breaks until v3.1+
- Deterministic hash outputs identical for same inputs

## Packaging Strategy

- Package name: `core-runtime-engine`
- Core has zero circuit dependencies
- `pip install core-runtime-engine` = domain-agnostic runtime only
- `pip install core-runtime-engine[circuits]` = adds circuit domain
- `pip install core-runtime-engine[linear-system]` = adds linear_system domain
- Domains version independently

## Versioning Strategy

- CORE: v3.0.0 (foundation transition from v2.15)
- domains/circuits: v2.15.0 (carries forward frozen lineage)
- domains/linear_system: v0.1.0 (new proof-of-concept domain)
- Future domains version independently from CORE
- CORE breaking changes = major version bump
- Domain breaking changes = domain's own major version
