# CORE Migration Log

Every migration decision is recorded here. This file is mandatory
and must be updated with every structural change.

## 2026-05-19 -- Foundation Transition (v2.15 -> CORE v3.0.0)

### Audit Results

**Domain-agnostic modules (no circuit deps)** -- move to core/:
- backend/runtime/cost_estimator.py
- backend/runtime/embedding_runtime.py
- backend/runtime/execution_scheduler.py
- backend/runtime/experience_dataset_schema.py
- backend/runtime/faiss_runtime.py
- backend/runtime/operational_experience_schema.py
- backend/runtime/projection_experience.py
- backend/runtime/projection_scheduler.py
- backend/runtime/trajectory_analysis.py
- backend/runtime/warmstart_runtime.py
- backend/core_runtime/capability_router.py
- backend/core_runtime/exact_cache.py
- backend/core_runtime/execution_policy.py
- backend/core_runtime/execution_trace.py
- backend/core_runtime/memory_runtime.py
- backend/core_runtime/task_hashing.py

**Circuit-dependent modules** -- move to domains/circuits/:
- backend/core_runtime/oracle_protocol.py (MNA adapter)
- backend/core_runtime/projection_runtime.py (KCL/KVL)
- backend/core_runtime/surrogate_runtime.py (CircuitGraph)
- backend/core_runtime/confidence_runtime.py (KCL residual)
- backend/circuits/* (all 21 modules)

**Modules requiring refactoring** (circuit refs to be abstracted):
- backend/core_runtime/task_runtime.py (imports CanonicalCircuitGraph)
- backend/runtime/retrieval_memory.py (topology_families stat)
- backend/core_runtime/dataset_registry.py (topology_families field)

### Naming Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Core package name | core_runtime | Matches existing convention, clear intent |
| Domain SDK location | core_runtime/core/domain_sdk/ | Central, discoverable |
| Circuit domain package | core_runtime/domains/circuits/ | First validated domain |
| Linear system domain | core_runtime/domains/linear_system/ | Proof-of-concept domain |
| Compatibility shims | backend/ -> core_runtime re-exports | Zero test breakage |

### Versioning Decisions

| Component | Version | Rationale |
|-----------|---------|-----------|
| CORE runtime | v3.0.0 | Major = architectural transition |
| domains/circuits | v2.15.0 | Carries frozen v2.15 lineage |
| domains/linear_system | v0.1.0 | New proof-of-concept |

### Compatibility Adapters

| Old Import | New Location | Adapter Type |
|------------|-------------|--------------|
| backend.runtime.projection_scheduler | core_runtime.core.runtime.scheduling | Re-export shim |
| backend.runtime.trajectory_analysis | core_runtime.core.runtime.scheduling | Re-export shim |
| backend.core_runtime.oracle_protocol | core_runtime.domains.circuits | Re-export shim |
| backend.core_runtime.task_runtime | core_runtime.core.runtime.task_runtime | Refactored + shim |
| backend.circuits.* | core_runtime.domains.circuits.* | Re-export shim |

### File Moves (Phase 5+)

**Core modules copied to core_runtime/core/:**

| Source | Destination | Notes |
|--------|-------------|-------|
| backend/runtime/projection_scheduler.py | core_runtime/core/scheduling/ | Copied verbatim |
| backend/runtime/trajectory_analysis.py | core_runtime/core/scheduling/ | Copied verbatim |
| backend/runtime/cost_estimator.py | core_runtime/core/scheduling/ | Copied verbatim |
| backend/runtime/warmstart_runtime.py | core_runtime/core/scheduling/ | Copied verbatim |
| backend/runtime/execution_scheduler.py | core_runtime/core/routing/ | Copied verbatim |
| backend/core_runtime/capability_router.py | core_runtime/core/routing/ | Copied verbatim |
| backend/core_runtime/execution_policy.py | core_runtime/core/routing/ | Copied verbatim |
| backend/core_runtime/exact_cache.py | core_runtime/core/memory/ | Copied verbatim |
| backend/runtime/retrieval_memory.py | core_runtime/core/memory/ | Copied verbatim |
| backend/runtime/faiss_runtime.py | core_runtime/core/memory/ | Copied verbatim |
| backend/runtime/embedding_runtime.py | core_runtime/core/memory/ | Copied verbatim |
| backend/core_runtime/memory_runtime.py | core_runtime/core/memory/ | Copied verbatim |
| backend/runtime/operational_experience_schema.py | core_runtime/core/experience/ | Copied verbatim |
| backend/runtime/experience_dataset_schema.py | core_runtime/core/experience/ | Copied verbatim |
| backend/runtime/projection_experience.py | core_runtime/core/experience/ | Copied verbatim |
| backend/core_runtime/execution_trace.py | core_runtime/core/tracing/ | Copied verbatim |
| backend/core_runtime/task_hashing.py | core_runtime/core/specs/ | Copied verbatim |
| backend/core_runtime/dataset_registry.py | core_runtime/core/specs/ | Copied verbatim |
| backend/core_spec/experiment_spec.py | core_runtime/core/specs/ | Copied verbatim |
| backend/core_spec/failure_taxonomy.py | core_runtime/core/specs/ | Copied verbatim |
| backend/core_spec/memory_spec.py | core_runtime/core/specs/ | Copied verbatim |
| backend/core_spec/model_spec.py | core_runtime/core/specs/ | Copied verbatim |
| backend/core_spec/projection_spec.py | core_runtime/core/specs/ | Copied verbatim |
| backend/core_spec/report_spec.py | core_runtime/core/specs/ | Copied verbatim |

**Circuit domain modules copied to core_runtime/domains/circuits/:**

| Source | Destination | Notes |
|--------|-------------|-------|
| backend/core_runtime/oracle_protocol.py | circuits/oracle_adapter.py | Renamed |
| backend/core_runtime/projection_runtime.py | circuits/projection_runtime.py | Copied |
| backend/core_runtime/surrogate_runtime.py | circuits/surrogate_adapter.py | Renamed |
| backend/core_runtime/confidence_runtime.py | circuits/confidence_adapter.py | Renamed |

**New files created:**

| File | Purpose |
|------|---------|
| core_runtime/__init__.py | Package root with version |
| core_runtime/core/__init__.py | Domain SDK re-exports |
| core_runtime/core/domain_sdk/__init__.py | Protocol interfaces + registry |
| core_runtime/core/runtime/task_runtime.py | Domain-agnostic RuntimeTask |
| core_runtime/core/runtime/domain_adapters.py | Circuit -> DomainTaskBase adapters |
| core_runtime/domains/linear_system/__init__.py | Linear system domain implementation |
| core_runtime/domains/linear_system/tests.py | Linear system domain tests |
| core_runtime/tests/test_core_domain_sdk.py | Domain SDK validation |
| core_runtime/tests/test_linear_system_domain.py | Linear system E2E |
| core_runtime/tests/test_core_migration.py | Migration + layout + guarantees |
| scripts/audit_domain_dependencies.py | Dependency audit tool |
| scripts/migrate_runtime_artifacts.py | Artifact migration tool |
| pyproject.toml | Packaging configuration |
| docs/CORE_ARCHITECTURE.md | Architecture documentation |
| docs/CORE_PRINCIPLES.md | 10 operational guarantees |
| docs/CORE_DOMAIN_MODEL.md | Domain SDK model |
| docs/CORE_RUNTIME_GUARANTEES.md | Deterministic guarantees |
| docs/PAPER_POSITIONING.md | Academic contribution positioning |
| docs/VERSIONING_POLICY.md | Semantic versioning rules |
| docs/CORE_RELEASE_README.md | Release README |
| CORE_v3_FOUNDATION_MANIFEST.json | Release manifest |

### Artifact Migration

Files migrated: 26

| Source | Destination | Hash Preserved |
|--------|-------------|----------------|
| workspace/operational_experience/family_statistics.json | core_runtime/data/operational_experience/family_statistics.json | YES |
| workspace/operational_experience/operational_experience.csv | core_runtime/data/operational_experience/operational_experience.csv | YES |
| workspace/operational_experience/operational_experience.jsonl | core_runtime/data/operational_experience/operational_experience.jsonl | YES |
| workspace/operational_experience/retrieval_statistics.json | core_runtime/data/operational_experience/retrieval_statistics.json | YES |
| workspace/operational_experience/scheduler_statistics.json | core_runtime/data/operational_experience/scheduler_statistics.json | YES |
| workspace/operational_experience/trajectory_statistics.json | core_runtime/data/operational_experience/trajectory_statistics.json | YES |
| workspace/runtime_reports/domain_dependency_audit.json | core_runtime/data/runtime_reports/domain_dependency_audit.json | YES |
| workspace/runtime_reports/retrieval_effectiveness_report.json | core_runtime/data/runtime_reports/retrieval_effectiveness_report.json | YES |
| workspace/runtime_reports/scheduler_overhead_report.json | core_runtime/data/runtime_reports/scheduler_overhead_report.json | YES |
| workspace/runtime_reports/v215_final_validation_report.md | core_runtime/data/runtime_reports/v215_final_validation_report.md | YES |
| workspace/paper_figures/fig1_fixed_vs_adaptive_iterations.csv | core_runtime/data/paper_figures/fig1_fixed_vs_adaptive_iterations.csv | YES |
| workspace/paper_figures/fig1_fixed_vs_adaptive_iterations.png | core_runtime/data/paper_figures/fig1_fixed_vs_adaptive_iterations.png | YES |
| workspace/paper_figures/fig2_runtime_reduction_distribution.csv | core_runtime/data/paper_figures/fig2_runtime_reduction_distribution.csv | YES |
| workspace/paper_figures/fig2_runtime_reduction_distribution.png | core_runtime/data/paper_figures/fig2_runtime_reduction_distribution.png | YES |
| workspace/paper_figures/fig3_convergence_trajectory_classes.csv | core_runtime/data/paper_figures/fig3_convergence_trajectory_classes.csv | YES |
| workspace/paper_figures/fig3_convergence_trajectory_classes.png | core_runtime/data/paper_figures/fig3_convergence_trajectory_classes.png | YES |
| workspace/paper_figures/fig4_scheduler_routing_distribution.csv | core_runtime/data/paper_figures/fig4_scheduler_routing_distribution.csv | YES |
| workspace/paper_figures/fig4_scheduler_routing_distribution.png | core_runtime/data/paper_figures/fig4_scheduler_routing_distribution.png | YES |
| workspace/paper_figures/fig5_retrieval_assisted_convergence.csv | core_runtime/data/paper_figures/fig5_retrieval_assisted_convergence.csv | YES |
| workspace/paper_figures/fig5_retrieval_assisted_convergence.png | core_runtime/data/paper_figures/fig5_retrieval_assisted_convergence.png | YES |
| workspace/paper_figures/fig6_topology_family_convergence.csv | core_runtime/data/paper_figures/fig6_topology_family_convergence.csv | YES |
| workspace/paper_figures/fig6_topology_family_convergence.png | core_runtime/data/paper_figures/fig6_topology_family_convergence.png | YES |
| workspace/paper_figures/fig7_scheduler_overhead_vs_savings.csv | core_runtime/data/paper_figures/fig7_scheduler_overhead_vs_savings.csv | YES |
| workspace/paper_figures/fig7_scheduler_overhead_vs_savings.png | core_runtime/data/paper_figures/fig7_scheduler_overhead_vs_savings.png | YES |
|| workspace/paper_figures/figures_summary.csv | core_runtime/data/paper_figures/figures_summary.csv | YES |

## 2026-05-19 -- v3.1 Observability & SDK Validation

### linear_system domain upgraded to v0.2.0

| Change | Description |
|--------|-------------|
| LinearSystemConfidence | Added exponential decay confidence scoring |
| LinearSystemTrace | Added deterministic execution trace builder |
| execute_linear_system_pipeline() | Full E2E pipeline: surrogate -> projection -> evaluator -> confidence -> trace |
| DomainConfidence registered | linear_system now registers confidence in domain registry |
| Version bump | 0.1.0 -> 0.2.0 |

### Manifest updated to v1.1

- Added sdk_validation section with per-protocol status
- Added operational_experience SHA-256 hashes
- Added paper_figures and runtime_reports references
- Added v3_1_additions section
- Added frozen_guarantees list (14 guarantees)

### Operational experience verification

- 300 execution dataset confirmed intact in core_runtime/data/
- SHA-256 hashes computed and recorded in manifest
- Workspace copies verified identical
