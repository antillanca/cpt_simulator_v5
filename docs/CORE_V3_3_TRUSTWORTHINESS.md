# CORE v3.3 — Trustworthiness & Operational Explainability

## Status: COMPLETE

## Overview

CORE v3.3 adds a deterministic operational trust layer that tracks
verifiability, stability, uncertainty, degradation, and traceability
WITHOUT altering the physical correctness or the final projection authority
of the engine.

## Key Principle

Trust/audit layers are OBSERVABILITY ONLY. They MUST NOT:
- alter physics
- alter execution results
- alter projection equations
- score human values, morality, or ideology

## New Modules

### 1. Trustworthiness Runtime (`core_runtime/core/trustworthiness_runtime.py`)

Classifies every execution into one of three trust levels:

| Level | Meaning | Flags |
|-------|---------|-------|
| CERTAIN | No trust flags triggered | 0 |
| TRANSITIONAL | Exactly one flag (HIGH_RESIDUAL_HEURISTIC) | 1 |
| INDETERMINATE | Multiple flags or critical failures | 2+ |

Trust flags:
- `high_residual_heuristic` — residual > 0.01
- `unstable_convergence` — divergence_risk or stalled trajectory
- `escalation_triggered` — projection escalation was required
- `low_confidence` — confidence < 0.5
- `ood_execution` — out-of-distribution topology
- `high_projection_effort` — iterations > family budget

All audits are frozen dataclasses with SHA-256 fingerprints.

### 2. Uncertainty Memory (`core_runtime/core/uncertainty_memory.py`)

Stores transitional/indeterminate executions for later analysis.
This is NOT retrieval memory. This is NOT exact cache.
This is memory of indeterminacy — what went wrong and where.

Properties:
- Deterministic ordering (insertion order + task_hash tiebreak)
- Atomic persistence (JSONL with SHA-256 anchors)
- Entries are frozen dataclasses
- Searchable by reason, trust level, topology

### 3. Failure Report (`core_runtime/core/failure_report.py`)

Structured, deterministic failure analysis replacing free-text output.

Properties:
- No LLM-generated text
- Deterministic cause inference from error_type + conditions
- Deterministic action inference
- SHA-256 fingerprint for traceability

### 4. Explainability Runtime (`core_runtime/core/explainability_runtime.py`)

Deterministic post-hoc explanations for runtime results.

Explanation types:
- `exact_cache_hit` — deterministic cache reuse
- `warmstart_assisted` — warmstart reduced initial residual
- `projection_escalated` — escalation required
- `ood_conservative` — OOD topology conservative budget
- `unstable_convergence` — unstable trajectory
- `certain_execution` — standard high-confidence execution
- `transitional_execution` — marginal residual flagged
- `indeterminate_execution` — multiple flags raised

### 5. Audit Trace Store (`core_runtime/core/audit_trace_store.py`)

Persist trustworthiness audits separately from memory and exact cache.

Properties:
- Deterministic ordering
- Atomic JSONL persistence
- No influence on runtime outputs
- Queryable by task_hash
- Statistics: trust_level_distribution, flag_distribution, escalation_rate

### 6. Router Integration (`core_runtime/core/routing/capability_router.py`)

The capability router now accepts an optional `trust_audit` parameter.

Trust-aware routing rules:
- INDETERMINATE audit → increased_budget with force_oracle
- TRANSITIONAL audit → conservative budget (standard_projection)
- CERTAIN audit → normal routing (no change)
- Exact cache hit always wins regardless of trust level

## Benchmark

`scripts/run_trustworthiness_benchmark.py` runs the linear_system domain
with and without the audit layer, then verifies that correctness is
unchanged.

Reports:
- avg_residual, avg_projection_iterations, avg_runtime_ms
- avg_trust_score, avg_uncertainty_score
- certain_rate, transitional_rate, indeterminate_rate
- audit_trigger_rate, uncertainty_memory_hit_rate
- flag_distribution, explanation_distribution
- correctness_unchanged (must be True)

## Tests

108 new v3.3 tests, 304 total core tests passing.

Test files:
- `tests/test_core_trustworthiness_runtime.py` — 20 tests
- `tests/test_core_uncertainty_memory.py` — 16 tests
- `tests/test_core_failure_report.py` — 20 tests
- `tests/test_core_explainability_runtime.py` — 18 tests
- `tests/test_core_audit_trace_store.py` — 18 tests
- `tests/test_core_router_trust_integration.py` — 8 tests
- `tests/test_core_trust_benchmark.py` — 8 tests

## Safety Guarantees Preserved

1. Same input -> same audit (deterministic classification)
2. Audit layer does NOT modify projection results
3. Audit layer does NOT modify exact cache behavior
4. Trust scores are observability signals only — no feedback into physics
5. Exact cache always first regardless of trust level
6. Projection remains final authority
7. Uncertainty memory entries are barred from clean retrieval cache
8. Knowledge/specs remain frozen
9. Domain logic remains isolated from trust orchestration
