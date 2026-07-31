# CORE v3.2 — Operational Hardening

## Overview

CORE v3.2 is an **operational hardening** release. It does NOT enable
learning, distributed execution, or any new runtime capabilities. It makes
CORE more reliable, more reproducible, easier to use, more observable, and
ready for future learning systems.

## What Changed

### 1. Principle Regression Tests (Phase 1)

10 explicit tests for frozen architectural principles:
- Deterministic input -> deterministic trace
- Exact cache always first
- Projection remains final authority
- Warmstart never bypasses projection
- Degraded runs never enter clean retrieval cache
- Same task hash -> same execution route
- Scheduler does not change physics
- Domain logic remains isolated from runtime orchestration
- Knowledge/specs remain frozen
- Same input + same seed -> same outputs

File: `tests/test_core_principles.py` (37 tests)

### 2. CI Smoke Benchmark (Phase 2)

Fast benchmark suitable for pull requests:
- Runs on a tiny representative subset
- Completes in seconds
- Compares fixed vs adaptive behavior
- Verifies correctness and determinism
- Reports cache hit rate, retrieval hit rate, degraded rate

Files: `scripts/run_smoke_benchmark.py`, `scripts/run_runtime_benchmark.py --smoke`

### 3. Deterministic Fuzzing (Phase 3)

Generates synthetic tasks deterministically and runs them twice:
- Fixed seed per task
- Compares trace hash equality
- Compares result equality
- Compares routing equality
- Detects nondeterministic traces, unstable routing, unstable projection

Files: `scripts/fuzz_runtime_deterministic.py`, `tests/test_core_deterministic_fuzzing.py`

### 4. Tutorial Example (Phase 4)

End-to-end walkthrough for new developers:
- Define a DomainTask
- Implement an OracleProtocol
- Run the runtime
- Inspect the trace
- Read the evaluation report
- See memory registration

No GPU required, no circuit dependencies, fully deterministic.

Files: `examples/01_linear_system_walkthrough.py`, `docs/TUTORIAL_LINEAR_SYSTEM.md`

### 5. Official Docker Packaging (Phase 5)

Minimal, deterministic Docker image:
- Supports smoke benchmark, tutorial, and test suite
- Optional extras: circuits, linear-system

Files: `Dockerfile`, `.dockerignore`

### 6. PyPI Pre-Release Readiness (Phase 6)

Packaging supports installable extras:
```bash
pip install core-runtime-engine
pip install core-runtime-engine[circuits]
pip install core-runtime-engine[linear-system]
pip install core-runtime-engine[dev]
pip install core-runtime-engine[docs]
```

No mandatory circuit dependencies for base install.

Files: `pyproject.toml`, `docs/PACKAGING_AND_INSTALL.md`

### 7. Operational Experience Expansion (Phase 7)

Script to grow operational experience dataset:
- SHA-256 anchored entries
- Deterministic manifests
- Aggregate statistics (convergence, routing, runtime distributions)

Files: `scripts/expand_operational_experience.py`, `data/`

### 8. Local Dashboard / HTML Report (Phase 8)

Static local HTML report from operational experience:
- No web server, no backend, no JS framework
- Inline SVG charts (zero external dependencies)
- Shows projection iterations, routing, residuals, failure types

Files: `scripts/generate_operational_dashboard.py`, `workspace/operational_dashboard/`

### 9. Third Domain: Propositional Logic (Phase 9)

A small symbolic domain proving CORE is not limited to numerical domains:
- Oracle: brute-force truth table
- Surrogate: unit-clause shortcut
- Projection: iterative simplification (unit propagation + pure literal elimination)
- Evaluator: oracle vs projected comparison
- Full pipeline with trace

Files: `core_runtime/domains/propositional_logic/`, `tests/test_propositional_logic_domain.py`

### 10. Feature Flags (Phase 10)

Frozen feature flags for future capabilities:
- `enable_lora_experts` (default: False)
- `enable_replay` (default: False)
- `enable_continual_training` (default: False)
- `enable_distributed_execution` (default: False)
- `enable_training_sandbox` (default: False)

All flags are frozen, default to disabled, and are safe to ignore.

File: `core_runtime/core/runtime_config.py`, `tests/test_core_feature_flags.py`

### 11. Training Sandbox Boundary (Phase 11)

Boundary for future training experiments:
- Operates on COPIES of surrogates, never originals
- Produces CANDIDATE artifacts, never production
- Requires EXPLICIT promotion via immutable ledger
- Production runtime never reads sandbox artifacts

File: `core_runtime/core/training_sandbox.py`, `tests/test_training_sandbox_boundary.py`

## What Did NOT Change

- Projection equations: UNCHANGED
- Scheduler semantics: UNCHANGED
- Deterministic hashing: UNCHANGED
- Exact cache behavior: UNCHANGED
- No LoRA training added
- No replay learning added
- No continual learning added
- No distributed execution added
- No GUI added
- No plugins added
- No auto-modifying physics

## Test Status

- v3.1 baseline: 867 tests passing
- v3.2 adds: 132 new tests across 9 new test files
- Total: 999+ tests passing
- All frozen guarantees validated
- No regressions

## Success Criteria Verification

| Criterion | Status |
|-----------|--------|
| Principle regression tests pass | YES |
| Smoke benchmark is CI-ready | YES |
| Deterministic fuzzing exists | YES |
| Tutorial is runnable | YES |
| Dockerfile works | YES |
| Packaging supports installable extras | YES |
| Operational experience is analyzable | YES |
| Local dashboard is generated | YES |
| Propositional logic domain runs end-to-end | YES |
| Feature flags are frozen and safe | YES |
| Training sandbox boundary exists | YES |
| All tests pass | YES |
| No frozen guarantees regress | YES |
