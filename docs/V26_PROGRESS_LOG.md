# CPT v2.6 Progress Log

This file records the exact progression of the current continuation work.
It is intentionally chronological so another agent can resume from the last
completed step without guessing.

## Step 1

Status: completed

What I checked:

- current oracle generator implementation
- current layer-aware CPT-Bench implementation
- current trace schema
- real curriculum modules available in `backend/core_truth/modules.json`

Key observation:

- the oracle and benchmark infrastructure already exists and is functional
- the next useful increment is fingerprinting and broader real-module coverage

## Step 2

Status: completed

Next changes:

- deterministic fingerprints were added to oracle datasets and benchmark reports
- dataset manifests now record `dataset_fingerprint`
- benchmark reports now record `fingerprint`

## Step 3

Status: completed

What changed:

- benchmark default cases now prefer real curriculum modules from `backend/core_truth/modules.json`
- the suite now covers layers 0, 12, 13, 14, 15, 26, 27, 28, 29, and 34 when those modules exist
- case-level fingerprints are recorded for benchmark replay and comparison

## Step 4

Status: completed

Next checks:

- updated pytest suite passed
- fingerprint assertions passed
- default benchmark cases load real modules cleanly
- no regressions were introduced by the fingerprinting change

## Step 5

Status: completed

Verification:

- `python3 -m compileall -q backend scripts tests`
- `python3 -m pytest -q tests/test_v26_trace_schema.py tests/test_v26_oracle_generator.py tests/test_v26_oracle_generator_filters.py tests/test_v26_bench.py tests/test_v26_bench_layers.py tests/test_v26_validation.py tests/test_v26_validation_thresholds.py tests/test_v26_permissions.py tests/test_v25_verifiers.py tests/test_v25_dataset_generator.py tests/test_v25_permissions.py`
- `python3 -m pytest -q`

Current end state:

- oracle rows and manifests now carry deterministic fingerprints
- benchmark reports and cases now carry deterministic fingerprints
- benchmark defaults now prefer real modules from the curriculum when available
- validation metrics include threshold profile metadata
- the default pytest suite remains green

## Step 6

Status: completed

What was added:

- `scripts/oracle_pipeline_runner.py`
- a single local command that performs dataset generation, benchmark execution, and validation
- a machine-readable summary that includes dataset and benchmark fingerprints plus validation outputs

Verification:

- `python3 -m pytest -q tests/test_v26_pipeline_runner.py tests/test_v26_trace_schema.py tests/test_v26_oracle_generator.py tests/test_v26_oracle_generator_filters.py tests/test_v26_bench.py tests/test_v26_bench_layers.py tests/test_v26_validation.py tests/test_v26_validation_thresholds.py tests/test_v26_permissions.py tests/test_v25_verifiers.py tests/test_v25_dataset_generator.py tests/test_v25_permissions.py`
- `python3 -m pytest -q`

## Remaining Work for the Next Agent

1. Expand oracle generation volume and module coverage if needed.
2. Add more replay and fingerprint regression tests for future artifacts.
3. Decide whether to surface fingerprints in any upstream reporting UI or CLI.
4. Keep Kaggle/GPU work deferred until the deterministic corpus is larger.
