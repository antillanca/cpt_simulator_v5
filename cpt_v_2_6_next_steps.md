# CPT v2.6 Next Steps

This file is the execution handoff for the next agent.
It documents the current v2.6 continuation plan step by step so work can resume
without reconstructing context.

## Current State

Completed:

- structured traces
- oracle dataset generation integration
- layer-aware CPT-Bench
- validation runner integration
- invariant-family thresholds
- default pytest stability
- handoff documentation

Still pending:

- larger oracle coverage across real curriculum modules
- more benchmark cases per layer
- replay/regression hardening for future artifacts
- neural evaluator implementations
- Kaggle/GPU work, deferred for now

## Working Rules

- Keep the Lua sandbox as the only source of truth.
- Do not touch `backend/core_truth/modules.json` unless a task explicitly requires it.
- Do not rewrite the orchestrator or remove existing features.
- Keep changes incremental and deterministic.
- Do not start GPU/Kaggle training in this phase.

## Step 1. Reconfirm the baseline

Status: done, but should be re-run by the next agent if anything changes.

Goal:

- verify the repo still passes the default local suite
- confirm the v2.6 tests still pass

What to run:

```bash
python3 -m pytest -q
```

Files to watch:

- `pytest.ini`
- `tests/test_v26_*`
- `tests/test_relaxed_validator.py`
- `tests/test_learning_loop.py`
- `tests/test_concurrency.py`
- `tests/test_teacher_hint.py`
- `scripts/test_student_complex.py`

Exit criteria:

- default suite passes
- only intentional skips remain

## Step 2. Expand oracle generation coverage

Status: partially done, needs expansion.

Goal:

- generate more deterministic samples from real curriculum modules
- keep reasoning and answers sandbox-derived only

Touch points:

- `backend/datasets/oracle_generator.py`
- `scripts/dataset_oracle_generator.py`
- `backend/core_truth/modules.json` only if a new verified module is required

What to add:

- more real module filters by `module_key`
- more `curriculum_layer` sweeps
- batch generation presets for common layer groups
- manifest fields for future fingerprinting

Example command:

```bash
python3 scripts/dataset_oracle_generator.py \
  --output reports/oracle/oracle.jsonl \
  --modules backend/core_truth/modules.json \
  --seed 42 \
  --layer 0 \
  --layer 5 \
  --layer 12
```

Exit criteria:

- generated rows are reproducible
- each row still includes structured trace, invariant metadata, seed, timestamp, and module source

## Step 3. Increase benchmark coverage by layer

Status: partially done, needs more real-module breadth.

Goal:

- make CPT-Bench representative of the curriculum
- keep reports deterministic and replayable

Touch points:

- `backend/benchmarks/cpt_bench/suite.py`
- `scripts/benchmark_runner.py`

Suggested layer families:

- layer 0: logical primitives
- layer 5: arithmetic
- layer 12: kinematics/energy
- layer 20: thermodynamics
- layer 27: electromagnetism
- layer 34: quantum logic

Metrics to preserve:

- `exact_match_rate`
- `invariant_violation_rate`
- `symbolic_consistency`
- `trajectory_deviation`
- `causal_consistency`

Example command:

```bash
python3 scripts/benchmark_runner.py \
  --output reports/benchmarks/cpt_bench.json
```

Exit criteria:

- benchmark report is machine-readable
- each case carries replay data
- validation can consume the benchmark report directly

## Step 4. Keep validation as the gate

Status: done, needs future extension only.

Goal:

- validate datasets and benchmark outputs automatically
- keep family thresholds configurable

Touch points:

- `backend/validation/pipeline.py`
- `backend/validation/thresholds.py`
- `scripts/validation_runner.py`

Threshold env vars:

- `ENERGY_THRESHOLD`
- `MOMENTUM_THRESHOLD`
- `LOGIC_THRESHOLD`
- `QUANTUM_THRESHOLD`
- `DEFAULT_THRESHOLD`
- `NEURAL_APPROX_TOLERANCE`

Example commands:

```bash
python3 scripts/validation_runner.py --mode dataset --input reports/oracle/oracle.jsonl
python3 scripts/validation_runner.py --mode benchmark --input reports/benchmarks/cpt_bench.json
```

Exit criteria:

- validation rejects invariant failures
- family thresholds are respected
- outputs are reproducible

## Step 5. Harden replay and fingerprinting

Status: pending.

Goal:

- make artifacts comparable across runs
- guarantee deterministic replay of traces and benchmark cases

Touch points:

- `backend/traces/schema.py`
- `backend/datasets/oracle_generator.py`
- `backend/benchmarks/cpt_bench/suite.py`
- new regression tests

What to add:

- dataset hash/fingerprint field
- benchmark hash/fingerprint field
- replay assertions for trace continuity
- seed and module-version checks

Exit criteria:

- same seed + same inputs => same output
- trace replay and benchmark replay both remain stable

## Step 6. Add regression tests for the new pipeline

Status: partially done, can be expanded.

Goal:

- cover replay
- cover reproducibility
- cover validation thresholds
- cover benchmark consistency

Current tests:

- `tests/test_v26_trace_schema.py`
- `tests/test_v26_oracle_generator.py`
- `tests/test_v26_oracle_generator_filters.py`
- `tests/test_v26_validation.py`
- `tests/test_v26_validation_thresholds.py`
- `tests/test_v26_bench.py`
- `tests/test_v26_permissions.py`

What to add next:

- dataset fingerprint tests
- benchmark replay tests
- failure-path tests for invalid traces
- module filter tests for more layers

Exit criteria:

- tests cover the main deterministic paths
- no regression causes silent acceptance of bad traces

## Step 7. Leave neural interfaces as abstract-only

Status: done for the current phase.

Goal:

- keep neural support minimal
- avoid training large models now

Touch points:

- `backend/neural/models/`
- `backend/neural/trainers/`
- `backend/neural/exporters/`
- `backend/neural/evaluators/`

Exit criteria:

- interfaces exist
- no large training job is introduced

## Step 8. Keep Kaggle and Hermes restricted

Status: done.

Goal:

- preserve tooling support
- keep truth-authority boundaries strict

Touch points:

- `backend/tooling/permissions.py`
- `backend/tooling/hermes.py`
- `scripts/kaggle_trainer.py`

Do not allow:

- bypassing verification
- changing core truth automatically
- approving physics or invariants without human approval

Exit criteria:

- tooling can assist
- tooling cannot override source of truth

## Recommended Order For The Next Agent

1. Expand oracle coverage with more curriculum layers.
2. Add more benchmark cases using real modules.
3. Add fingerprint/replay regression coverage.
4. Keep validation as the gate after every artifact generation step.
5. Stop before Kaggle/GPU or distillation work.

## Minimal Handoff Summary

If another agent starts here, they should know:

- v2.6 infrastructure is already in place
- the repo currently passes default pytest
- the next value is in broader deterministic coverage, not new architecture
- the safest path is to extend the current oracle pipeline, not rewrite it

