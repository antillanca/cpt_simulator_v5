# CPT v2.6 Handoff

This document is the continuation guide for the current v2.6 migration state.
It is intended for a new agent taking over the repo without re-reading the
entire conversation.

## Current Goal

The repo is in the "oracle pipeline integration" phase of CPT v2.6.
The infrastructure exists and the local workflow is now connected:

- `modules.json` -> sandbox execution
- structured traces
- oracle JSONL generation
- benchmark execution by curriculum layer
- automatic validation reports
- invariant-family thresholds

## What Is Already Implemented

- Structured trace schema and replay:
  - `backend/traces/schema.py`
- Oracle dataset generator:
  - `backend/datasets/oracle_generator.py`
  - `scripts/dataset_oracle_generator.py`
- Validation pipeline and thresholds:
  - `backend/validation/pipeline.py`
  - `backend/validation/thresholds.py`
  - `scripts/validation_runner.py`
- CPT-Bench layer-aware suite:
  - `backend/benchmarks/cpt_bench/suite.py`
  - `scripts/benchmark_runner.py`
- Permission isolation:
  - `backend/tooling/permissions.py`
- v2.6 docs:
  - `docs/ARCHITECTURE_V26.md`
  - `docs/TRACE_SCHEMA.md`
  - `docs/CPT_BENCH.md`
  - this file
- Tests:
  - `tests/test_v26_*`
  - legacy integration probes are now opt-in or skipped in default pytest runs

## Default Test State

The repository now passes the default local suite with:

```bash
python3 -m pytest -q
```

Current behavior:

- Default suite passes
- Integration probes are skipped unless explicit opt-in is enabled
- `RUN_INTEGRATION_TESTS=1` re-enables live-server style probes

## Canonical Commands

Generate an oracle dataset:

```bash
python3 scripts/dataset_oracle_generator.py \
  --output reports/oracle/oracle.jsonl \
  --modules backend/core_truth/modules.json \
  --seed 42 \
  --layer 12
```

Run a benchmark report:

```bash
python3 scripts/benchmark_runner.py \
  --output reports/benchmarks/cpt_bench.json
```

Run validation against a dataset or benchmark:

```bash
python3 scripts/validation_runner.py --mode dataset --input reports/oracle/oracle.jsonl
python3 scripts/validation_runner.py --mode benchmark --input reports/benchmarks/cpt_bench.json
```

Run the full local oracle pipeline in one step:

```bash
python3 scripts/oracle_pipeline_runner.py \
  --dataset-output reports/oracle/oracle.jsonl \
  --benchmark-output reports/benchmarks/cpt_bench.json
```

## Invariant Thresholds

Validation supports family-specific thresholds via env vars:

- `ENERGY_THRESHOLD`
- `MOMENTUM_THRESHOLD`
- `LOGIC_THRESHOLD`
- `QUANTUM_THRESHOLD`
- `DEFAULT_THRESHOLD`
- `NEURAL_APPROX_TOLERANCE`

The validation pipeline resolves them through `backend/validation/thresholds.py`.

## Data Contract

Oracle rows now include:

- `question`
- `structured_state`
- `reasoning_trace`
- `equations_used`
- `invariants_checked`
- `final_answer`
- `verification_status`
- `module_source`
- `curriculum_layer`
- `seed`
- `timestamp`

Reasoning and answers must continue to originate from deterministic sandbox
execution only. LLMs may verbalize questions or assist with tooling, but they
must not fabricate traces or answers.

## Benchmark Contract

CPT-Bench is now layer-aware and records:

- exact match rate
- invariant violation rate
- symbolic consistency
- trajectory deviation
- causal consistency

Each benchmark case also carries replay data so the report is machine-readable
and reproducible.

## Safe Next Steps

1. Expand oracle generation coverage across more real curriculum modules.
2. Add more benchmark cases per layer using actual module examples.
3. Harden replay helpers and add regression tests around trace continuity.
4. Extend validation gating for future neural approximation experiments.
5. Only then consider Kaggle/GPU workflows for large corpus generation.

## Do Not Change Without Explicit Need

- `backend/core_truth/modules.json`
- `scripts/training_orchestrator.py`
- the Lua sandbox
- Kaggle integration entrypoints
- Hermes permission boundaries

## Notes For A New Agent

If you are taking over:

- Read this file first.
- Do not clean unrelated modified files unless the task explicitly requires it.
- Keep changes incremental and deterministic.
- Prefer extending the current pipeline over refactoring it.
