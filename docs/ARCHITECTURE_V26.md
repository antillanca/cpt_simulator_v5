# CPT v2.6 Architecture

## Goal

CPT v2.6 is an oracle-readiness layer. The sandbox remains the only source of truth, and every generated dataset, trace, benchmark result, and validation decision must be replayable from deterministic execution.

## Layers

1. Core truth
1. Structured traces
1. Oracle datasets
1. Validation pipeline
1. Benchmarks
1. Neural experiment interfaces
1. Tooling permissions

## Oracle Pipeline

1. Select a curriculum module or deterministic rule.
1. Build a reproducible initial state and parameter sweep.
1. Execute the Lua sandbox.
1. Canonicalize the trace.
1. Verify invariants.
1. Serialize JSONL rows and manifest metadata.

## Validation Flow

1. Sandbox execution produces the reference trajectory.
1. Model predictions are compared against the sandbox.
1. Symbolic consistency, exact match rate, prediction error, and trajectory deviation are computed.
1. Any invariant violation above threshold rejects the candidate.

## Benchmark Flow

1. Run local benchmark cases.
1. Replay the sandbox for each category.
1. Collect versioned metrics.
1. Write a deterministic report.

## Neural Infrastructure

The neural layer is intentionally minimal. It only exposes abstract interfaces for tiny transformers, GNNs, PINNs, trainers, exporters, and evaluators. Training large models is out of scope for v2.6.

## Permission Boundaries

LLMs may generate docs, wrappers, tests, log analysis, and dataset verbalizations.
LLMs may not modify core truth, invariants, verifiers, or bypass verification.
Hermes may propose patches but cannot merge or override protected paths without explicit human approval.

## Handoff Guidance

For continuation, see `docs/HANDOFF_V26.md`. That file captures the current
workflow, default commands, thresholds, and next steps for another agent.
