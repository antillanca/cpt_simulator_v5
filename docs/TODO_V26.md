# CPT v2.6 TODO

## Foundation

- [x] Structured reasoning trace schema
- [x] Oracle dataset generator connected to real modules
- [x] Validation rejection thresholds per module family
- [x] Benchmark suite expanded by curriculum layer
- [x] Validation runner integrated into local workflows
- [x] Replay CLI utilities
- [x] Permission boundary tightening
- [ ] Oracle dataset generator hardening at scale
- [ ] Neural evaluator implementations
- [ ] Additional trace regression tests for future modules

## Compatibility

- [x] Preserve modules.json
- [x] Preserve Kaggle integration entrypoints
- [x] Preserve Hermes tooling
- [x] Preserve Lua sandbox
- [x] Preserve orchestrator compatibility

## Current Continuation Path

The next agent should focus on:

1. Increasing real-module coverage in oracle generation.
2. Adding more benchmark cases for layers 0, 5, 12, 20, 27, and 34.
3. Extending validation regression tests once the next functional slice lands.
4. Deferring Kaggle/GPU training until the deterministic corpus is larger.

