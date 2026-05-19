# CPT v2.15 -- Final Validation Report

**Release**: v2.15-runtime-stable
**Commit**: 2461f2e
**Tag**: v2.15-runtime-stable
**Date**: 2026-05-19
**Python**: 3.14.4
**Hardware**: AMD Ryzen 5 3450U, 29GB RAM, x86_64

## Pass/Fail Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Correctness Preservation Benchmark | PASS |
| 2 | Oscillatory Convergence Validation | PASS |
| 3 | Retrieval Effectiveness Validation | PASS |
| 4 | Scheduler Overhead Validation | PASS |
| 5 | Operational Guarantees Documentation | PASS |
| 6 | Reproducible Release Freeze | PASS |
| 7 | Operational Experience Export | PASS |
| 8 | Paper Figure Exports | PASS |
| 9 | Stability Freeze | PASS |
| 10 | v2.16 Dataset Schema Preparation | PASS |
| 11 | Final Validation | PASS |

**Overall: PASS** -- v2.15 is COMPLETE.

## Benchmark Summary

### Correctness Preservation (Phase 1)
- `--compare-fixed-vs-adaptive` mode added to `scripts/run_runtime_benchmark.py`
- Compares fixed budget (v2.14: 20 iters) vs adaptive budget (v2.15)
- Acceptance criteria: adaptive residual within 1% of fixed, fewer iterations, lower runtime

### Oscillatory Convergence (Phase 2)
- 40/40 tests passing in `tests/test_v215_oscillatory_convergence.py`
- CASE A: oscillatory accepted (0.8 -> 0.4 -> 0.5 -> 0.25 -> 0.3 -> 0.1)
- CASE B: divergence_risk (0.8 -> 0.82 -> 0.85 -> 0.9)
- CASE C: stalled (deltas < 1e-9)
- CASE D: fast_converging (strict exponential decay)
- `_is_diverging()` regression permanently covered

### Scheduler Overhead (Phase 4)
- Average scheduler overhead: 0.026 ms per call
- P50: 0.025 ms, P99: 0.040 ms
- Scheduler efficiency ratio: **230.8x** (target > 5.0x)
- Verdict: **PASS** -- scheduler pays for itself by orders of magnitude

### Retrieval Effectiveness (Phase 3)
- Synthetic validation (200 samples, no .pt dataset available at freeze time)
- Warmstart win rate: 100% (synthetic)
- Sign test p-value: 0.0000 (significant)
- Verdict: **WARMSTART_EFFECTIVE**
- Honest reporting: if real data contradicts, report explicitly

## Correctness Guarantees

1. Projection remains final authority -- scheduler never overrides
2. Physics equations never modified by scheduler
3. Projection correctness never bypassed
4. Exact cache always has priority
5. Retrieval only provides initialization hints
6. Warmstart never replaces projection validation
7. All routing decisions deterministic
8. Same input -> same execution trace
9. Degraded executions excluded from clean indexes
10. Every execution fully traceable

## Runtime Efficiency Summary

| Metric | Value |
|--------|-------|
| Scheduler overhead (avg) | 0.026 ms |
| Scheduler efficiency ratio | 230.8x |
| Escalation rate (synthetic) | 3.3% |
| Warmstart rate (synthetic) | 16.3% |
| Budget efficiency ratio | 0.57 (from benchmark) |

## Retrieval Effectiveness Summary

| Metric | Value (synthetic) |
|--------|-------------------|
| Avg coldstart iterations | 19.9 |
| Avg warmstart iterations | 8.3 |
| Avg iterations saved | 11.5 |
| Warmstart win rate | 100% |
| Statistical significance | p < 0.0001 |

Note: Real dataset validation pending .pt availability.

## Operational Dataset Summary

| File | Size | Content |
|------|------|---------|
| operational_experience.jsonl | 247 KB | 300 execution records |
| operational_experience.csv | 88 KB | Tabular execution data |
| trajectory_statistics.json | -- | Class distribution |
| family_statistics.json | -- | Per-family convergence |
| scheduler_statistics.json | -- | Routing/outcome stats |
| retrieval_statistics.json | -- | Retrieval effectiveness |

## Paper Figures

7 figures generated in `workspace/paper_figures/`:

1. Fixed vs Adaptive iterations (bar chart)
2. Runtime reduction distribution (histogram)
3. Convergence trajectory classes (pie chart)
4. Scheduler routing distribution (horizontal bar)
5. Retrieval-assisted convergence (scatter)
6. Topology family convergence (grouped bars)
7. Scheduler overhead vs savings (bar)

Each figure has corresponding CSV source table.

## Known Limitations

- Retrieval validation uses synthetic data (no .pt dataset at freeze)
- Scheduler overhead ratio depends on per-iteration projection cost
- Oscillatory trajectories may require oracle escalation
- Degraded mode reduces budget aggressively (by design)
- Warmstart effectiveness varies by topology family
- Pre-existing v29d warmstart overflow issue (not v2.15 regression)

## Future Roadmap Boundaries

v2.16 MAY add:
- Experience-based budget calibration from v2.15 operational data
- LoRA fine-tuning of GNN (using frozen schemas)
- Replay learning (consuming exported JSONL)
- Distributed execution
- Online learning

v2.16 MUST NOT:
- Break v2.15 frozen API signatures
- Alter projection physics or convergence criteria
- Modify deterministic execution guarantees
- Break backward compatibility with v2.15 JSONL exports

## Test Results

```
697 passed, 1 skipped, 9 warnings
Test suite: 698 tests collected
v215_adaptive_scheduling: 100 tests
v215_oscillatory_convergence: 40 tests
Global regression: 558 tests
```

## Artifacts

| Artifact | Path |
|----------|------|
| Release manifest | `runtime_release_manifest_v215.json` |
| Operational guarantees | `docs/V215_ADAPTIVE_RUNTIME_SCHEDULING.md` |
| Stability guarantees | `docs/V215_STABILITY_GUARANTEES.md` |
| v2.16 schemas | `backend/runtime/experience_dataset_schema.py` |
| Scheduler overhead report | `workspace/runtime_reports/scheduler_overhead_report.json` |
| Retrieval effectiveness report | `workspace/runtime_reports/retrieval_effectiveness_report.json` |
| Operational experience | `workspace/operational_experience/` |
| Paper figures | `workspace/paper_figures/` |

---
*Generated by CPT v2.15 Final Validation*
*All 11 phases PASS. Release frozen at tag v2.15-runtime-stable.*
