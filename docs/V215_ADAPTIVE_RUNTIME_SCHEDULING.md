# CPT v2.15 — Adaptive Runtime Scheduling

## Overview

v2.15 replaces fixed projection budgets with a **deterministic runtime scheduler**
that optimizes operational efficiency (max iterations, early stopping, escalation)
without modifying GNN architecture, projection physics, or determinism guarantees.

The scheduler ONLY decides:
- `max_iterations`: how many projection steps to allow
- `stagnation_patience`: how many steps without improvement before stopping
- `min_improvement`: threshold for what counts as meaningful improvement
- `convergence_target`: residual target for early stopping
- `escalation_threshold`: residual above which to escalate early

Projection remains the final authority. The scheduler only allocates effort and
decides when to stop or escalate.

## Architecture

```
Task → ExecutionScheduler.schedule()
         ├── Route: cache_hit / retrieval_warmstart / retrieval_semantic /
         │          ood_escalated / standard / oracle_forced / degraded
         └── ProjectionBudget (max_iterations, patience, convergence_target, ...)

     → Projection Loop (per iteration)
         └── ProjectionScheduler.should_stop()
               ├── Budget exhausted → STOP
               ├── Converged → STOP (converged)
               ├── Escalation threshold → STOP + escalate
               ├── Divergence → STOP + escalate
               ├── Stagnation → STOP (stagnated)
               ├── Diminishing returns → STOP
               └── Otherwise → CONTINUE

     → TrajectoryAnalyzer.analyze(residuals)
           └── Classification: fast_converging / stable_linear / oscillatory /
                              stalled / divergence_risk / retrieval_assisted

     → ExecutionScheduler.compute_outcome()
           └── Outcome: cache_hit / converged_early / success / budget_exhausted /
                      stagnated / diverged / degraded / escalated

     → OperationalExperienceAccumulator.add(entry)
           └── Persistent experience for future budget calibration
```

## Execution Order (9-step pipeline)

1. **Exact Cache** — bypass if hash match
2. **Retrieval Memory** — semantic similarity search
3. **Confidence** — estimate projection difficulty
4. **Cost** — estimate runtime/resource cost
5. **Scheduler Budget** — allocate deterministic ProjectionBudget
6. **Warmstart** — apply retrieval result as initial projection state
7. **Projection** — iterate with should_stop checks
8. **Escalation** — trigger oracle if divergence/stagnation
9. **Trace Commit** — persist OperationalExperienceEntry

## Modules

| Module | File | Purpose |
|--------|------|---------|
| ProjectionBudget | `backend/runtime/projection_scheduler.py` | Immutable budget dataclass |
| ProjectionScheduler | `backend/runtime/projection_scheduler.py` | Budget allocation + stopping decisions |
| StopDecision | `backend/runtime/projection_scheduler.py` | Immutable stop/continue decision |
| TrajectoryAnalyzer | `backend/runtime/trajectory_analysis.py` | Residual trajectory classification |
| TrajectoryMetrics | `backend/runtime/trajectory_analysis.py` | Numerical trajectory summary |
| ExecutionScheduler | `backend/runtime/execution_scheduler.py` | 7 routes, 8 outcomes, failure tracking |
| OperationalExperienceEntry | `backend/runtime/operational_experience_schema.py` | Single execution record |
| OperationalExperienceAccumulator | `backend/runtime/operational_experience_schema.py` | Aggregate stats, JSONL export |

## Routes (7)

| Route | Condition | Budget |
|-------|-----------|--------|
| `cache_hit` | Exact hash match | None (bypass) |
| `retrieval_warmstart` | similarity >= 0.7 | Reduced (60% of standard) |
| `retrieval_semantic` | 0.3 <= similarity < 0.7 | Moderate |
| `ood_escalated` | OOD + low confidence | Large + forced oracle |
| `standard` | Default path | Full budget |
| `oracle_forced` | 3+ consecutive failures | Oracle-only |
| `degraded` | System degradation flag | Oracle fallback |

## Outcomes (8)

| Outcome | Meaning |
|---------|---------|
| `cache_hit` | Bypassed projection via exact cache |
| `converged_early` | Converged before 50% of budget |
| `success` | Converged using 50%+ of budget |
| `budget_exhausted` | Ran all iterations without convergence |
| `stagnated` | Stopped due to stagnation detection |
| `diverged` | Residuals increased monotonically |
| `degraded` | Degraded execution (oracle fallback) |
| `escalated` | Escalation threshold triggered |

## Trajectory Classification (6 classes)

| Class | Detection |
|-------|-----------|
| `fast_converging` | Early slope > 2x average slope + final residual < 0.01 |
| `stable_linear` | Default: steady linear decrease |
| `oscillatory` | Oscillation rate > 0.3 (local extrema density) |
| `stalled` | Total improvement < 1e-8 |
| `divergence_risk` | Last 3 residuals monotonically increasing |
| `retrieval_assisted` | Warmstart + fast convergence + final < 0.01 |

## Stop Reasons (7)

| Reason | Trigger |
|--------|---------|
| `continue` | No stop condition met |
| `converged` | current_residual <= convergence_target |
| `stagnated` | No improvement in patience window |
| `diminishing_returns` | Late improvement < 10% of early improvement |
| `divergence` | 3+ monotonically increasing residuals |
| `escalate` | current_residual >= escalation_threshold |
| `budget_exhausted` | iteration >= max_iterations |

## Operational Guarantees

These principles are invariant under all v2.15 runtime configurations.
Violating any of these constitutes a critical regression.

1. **Projection remains the final authority.** The scheduler allocates
   effort budget but never overrides projection results. Every converged
   state is produced by the projection loop, not the scheduler.

2. **The scheduler never modifies physics equations.** Kirchhoff
   constraints, MNA formulation, and GNN forward passes are untouched.
   The scheduler only decides when to stop iterating.

3. **The scheduler never bypasses projection correctness.** Early
   stopping, stagnation detection, and escalation all preserve the
   invariant that the final state satisfies projection convergence
   criteria or is explicitly marked as non-converged.

4. **Exact cache always has priority.** If a task hash matches a
   previous successful execution, the cached result is returned
   immediately. No adaptive logic overrides a cache hit.

5. **Retrieval memory only provides initialization hints.** Warmstart
   vectors seed the projection initial state but do not replace
   projection validation. The projection loop still runs and checks
   convergence.

6. **Warmstart never replaces projection validation.** Even when
   retrieval similarity is high, the projection loop executes with at
   least a reduced budget and validates convergence against the same
   residual thresholds.

7. **All routing decisions are deterministic.** Given the same inputs
   (task hash, retrieval similarity, confidence, cost estimate, family
   statistics), the scheduler produces the same budget allocation every
   time. There is zero randomness in the scheduling path.

8. **Same input produces the same execution trace.** Deterministic
   routing plus deterministic projection means identical inputs yield
   identical iteration counts, residual trajectories, and outcomes.

9. **Degraded executions never enter clean retrieval indexes.** When
   the runtime is in degraded mode (system resource constraints), the
   execution is flagged and its results are excluded from the FAISS
   retrieval index and operational experience accumulation.

10. **Every execution is fully traceable.** Each run produces an
    `OperationalExperienceEntry` recording the full decision path:
    route, budget, trajectory, outcome, and all intermediate metrics.

## Determinism Guarantees

- **Zero randomness**: all decisions are deterministic functions of inputs
- **No learning**: no LoRA, replay buffers, or parameter updates in v2.15
- **No hidden state**: scheduler state is explicitly passed, never implicit
- **Same inputs → same budget**: always, regardless of call order
- **Family stats blending**: 60% heuristic + 40% experience (requires >= 3 samples)

## Budget Allocation Policy

```
difficulty_tier = f(confidence, cost_estimate, retrieval_similarity, is_ood, family_stats)

Tiers         Max Iters   Patience
───────────── ──────────  ────────
trivial       3           2
easy          5           3
moderate      10          5
hard          20          7
extreme       50          10

Adjustments:
- retrieval_similarity >= 0.5 → budget * 0.6, patience * 0.7
- is_ood → budget * 1.5
- family experience (>= 3 samples) → 60% heuristic + 40% avg_iterations
- cost_estimate suggests more → upgrade (cap at extreme)
```

## Test Coverage

100 tests in `tests/test_v215_adaptive_scheduling.py`:

| Section | Count | Target |
|---------|-------|--------|
| ProjectionBudget | 10 | Construction, validation, frozen, JSON |
| StopDecision | 6 | Validation, frozen, invalid inputs |
| ProjectionScheduler | 20 | Allocation, stopping, trajectory, escalation |
| TrajectoryAnalyzer | 14 | 6 classes, edge cases, determinism |
| ExecutionScheduler | 16 | 7 routes, 8 outcomes, failure recording |
| OperationalExperienceEntry | 8 | Schema, frozen, roundtrip |
| OperationalExperienceAccumulator | 10 | Stats, distribution, export |
| Constants | 4 | Enum counts |
| Cross-module Integration | 6 | Full pipeline, determinism |

## Benchmark Metrics (v2.15)

The benchmark runner (`scripts/run_runtime_benchmark.py`) reports:

```json
{
  "v215_adaptive_budget": {
    "avg_iterations_allocated": 15.2,
    "avg_iterations_used": 8.7,
    "budget_efficiency_ratio": 0.57,
    "stop_reason_distribution": {
      "converged": 45,
      "budget_exhausted": 10,
      "stagnated": 5,
      "divergence": 2,
      "escalate": 1
    }
  }
}
```

## Bug Fix

**`_is_diverging()` dual-convention fix**: When `classify_trajectory()` passes
`residuals[-1]` as both `current_residual` and part of `previous_residuals`,
the strict `>` comparison fails (current == last3[2]). Fixed to `>=` to handle
both calling conventions correctly.

## Files Modified/Created

| File | Action |
|------|--------|
| `backend/runtime/projection_scheduler.py` | Fix `_is_diverging` >= operator |
| `scripts/run_runtime_benchmark.py` | v2.15 metrics + persistence |
| `tests/test_v215_adaptive_scheduling.py` | 100 tests (new) |
| `docs/V215_ADAPTIVE_RUNTIME_SCHEDULING.md` | This document |

## Benchmark Methodology

### Correctness Preservation Benchmark

Run with `--compare-fixed-vs-adaptive` flag:

```bash
python scripts/run_runtime_benchmark.py \
    --dataset workspace/datasets/circuit_v29f_10k.pt \
    --compare-fixed-vs-adaptive \
    --seed 42
```

Compares fixed-budget (v2.14: max_iterations=20, patience=10) against
adaptive-budget (v2.15) execution. Acceptance conditions:

| Condition | Threshold |
|-----------|-----------|
| Residual correctness | adaptive <= fixed * 1.01 |
| KCL violation | adaptive <= fixed |
| KVL violation | adaptive <= fixed |
| Average iterations | adaptive < fixed |
| Average runtime | adaptive < fixed |

### Oscillatory Convergence Validation

```bash
pytest tests/test_v215_oscillatory_convergence.py -v
```

40 test cases covering: oscillatory-accepted, divergence-risk,
stalled, fast-converging, and `_is_diverging()` regression.

### Scheduler Overhead Validation

```bash
python scripts/validate_scheduler_overhead.py --n-samples 500 --seed 42
```

Measures `scheduler_efficiency_ratio = saved_projection_time / scheduler_overhead`.
Target: > 5.0x. Emits explicit warning if overhead exceeds savings.

### Retrieval Effectiveness Validation

```bash
python scripts/validate_retrieval_warmstart.py \
    --dataset workspace/datasets/circuit_v29f_10k.pt \
    --max-samples 100 --seed 42
```

Compares warmstart vs coldstart iterations. Uses two-sided sign test
for statistical significance. Reports honestly: if retrieval does not
help, says so explicitly.

## Retrieval-Assisted Scheduling

When the FAISS retrieval index contains a semantically similar past
execution (cosine similarity >= 0.7), the scheduler:

1. Routes to `retrieval_warmstart` path
2. Reduces budget to 60% of standard (fewer iterations needed)
3. Reduces patience to 70% of standard
4. Seeds the projection initial state with the retrieved solution

If similarity is between 0.3 and 0.7, routes to `retrieval_semantic`
with moderate budget. Below 0.3, falls through to standard routing.

Key invariant: retrieval similarity never bypasses the projection loop.
It only changes the starting point and the effort budget.

## Operational Metrics

| Metric | Source | Purpose |
|--------|--------|---------|
| `projection_iterations` | ProjectionResult | Convergence effort |
| `final_residual` | ProjectionResult | Solution quality |
| `scheduler_overhead_ms` | ExecutionOutcome | Scheduler cost |
| `projection_runtime_ms` | ExecutionOutcome | Projection cost |
| `total_runtime_ms` | ExecutionOutcome | End-to-end cost |
| `escalation_count` | ExecutionOutcome | Oracle intervention rate |
| `warmstart_iterations_saved` | ExecutionOutcome | Retrieval benefit |
| `trajectory_class` | TrajectoryAnalyzer | Convergence behavior |
| `convergence_class` | TrajectoryAnalyzer | Classification label |
| `budget_efficiency_ratio` | Benchmark | Used/allocated ratio |

## Reproducibility Instructions

1. Clone repository at tag `v2.15-runtime-stable`
2. Install dependencies: `pip install -r requirements.txt`
3. Verify: `pytest tests/ -q` (expect 697+ passing)
4. Run benchmark: `python scripts/run_runtime_benchmark.py --dataset <path> --seed 42`
5. Run overhead validation: `python scripts/validate_scheduler_overhead.py --seed 42`
6. Run retrieval validation: `python scripts/validate_retrieval_warmstart.py --seed 42`
7. Export experience: `python scripts/export_operational_experience.py --seed 42`
8. Export figures: `python scripts/export_paper_figures.py --seed 42`
9. Verify manifest: `cat runtime_release_manifest_v215.json`

All scripts accept `--seed` for deterministic reproduction.
Same seed + same dataset = identical results.

## Known Limitations

- Retrieval effectiveness depends on FAISS index size and diversity
- Scheduler overhead ratio depends on per-iteration projection cost
- Oscillatory trajectories may require oracle escalation
- Degraded mode reduces budget aggressively (by design)
- Warmstart effectiveness varies by topology family
