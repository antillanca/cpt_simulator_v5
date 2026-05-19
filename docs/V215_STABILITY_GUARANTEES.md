# CPT v2.15 -- Stability Guarantees

## Release Tag

`v2.15-runtime-stable`

## Frozen APIs

The following public interfaces are frozen and will not change in
patch releases within v2.15.x:

| API | Module | Status |
|-----|--------|--------|
| `ProjectionBudget` | `backend/runtime/projection_scheduler.py` | FROZEN |
| `ProjectionScheduler` | `backend/runtime/projection_scheduler.py` | FROZEN |
| `StopDecision` | `backend/runtime/projection_scheduler.py` | FROZEN |
| `TrajectoryAnalyzer` | `backend/runtime/trajectory_analysis.py` | FROZEN |
| `TrajectoryMetrics` | `backend/runtime/trajectory_analysis.py` | FROZEN |
| `ExecutionScheduler` | `backend/runtime/execution_scheduler.py` | FROZEN |
| `ExecutionSchedule` | `backend/runtime/execution_scheduler.py` | FROZEN |
| `ExecutionOutcome` | `backend/runtime/execution_scheduler.py` | FROZEN |
| `OperationalExperienceEntry` | `backend/runtime/operational_experience_schema.py` | FROZEN |
| `OperationalExperienceAccumulator` | `backend/runtime/operational_experience_schema.py` | FROZEN |
| `ExactMatchCache` | `backend/runtime/exact_cache.py` | FROZEN |
| `RetrievalMemory` | `backend/runtime/retrieval_memory.py` | FROZEN |
| `RetrievalEntry` | `backend/runtime/retrieval_memory.py` | FROZEN |
| `WarmstartRuntime` | `backend/runtime/warmstart_runtime.py` | FROZEN |
| `ConfidenceRuntime` | `backend/runtime/confidence_runtime.py` | FROZEN |
| `CostEstimator` | `backend/runtime/cost_estimator.py` | FROZEN |
| `CapabilityRouter` | `backend/runtime/capability_router.py` | FROZEN |

Frozen means: no signature changes, no field removals, no semantic
changes to existing methods. Additions (new methods, new optional
fields with defaults) are permitted in minor versions.

## v2.16 Schema Preview (Frozen)

The following data schemas are frozen in v2.15 for v2.16 consumption:

| Schema | Module | Status |
|--------|--------|--------|
| `ConvergenceTraceSchema` | `backend/runtime/experience_dataset_schema.py` | FROZEN |
| `RoutingOutcomeSchema` | `backend/runtime/experience_dataset_schema.py` | FROZEN |
| `RetrievalOutcomeSchema` | `backend/runtime/experience_dataset_schema.py` | FROZEN |
| `EscalationEventSchema` | `backend/runtime/experience_dataset_schema.py` | FROZEN |
| `WarmstartPerformanceSchema` | `backend/runtime/experience_dataset_schema.py` | FROZEN |
| `TopologyClusterSchema` | `backend/runtime/experience_dataset_schema.py` | FROZEN |
| `RuntimeCostDistributionSchema` | `backend/runtime/experience_dataset_schema.py` | FROZEN |
| `DatasetManifestSchema` | `backend/runtime/experience_dataset_schema.py` | FROZEN |

## Deterministic Guarantees

1. **Same inputs + same seed = same execution trace.** The scheduler,
   projection loop, and trajectory analysis are all pure functions of
   their inputs. No hidden state, no randomness.

2. **Budget allocation is deterministic.** Given (confidence,
   cost_estimate, retrieval_similarity, is_ood, family_stats), the
   ProjectionScheduler always returns the same ProjectionBudget.

3. **Routing is deterministic.** Given (task, cache_hit,
   retrieval_similarity, is_degraded, node_count, edge_count), the
   ExecutionScheduler always selects the same route.

4. **Trajectory classification is deterministic.** Given the same
   residual history, TrajectoryAnalyzer always returns the same class.

5. **Operational experience entries are immutable.** Once created,
   entries cannot be modified. The accumulator only appends.

## Supported Runtime Contracts

| Contract | Guarantee |
|----------|-----------|
| Cache hit | Returns cached result in O(1), no projection |
| Warmstart | Projection runs with reduced budget |
| Standard | Projection runs with full budget |
| OOD escalated | Projection runs with 1.5x budget + oracle |
| Oracle forced | Oracle-only execution after 3+ failures |
| Degraded | Oracle fallback, result excluded from indexes |

## Backward Compatibility Expectations

- v2.15.x patch releases maintain full test backward compatibility
- v2.16 may add new fields to frozen schemas (with defaults)
- v2.16 may add new routes and outcomes (extending enums)
- v2.16 must not break v2.15 saved operational experience JSONL
- v2.16 must not alter projection physics or convergence criteria

## Stability Freeze Rules

After this tag is created:

1. NO feature additions
2. NO architectural changes
3. ONLY critical bug fixes allowed
4. All fixes must include regression tests
5. All fixes must be verified against full test suite (698+ tests)
6. Any fix that changes frozen API signatures requires minor version bump

## What v2.16 Will Do (NOT in v2.15)

The following are explicitly OUT OF SCOPE for v2.15:

- LoRA fine-tuning of the GNN
- Replay buffers or experience replay
- Continual training or online learning
- Distributed execution
- Adaptive physics equations
- Modified projection mathematics
- Modified GNN architecture
- Modified retrieval semantics

v2.16 will CONSUME the operational experience datasets exported by
v2.15, using the frozen schemas in `experience_dataset_schema.py`.
