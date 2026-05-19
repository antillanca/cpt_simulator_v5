# CORE Runtime Guarantees

## Deterministic Guarantees

### Same Input = Same Trace

Given identical task input and identical runtime configuration,
the CORE runtime produces an identical execution trace. This is
ensured by:

- Deterministic task hashing (SHA-256 of canonical task representation)
- Deterministic capability routing (same task = same route)
- Deterministic scheduling (same trajectory = same decisions)
- Deterministic cache lookups (same hash = same result)

### Hash Canonicalization

Task fingerprints use deterministic serialization:
- Sorted dictionary keys
- Fixed-precision floating point
- Canonical byte ordering

This guarantee is frozen from v2.15 and MUST NOT change.

## Frozen APIs (from v2.15)

The following APIs are frozen and remain valid in v3.0.x:

- DomainTaskBase interface
- ExactMatchCache lookup semantics
- RetrievalMemory query semantics
- TrajectoryAnalyzer classification
- ProjectionBudget allocation
- ExecutionTrace recording
- OperationalExperienceSchema serialization
- ExperienceDatasetSchema structure

Breaking any frozen API requires a MAJOR version bump.

## Backward Compatibility

### Import Path Compatibility

Old import paths remain functional through shims:

| Old Path | New Path | Shim Status |
|---|---|---|
| backend.runtime.* | core_runtime.core.* | Active with deprecation warning |
| backend.core_runtime.* | core_runtime.core.* | Active with deprecation warning |
| backend.core_spec.* | core_runtime.core.specs.* | Active with deprecation warning |

Shims will be removed in v3.1+. Deprecation warnings are
emitted on first import.

### Data Compatibility

- v2.15 operational experience JSONL/CSV remain loadable
- v2.15 benchmark results remain readable
- v2.15 paper figures remain valid
- v2.15 release manifest remains verifiable

## Projection Guarantees

1. Projection is always the final authority
2. No scheduling decision bypasses projection
3. Warmstart solutions must pass projection validation
4. Cached solutions must pass projection verification
5. Projection mathematics are domain-specific and untouched by core

## Cache Guarantees

1. Exact cache always has priority over surrogate
2. Cache hit = exact match on deterministic hash
3. Cache misses are deterministic (same hash = same miss)
4. Hash canonicalization rules are frozen

## Retrieval Guarantees

1. Retrieval provides initialization hints only
2. Retrieved solutions never bypass projection
3. Degraded executions are excluded from clean indexes
4. Retrieval scoring is deterministic for same index state

## Scheduling Guarantees

1. Adaptive scheduling never sacrifices correctness for speed
2. Scheduler overhead is measured and bounded
3. Escalation to oracle is always available as fallback
4. Budget allocation is deterministic for same trajectory

## Degraded Execution Guarantees

1. Degraded executions are tagged in traces
2. Degraded solutions never enter clean retrieval indexes
3. Degraded executions are fully traceable
4. Degraded execution rates are reported in metrics

## Known Limitations

- Single-process execution only
- No distributed computation
- No online learning or model updates
- No LoRA, replay, or continual training in v3.0.x
- Circuit domain adapters have import resolution pending

## Future Roadmap Boundaries

The following are explicitly OUT OF SCOPE for v3.0.x:

- LoRA fine-tuning of surrogates
- Replay learning from operational experience
- Continual training pipelines
- Distributed execution
- Adaptive physics equations

These may appear in v3.1+ or v4.0+ with appropriate versioning.
