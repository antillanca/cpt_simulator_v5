# CPT v2.14 — Agent Handover Document

## Current Version: v2.14
## Status: COMPLETE — All 11 phases delivered, 214/214 tests passing

---

## COMPLETED VERSIONS

### v2.11 — Core Standardization
- RuntimeTask, RuntimeResult, RuntimeExecutor
- Oracle/Surrogate/Projection/Evaluator protocols
- MNAOracleAdapter, SurrogateRuntime
- ProjectionRuntime with KCL/KVL enforcement
- MemoryRuntime with atomic persistence
- ExecutionTrace + TraceStore
- DatasetManifest + DatasetRegistry

### v2.12 — Runtime Integration
- Full pipeline: task → oracle → surrogate → projection → evaluation → memory
- EvaluationReport with MAE/KCL/KVL metrics
- E2E test coverage

### v2.13 — Resilient Runtime
- ExactMatchCache with SHA-256 canonical hashing
- ExecutionPolicy + RecoveryHandler
- ConfidenceRuntime (heuristic estimation)
- CapabilityRouter (5 routing actions)
- Atomic memory persistence (fsync + os.replace)
- CompactMemoryStore utility

### v2.14 — Retrieval Memory, Semantic Warm-Start & Cost Estimation
- RetrievalMemory (deterministic, SHA-256 indexed, atomic persistence)
- EmbeddingRuntime (GNN latent extraction, inference-only)
- FaissRuntime (IndexFlatIP, NaN rejection, exact cache priority)
- WarmstartRuntime (accepted only if residual improves)
- CostEstimator (heuristic, 5 difficulty levels)
- ProjectionExperienceMemory (convergence behavior storage)
- CapabilityRouter v2.14 (7 routing actions, retrieval + cost aware)
- Benchmark v2.14 (11 new metrics)

---

## ACTIVE STATE

- **Directory**: `/home/john/www/cpt_simulator_v5`
- **Git**: branch `master`, commit `191795d`
- **Tests**: 214/214 passing (0 failures)
  - test_v211: core standardization
  - test_v212: runtime integration
  - test_v213: resilient runtime
  - test_v214: retrieval memory + warmstart + cost
- **FAISS**: installed (v1.13.2, CPU)
- **Hash Schema**: "v1" (via task_hashing.py)

---

## MEMORY LAYERS (STRICTLY SEPARATED)

| Layer      | Package          | Contents                                      |
|------------|------------------|-----------------------------------------------|
| Knowledge  | `core_spec`      | Frozen specs, failure taxonomy, contracts     |
| Memory     | `core_runtime`   | Exact executions, JSONL traces, deterministic |
| Experience | `runtime`        | Embeddings, similarity retrieval, warm-start  |

---

## KEY ARCHITECTURAL DECISIONS

1. **Exact cache ALWAYS first** — FAISS retrieval only after cache miss
2. **Projection is final authority** — warmstart NEVER bypasses projection
3. **Degraded executions NEVER stored in retrieval** — no contamination
4. **NaN embeddings NEVER added to FAISS** — safety guarantee
5. **All routing 100% deterministic** — no ML routing yet
6. **Float32 canonicalization before hashing** — determinism across hardware
7. **Atomic persistence everywhere** — temp → fsync → replace

---

## ROUTING ACTIONS (v2.14)

| Priority | Action               | Condition                          |
|----------|----------------------|------------------------------------|
| 1        | exact_cache_hit      | SHA-256 cache match                |
| 2        | degraded_execution   | Runtime failure detected           |
| 3        | oracle_verification  | Repeated failure topology (≥3)     |
| 4        | increased_budget     | OOD, no warmstart                  |
| 5        | warmstart_projection | High similarity (≥0.5)             |
| 6        | semantic_retrieval   | Moderate similarity (≥0.3)         |
| 7        | standard_projection  | Default                            |

---

## FILES (v2.14 additions)

### New (backend/runtime/)
- `__init__.py` — Package exports
- `retrieval_memory.py` — RetrievalEntry, RetrievalMemory
- `embedding_runtime.py` — EmbeddingResult, extract_graph_embedding
- `faiss_runtime.py` — FaissRuntime, TopKSimilarityResult
- `warmstart_runtime.py` — WarmstartRuntime, WarmStartResult
- `cost_estimator.py` — CostEstimator, ExecutionCostEstimate
- `projection_experience.py` — ProjectionExperienceMemory, ProjectionExperienceEntry

### Updated
- `core_runtime/capability_router.py` — 7 routing actions (was 5)
- `core_runtime/__init__.py` — Updated exports
- `scripts/run_runtime_benchmark.py` — v2.14 metrics
- `tests/test_v213_resilient_runtime.py` — Action name migration

### Docs
- `docs/V214_RETRIEVAL_MEMORY_RUNTIME.md` — Full architecture doc

### Tests
- `tests/test_v214_retrieval_memory.py` — 73 new tests

---

## REMAINING WORK (Future Roadmap)

| Phase | Feature                        | Depends On                    |
|-------|--------------------------------|-------------------------------|
| v2.15 | LoRA experts per topology      | ProjectionExperience data     |
| v2.16 | Replay learning from experience| Warmstart + Experience data   |
| v2.17 | Adaptive projection schedulers | Cost estimation feedback      |
| v2.18 | Multi-domain runtime           | Domain-agnostic architecture  |
| v2.19 | FAISS IVF index (scaling)      | Large-scale retrieval (>100K) |
| v2.20 | Continual learning loop        | All of the above              |

---

## CONSTRAINTS FOR FUTURE AGENTS

1. DO NOT modify GNN architecture or projection equations
2. DO NOT mix Knowledge/Memory/Experience layers
3. DO NOT introduce stochastic inference (no dropout at runtime)
4. DO NOT replace exact cache with FAISS (they coexist)
5. DO NOT cache degraded executions as valid hits
6. ALL retrieval decisions MUST remain deterministic
7. Maintain frozen=True on all result dataclasses
8. Use SHA-256 for all hashing (no MD5, no SHA-1)
9. Atomic persistence: temp → fsync → os.replace
10. Test before commit: 214/214 must pass
