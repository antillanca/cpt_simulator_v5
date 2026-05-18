# CPT v2.14 — Retrieval Memory, Semantic Warm-Start & Execution Cost Estimation

## Overview

v2.14 evolves the CPT runtime from a **hybrid simulator** into a **Deterministic Cognitive Execution Runtime** by adding semantic retrieval memory and warm-start capabilities. The core thesis:

| Component       | Role                                    |
|-----------------|-----------------------------------------|
| Oracle          | Provides correctness                    |
| Surrogate       | Provides acceleration                   |
| Projection      | Provides physical guarantees            |
| Retrieval       | Provides experience reuse               |
| Routing         | Provides adaptive execution             |
| Memory          | Provides continuity                     |
| Determinism     | Provides scientific reproducibility     |

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │           CapabilityRouter v2.14          │
                    │  (7 routing actions, cost-aware,          │
                    │   retrieval-aware, deterministic)         │
                    └─────┬────────┬────────┬──────────────────┘
                          │        │        │
           ┌──────────────┘        │        └──────────────┐
           ▼                       ▼                       ▼
  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐
  │  ExactMatchCache │   │ RetrievalMemory   │   │  CostEstimator   │
  │  (SHA-256 hit)   │   │ + FAISS Runtime   │   │  (heuristic)     │
  │  v2.13           │   │ (semantic search) │   │  v2.14           │
  └─────────────────┘   └────────┬─────────┘   └──────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  WarmstartRuntime       │
                    │  (similar solution →    │
                    │   initialize voltages)  │
                    └────────┬───────────────┘
                             │
                             ▼
                    ┌────────────────────────┐
                    │  ProjectionRuntime      │
                    │  (final authority)      │
                    └────────┬───────────────┘
                             │
                             ▼
                    ┌────────────────────────┐
                    │  ProjectionExperience   │
                    │  Memory (convergence    │
                    │  behavior storage)      │
                    └────────────────────────┘
```

## Memory Layer Separation (STRICT)

The three memory layers are **never mixed**:

| Layer      | Contents                                       | Package          |
|------------|------------------------------------------------|------------------|
| Knowledge  | Frozen specs, failure taxonomy, contracts      | `core_spec`      |
| Memory     | Exact executions, JSONL traces, deterministic  | `core_runtime`   |
| Experience | Embeddings, similarity retrieval, warm-start   | `runtime`        |

## Retrieval Flow

```
1. ExactMatchCache lookup (SHA-256)
   ├── HIT  → return cached result immediately
   └── MISS → continue to step 2

2. FAISS semantic search (inner product)
   ├── High similarity (≥0.5) → warmstart_projection
   ├── Moderate similarity (≥0.3) → semantic_retrieval
   └── No match → standard_projection

3. WarmstartRuntime evaluation
   ├── accepted → run projection with warmstart init
   └── rejected → fall back to standard initialization

4. ProjectionRuntime executes (FINAL AUTHORITY)
   │   Never bypassed. Always validates.
   └── Result stored in:
       ├── ExactMatchCache (for future exact hits)
       ├── RetrievalMemory (for future semantic search)
       ├── ProjectionExperienceMemory (for future learning)
       └── TraceStore (for audit)
```

## Exact Cache vs Semantic Retrieval

| Property          | ExactMatchCache         | FAISS Retrieval             |
|-------------------|-------------------------|------------------------------|
| Index method      | SHA-256 hash lookup     | Inner product similarity     |
| Match type        | Exact (byte-level)      | Approximate (cosine-like)    |
| Determinism       | Perfect                 | Deterministic ordering       |
| Priority          | ALWAYS FIRST            | SECOND (after cache miss)    |
| Storage           | JSONL + hash index      | FAISS IndexFlatIP            |
| Miss cost         | O(1) hash lookup        | O(n) linear scan             |

**CRITICAL**: Exact cache is ALWAYS checked first. FAISS retrieval is ONLY attempted after a cache miss.

## Warmstart Projection Flow

```
Standard path:
  zeros_init → projection(N iterations) → result

Warmstart path:
  retrieved_voltages → projection(M iterations) → result
  where M < N if warmstart is beneficial

Acceptance rule:
  warmstart accepted ONLY IF:
    initial_residual_after_warmstart < initial_residual_standard

Rejection triggers:
  - similarity < threshold (default: 0.5)
  - warmstart residual >= standard residual
  - convergence gain below minimum
  - projection diverges after warmstart → fallback

Projection is NEVER bypassed.
```

## Deterministic Retrieval Guarantees

1. **Embedding SHA-256**: Same graph → same GNN latent → same SHA-256
2. **Float32 canonicalization**: All embeddings normalized to float32 before hashing
3. **Deterministic ordering**: Retrieval results sorted by task_hash (alphabetical)
4. **FAISS search order**: Results ordered by similarity score (descending)
5. **No stochastic inference**: Dropout disabled, eval mode only
6. **Atomic persistence**: All index files written via temp → fsync → replace

## Failure Handling

### Rules (NEVER violated)

1. **Degraded executions are NEVER stored in retrieval index**
2. **NaN embeddings are NEVER added to FAISS**
3. **Low-confidence warmstarts are NEVER auto-accepted**
4. **Similarity below threshold → reject retrieval**
5. **Projection diverges after warmstart → fallback to standard init**
6. **All failures appear in trace logs**

### Safety checks in code

```python
# FaissRuntime.add_embedding()
if np.isnan(embedding).any():
    return False  # NaN rejected

# WarmstartRuntime.evaluate_warmstart()
if similarity < min_similarity:
    return WarmStartResult(accepted=False, ...)
if initial_residual_warmstart >= initial_residual_standard:
    return WarmStartResult(accepted=False, ...)

# Benchmark: only non-degraded results go to retrieval
if not is_degraded and not np.isnan(emb_np).any():
    retrieval_mem.add(entry)
    faiss_rt.add_embedding(...)
```

## Execution Cost Estimation (Heuristic)

Pure heuristics — no trained model yet.

| Factor                    | Effect on Cost                    |
|---------------------------|-----------------------------------|
| Node count                | +0.5 iterations/node             |
| Edge count                | +0.3 ms/edge                     |
| OOD topology              | 2x multiplier                    |
| Low confidence (<0.3)     | +1 difficulty score              |
| High resistance range     | +0.5-1 difficulty score          |
| Historical family avg     | 50/50 blend with heuristic       |

Difficulty levels: `trivial` → `easy` → `moderate` → `hard` → `extreme`

## CapabilityRouter v2.14 — Routing Actions

| Action               | Condition                                   | Budget    | Oracle |
|----------------------|---------------------------------------------|-----------|--------|
| exact_cache_hit      | SHA-256 cache match                         | 0         | No     |
| semantic_retrieval   | Partial similarity (0.3-0.5)                | Standard  | No     |
| warmstart_projection | High similarity (≥0.5)                      | Reduced   | No*    |
| standard_projection  | Default path                                | Standard  | No     |
| increased_budget     | OOD detected, no warmstart                  | High      | Yes    |
| oracle_verification  | Repeated failure topology (≥3)              | High      | Yes    |
| degraded_execution   | Runtime failure detected                    | High      | Yes    |

*Warmstart + OOD → oracle forced on

## Projection Experience Memory

Stores convergence behavior for future learning (LoRA, replay, adaptive routing).

Each entry records:
- `task_hash`, `topology_family`
- `initial_residual`, `final_residual`, `residual_slope`
- `iterations`, `converged`
- `kcl_residual`, `kvl_residual`
- `used_warmstart`, `warmstart_similarity`

Family-level statistics:
- Average iterations
- Convergence rate
- Average residual slope
- Warmstart usage rate

## Benchmark Methodology

v2.14 benchmark adds these metrics over v2.13:

| Metric                        | Description                           |
|-------------------------------|---------------------------------------|
| exact_cache_hit_rate          | SHA-256 exact matches                 |
| retrieval_hit_rate            | FAISS similarity matches              |
| warmstart_acceptance_rate     | Warmstarts accepted vs attempted      |
| avg_iterations_saved          | Iterations saved by warmstart         |
| avg_similarity                | Average retrieval similarity           |
| avg_confidence                | Average confidence score              |
| avg_estimated_cost            | Average heuristic cost estimate       |
| degraded_rate                 | Fraction of degraded executions       |
| routing_distribution          | Count per routing action              |
| projection_budget_distribution| Iteration budget buckets              |
| standard_vs_warmstart         | Comparison of avg iterations          |

## New Modules (backend/runtime/)

| File                          | Class/Function                        | Lines |
|-------------------------------|---------------------------------------|-------|
| `retrieval_memory.py`         | `RetrievalEntry`, `RetrievalMemory`   | ~250  |
| `embedding_runtime.py`        | `EmbeddingResult`, `extract_graph_embedding` | ~140 |
| `faiss_runtime.py`            | `FaissRuntime`, `TopKSimilarityResult` | ~260  |
| `warmstart_runtime.py`        | `WarmstartRuntime`, `WarmStartResult` | ~180  |
| `cost_estimator.py`           | `CostEstimator`, `ExecutionCostEstimate` | ~170  |
| `projection_experience.py`    | `ProjectionExperienceMemory`, `ProjectionExperienceEntry` | ~170 |

## Updated Modules

| File                          | Change                                |
|-------------------------------|---------------------------------------|
| `core_runtime/capability_router.py` | 7 routing actions (was 5), retrieval + cost aware |
| `core_runtime/__init__.py`    | Exports updated for v2.14 router      |
| `scripts/run_runtime_benchmark.py` | v2.14 metrics, retrieval/warmstart/cost integration |

## Test Coverage

73 new tests in `tests/test_v214_retrieval_memory.py`:

| Category                          | Tests |
|-----------------------------------|-------|
| RetrievalEntry construction       | 8     |
| RetrievalMemory CRUD              | 10    |
| EmbeddingResult + hashing         | 7     |
| GNN embedding extraction          | 3     |
| FAISS runtime                     | 7     |
| WarmstartRuntime                  | 6     |
| CostEstimator                     | 10    |
| CapabilityRouter v2.14            | 11    |
| ProjectionExperienceMemory        | 5     |
| E2E Integration                   | 6     |
| **Total**                         | **73** |

## Dependencies

| Package      | Version  | Purpose                    |
|--------------|----------|----------------------------|
| faiss-cpu    | 1.13.2   | Semantic similarity search |
| torch        | ≥2.0     | GNN encoder inference      |
| numpy        | ≥1.24    | FAISS input format         |

## Future Roadmap

| Phase | Feature                        | Depends On                    |
|-------|--------------------------------|-------------------------------|
| v2.15 | LoRA experts per topology      | ProjectionExperience data     |
| v2.16 | Replay learning from experience| Warmstart + Experience data   |
| v2.17 | Adaptive projection schedulers | Cost estimation feedback      |
| v2.18 | Multi-domain runtime           | Domain-agnostic architecture  |
| v2.19 | FAISS IVF index (scaling)      | Large-scale retrieval (>100K) |
| v2.20 | Continual learning loop        | All of the above              |

## Scientific Contribution

The v2.14 runtime demonstrates that:

1. **Retrieval-augmented execution** can reduce projection iterations without sacrificing physical correctness
2. **Deterministic retrieval** is achievable with canonical hashing + FAISS inner product
3. **Warmstart from experience** is safe when projection remains the final authority
4. **Cost-aware routing** enables intelligent budget allocation without ML
5. **Experience collection** prepares the ground for future learning without implementing it prematurely

The system remains **fully reproducible**: identical inputs produce identical outputs across runs, machines, and time.
