# CPT Simulator v2.13

A resilient, hybrid iterative solver for DC electrical circuit simulation. This project combines a Physics-Informed Graph Neural Network (PINN-GNN) with a deterministic Jacobi-style projection layer to achieve analytical accuracy while bypassing traditional $O(N^3)$ Modified Nodal Analysis (MNA) computational costs. 

As of version v2.13, the system features a hardened, production-grade core runtime that enforces deterministic cache hits, confidence-aware capability routing, structured execution recovery, and crash-safe atomic persistence.

---

## 🚀 System Architecture

### 1. Hybrid Solver Pipeline
The simulation resolves DC circuits via a two-stage hybrid pipeline:
1. **GNN Surrogate (`EdgeAwareCircuitGNN`):** Predicts initial node voltages from circuit graph representations. Implements dynamic topological feature extraction and logarithmic resistance normalization to maintain stability across extreme Out-Of-Distribution (OOD) resistance ranges ($0.1\Omega$ to $1M\Omega$).
2. **Physics Projection:** A deterministic, iterative Jacobi-style layer that corrects surrogate predictions to enforce Kirchhoff's Current Law (KCL) and Kirchhoff's Voltage Law (KVL).

### 2. True Global Virtual Node
To overcome the spectral radius bottleneck of purely local iterative solvers (where local iterations on high-diameter radial chains degrade convergence), the projection layer injects a virtual mathematical node. This node aggregates the global mean residual error and redistributes it universally, reducing the effective graph communication diameter to 1 and accelerating convergence.

### 3. Core Runtime Resilience (v2.13 Stack)
Version v2.13 introduces a hardened execution environment to guarantee zero silent failures, exact matches, and stable resource scaling:
* **Canonical Task Hashing (`task_hashing.py`):** Computes canonical SHA-256 hashes of circuit configurations. Sorts nodes alphabetically, orders edges by source-target indices, and normalizes float parameters to 8 significant figures to ensure equivalent topologies yield identical hashes.
* **Exact Match Cache (`exact_cache.py`):** Instantly retrieves matching outputs via `ExactMatchCache` without executing solvers, persisting cache entries to robust JSONL structures.
* **Execution Policy & Recovery (`execution_policy.py`):** Wraps operations in protective retry and degradation policies (`RecoveryHandler`). Detects NaN tensors, execution timeouts, surrogate instability, and projection divergence, marking degraded runs rather than failing silently.
* **Confidence Heuristics (`confidence_runtime.py`):** Mapped to deterministic rules (dynamic range, graph size, topological family, raw KCL residual) to produce a `ConfidenceEstimate` without stochastic operations.
* **Capability Router (`capability_router.py`):** Directs executions across 5 rule-based routing categories (`cache_hit`, `standard` with low projection budget, `increased_budget`, `ood_escalation` requiring oracle verification, and full `oracle_verification`).
* **Atomic Memory Persistence (`memory_runtime.py`):** Prevents file corruption via a write-to-temp, `fsync()`, and atomic `os.replace()` sequence. Includes compaction utilities.

---

## 🧠 Topological Curriculum and Failure Taxonomy

Training relies on structural progression rather than random sampling. Circuit graphs are scheduled and evaluated through `topology_curriculum.py`:
* **Trivial:** Trees, $\le 4$ nodes, 0 cycles.
* **Simple:** 1 independent cycle, $\le 6$ nodes.
* **Medium:** 2-3 cycles, $\le 10$ nodes.
* **Dense:** $> 3$ cycles, $> 10$ nodes (dense loops serve as physical regularizers, yielding exceptionally low baseline MAE).

### Structural Failure Taxonomy
Non-converging runs are categorized by topological cause to guide adaptive data generation:
* `cycle_drift_failure`: Non-convergent KCL residuals within closed cycles.
* `dense_mesh_leakage`: Signal attenuation in highly connected meshes.
* `bridge_node_instability`: Convergence drift across tree-like bridge bottlenecks.

---

## 📂 Workspace Structure (v2.13)

```
cpt_simulator_v5/
├── docs/
│   ├── AGENT_HANDOVER_V29F_COMPREHENSIVE.md  ← Contextual handover documentation
│   ├── ARCHITECTURE.md                       ← Structural architecture map
│   ├── CONTEXTO.md                           ← Core glossary & design constraints
│   ├── ESTADO_CURRICULO.md                   ← Curriculum execution status
│   ├── V29F_VIRTUAL_NODE_PROJECTION.md       ← Scientific report on Virtual Node
│   ├── V213_RUNTIME_RESILIENCE.md            ← Hardened runtime architecture (v2.13)
│   └── ORACLE_SDK_GUIDE.md                   ← Oracle integration guide
│
├── backend/
│   ├── circuits/
│   │   ├── dc_solver.py                ← Baseline analytical MNA oracle
│   │   ├── graph_dataset.py            ← Graph feature log-normalization
│   │   ├── physics_projection.py       ← Iterative projection with Virtual Node
│   │   ├── topology_curriculum.py      ← Structural complexity scheduler
│   │   └── failure_analysis.py         ← Topological failure classification
│   └── core_runtime/                   ← *[NEW in v2.13]* Resilient runtime engine
│       ├── exact_cache.py              ← Deterministic SHA-256 match cache
│       ├── task_hashing.py             ← Canonical float & structural hashing
│       ├── execution_policy.py         ← Safe policies & RecoveryHandler
│       ├── confidence_runtime.py       ← Heuristic confidence estimation
│       ├── capability_router.py        ← Rule-based task capability routing
│       └── memory_runtime.py           ← Atomic crash-safe memory operations
│
├── scripts/
│   ├── train_circuit_gnn.py            ← Training loop with PINN physics loss
│   ├── run_circuit_arena.py            ← Segregated topological evaluation
│   ├── run_runtime_benchmark.py        ← *[NEW in v2.13]* Benchmark with cache & routing metrics
│   └── compact_memory_store.py         ← *[NEW in v2.13]* Memory compaction utility
│
├── examples/
│   └── oracle_template.py              ← *[NEW in v2.13]* Working Oracle SDK reference
│
└── tests/
    ├── test_v29f_virtual_projection.py ← Physics projection unit tests
    ├── test_v29f_warmstart.py          ← Warm-start solver benchmark tests
    └── test_v213_resilient_runtime.py  ← *[NEW in v2.13]* 52 resilient runtime tests
```

---

## 🎯 System Performance and Verification

| Metric / Suite | Baseline (v2.9D) | Hardened Runtime (v2.13) |
| :--- | :---: | :---: |
| **Integrated Test Passes** | 16 / 16 | **141 / 141 PASSED (Zero Regression)** |
| **KCL Max Residual** | ~0.275 A | **$< 1e-6$ A** |
| **Cache Hit Recovery** | N/A | **Exact Match / Zero Solver Execution** |
| **Fault Recovery** | Silent Failures | **Zero Silent Failures (RecoveryHandler)** |
| **Write Integrity** | Standard File Open | **Crash-safe Atomic Temp-to-Replace** |

---

## 🛠️ Usage

Execute the physical validation and resilient runtime test suite:
```bash
pytest -v
```

Run the runtime benchmark to gather cache, latency, and routing distribution metrics:
```bash
python scripts/run_runtime_benchmark.py
```

Compact old run records and clean the atomic store:
```bash
python scripts/compact_memory_store.py
```
