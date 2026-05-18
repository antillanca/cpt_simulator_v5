# CPT Simulator v2.13 — Master Handover Document

> **Purpose**: Transfer comprehensive project context to any incoming developer or AI agent. This document outlines the system architecture, design rationale, v2.13 core runtime features, current status, and immediate next steps.

---

## 🎯 Project Scope

The CPT Simulator is a hybrid, fault-tolerant simulation engine for DC electrical circuits. It couples a Physics-Informed Graph Neural Network (PINN-GNN) acting as a fast pre-conditioner with a deterministic analytical projection solver to bypass the $O(N^3)$ computational cost of Modified Nodal Analysis (MNA) while guaranteeing physical conservation laws (KCL/KVL). 

As of version v2.13, the simulation environment is managed by a production-grade, resilient runtime that ensures deterministic cache hits, confidence-aware routing, automated error recovery, and atomic data persistence.

---

## 🏗️ Architecture Stack

The execution pipeline is organized into a highly structured tiered stack:

### 1. Ground Truth Oracle (`backend/circuits/dc_solver.py`)
- **Role**: Reference analytical solver utilizing Modified Nodal Analysis (MNA).
- **Usage**: Invoked only during dataset generation, test validations, and OOD escalation.

### 2. GNN Surrogate (`backend/neural/models/circuit_gnn.py`)
- **Role**: Establishes optimal warm-start initial states.
- **Features**: Utilizes topological feature extraction and logarithmic resistance scaling to maintain numerical stability during Out-Of-Distribution (OOD) resistance ranges.

### 3. Physics Projection Layer (`backend/circuits/physics_projection.py`)
- **Role**: Deterministic iterative Jacobi-style corrector.
- **Mechanism**: Eliminates KCL/KVL residual violations.
- **Virtual Node Integration**: Features `VirtualNodeProjection` to bypass spectral radius degradation in long radial networks, reducing the effective graph communication diameter to 1.

### 4. Resilient Core Runtime (`backend/core_runtime/`) — *[NEW in v2.13]*
- **Canonical Task Hashing (`task_hashing.py`)**: Generates unique SHA-256 hashes of circuit configurations. Sorts nodes/edges alphabetically and normalizes floats to 8 significant figures to map equivalent circuits to identical hashes.
- **Exact Match Cache (`exact_cache.py`)**: Stores and retrieves exact execution traces based on SHA-256 hashes to completely bypass redundant solver runs.
- **Execution Policy & Recovery (`execution_policy.py`)**: Monitors runtime execution via a robust `RecoveryHandler`. Intercepts timeouts, NaNs, surrogate instabilities, and projection divergences, marking them as degraded executions to prevent silent failures.
- **Confidence Heuristics (`confidence_runtime.py`)**: Computes deterministic confidence ratings (between 0.0 and 1.0) using topological features, graph size, and residual histories.
- **Capability Router (`capability_router.py`)**: Dynamically assigns execution paths (`cache_hit`, `standard` with low budget, `increased_budget`, `ood_escalation`, or full `oracle_verification`) according to computed confidence estimates.
- **Atomic Memory Persistence (`memory_runtime.py`)**: Combats storage corruption via atomic execution writes (temp write -> `fsync()` -> `os.replace()`).

---

## 🧠 Topological Curriculum and Diagnostics

### Structural Progression (`topology_curriculum.py`)
Training circuits are injected progressively through a deterministic curriculum to stabilize gradient paths:
- **Trivial**: Tree structures (0 independent cycles, $\le 4$ nodes).
- **Simple**: Single-loop circuits (1 cycle, $\le 6$ nodes).
- **Medium**: Moderately coupled loops (2-3 cycles, $\le 10$ nodes).
- **Dense**: Highly interconnected meshes ($> 3$ cycles). *Note: Highly interconnected meshes serve as natural graph regularizers, yielding exceptionally low baseline MAE.*

### Structural Failure Taxonomy (`failure_analysis.py`)
Unprojectable or degraded runs are classified according to topological root causes to guide dataset generation:
- `cycle_drift_failure`: Non-convergent KCL residuals within closed loops.
- `dense_mesh_leakage`: Signal attenuation in highly connected meshes.
- `bridge_node_instability`: Convergence drift across tree-like bridge bottlenecks.

---

## 📊 Current State (May 2026)

- **Phase 1 (Symbolic Domain Acquisition)**: 100% Complete.
- **Phase 2 (Topological Graph Curriculum)**: 100% Complete.
- **Phase 3 (Core Runtime Resilience)**: 100% Complete (v2.13 integration achieved).

### Unified Global Validation
- **Global Test Passes**: **141 / 141 PASSED** (0 regression).
- **Exact Cache Hit Rate**: 100% for isomorphic topologies.
- **Recovery Handler Integrity**: Zero silent failures observed; NaN, timeout, instability, and divergence events successfully caught and classified into one of five distinct degraded categories.
- **Atomic Writes**: Zero filesystem corruption under simulated execution crash-testing.

---

## 📂 Critical File Map

```
cpt_simulator_v5/
├── backend/
│   ├── circuits/
│   │   ├── dc_solver.py                ← Analytical oracle
│   │   ├── graph_dataset.py            ← Graph log-normalization features
│   │   ├── physics_projection.py       ← Virtual Node Projection layer
│   │   ├── topology_curriculum.py      ← Curriculum scheduler
│   │   └── failure_analysis.py         ← Topological failure taxonomy
│   ├── core_runtime/                   ← Hardened core runtime stack (v2.13)
│   │   ├── exact_cache.py              ← Deterministic match cache
│   │   ├── task_hashing.py             ← Canonical float & structure hashing
│   │   ├── execution_policy.py         ← Failure recovery & execution policies
│   │   ├── confidence_runtime.py       ← Heuristic confidence calculator
│   │   ├── capability_router.py        ← Five-way capability routing module
│   │   └── memory_runtime.py           ← Atomic persistence engine
│   └── neural/
│       └── models/
│           └── circuit_gnn.py          ← PINN-GNN Surrogate model
│
├── scripts/
│   ├── train_circuit_gnn.py            ← Training orchestrator
│   ├── run_circuit_arena.py            ← Topological benchmark suite
│   ├── run_runtime_benchmark.py        ← Extended runtime benchmarks (v2.13)
│   └── compact_memory_store.py         ← Memory store compaction utility
│
├── docs/
│   ├── V213_RUNTIME_RESILIENCE.md      ← Resilient runtime design specs
│   ├── ORACLE_SDK_GUIDE.md             ← Guide for external solvers
│   ├── V29F_VIRTUAL_NODE_PROJECTION.md ← Scientific report on Virtual Node
│   └── V29E_TOPOLOGY_AWARE_SURROGATE.md← Scientific report on GNN ablation
│
└── tests/
    ├── test_v29f_virtual_projection.py ← Physics projection tests
    ├── test_v29f_warmstart.py          ← Warm-start iteration tests
    └── test_v213_resilient_runtime.py  ← 52 E2E runtime resiliency tests
```

---

## ⚙️ Operational Commands

**Execute validation and resilient runtime test suites:**
```bash
pytest -v
```

**Run the runtime benchmark to gather cache, latency, and routing metrics:**
```bash
python scripts/run_runtime_benchmark.py
```

**Run the memory compaction utility:**
```bash
python scripts/compact_memory_store.py
```
