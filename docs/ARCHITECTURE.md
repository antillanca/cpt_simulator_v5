# System Architecture (v2.13)

This document formalizes the architectural specifications of the CPT Simulator v5, a resilient, hybrid DC circuit solver combining neural pre-conditioning, physical projection, and a deterministic capability routing runtime.

---

## 1. Hybrid Solver Core

The core simulation pipeline uses a multi-tiered approach to resolve voltages across electrical graphs.

```
                    ┌────────────────────────┐
                    │      RuntimeTask       │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   ExactMatchCache      │  ← Deterministic SHA-256
                    └───────────┬────────────┘
                          Miss  │  Hit
                                ├──────────────────────────┐
                                │                          │
                    ┌───────────▼────────────┐             │
                    │   ConfidenceRuntime    │             │
                    └───────────┬────────────┘             │
                                │                          │
                    ┌───────────▼────────────┐             │
                    │    CapabilityRouter    │             │
                    └───────────┬────────────┘             │
                                │                          │
       ┌────────────────────────┼────────────────────────┐ │
       │                        │                        │ │
 standard                ood_escalation        oracle_verification
 (Low Budget)            (High Budget)         (High Budget + Oracle)
       │                        │                        │ │
       └────────────────────────┼────────────────────────┘ │
                                │                          │
                    ┌───────────▼────────────┐             │
                    │    RuntimeExecutor     │             │
                    │  (Surrogate + Proj.)   │             │
                    └───────────┬────────────┐             │
                                │                          │
                    ┌───────────▼────────────┐             │
                    │    RecoveryHandler     │  ← Checks NaNs / Timeout / Divergence
                    └───────────┬────────────┘             │
                                │                          │
                    ┌───────────▼────────────┐             │
                    │     MemoryRuntime      │  ← Atomic Write-to-Replace
                    └───────────┬────────────┘             │
                                │                          │
                    ┌───────────▼────────────┐             │
                    │    ExactMatchCache     │             │
                    │       .put()           │             │
                    └───────────┬────────────┘             │
                                ├──────────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │     RuntimeResult      │
                    └────────────────────────┘
```

### 1.1 Ground Truth Oracle (`backend/circuits/dc_solver.py`)
- **Role**: Reference analytical solver utilizing Modified Nodal Analysis (MNA).
- **Complexity**: $O(N^3)$. Invoked for ground-truth synthesis, OOD escalation, and strict verification loops.

### 1.2 GNN Surrogate Pre-conditioner (`backend/neural/models/circuit_gnn.py`)
- **Role**: Establishes optimal initial state estimates near-instantaneously.
- **Features**: Physics-Informed Graph Neural Network (PINN-GNN) equipped with topological features and logarithmic resistance scaling to safeguard gradient values under extreme conditions ($0.1\Omega$ to $1M\Omega$).

### 1.3 Deterministic Physics Projection (`backend/circuits/physics_projection.py`)
- **Role**: Implements iterative Jacobi-style corrections to eliminate residual violations of physical laws (KCL/KVL).
- **True Global Virtual Node**: Connects to all nodes to compute and redistribute the global mean residual, compressing the communication diameter to 1 and preventing spectral radius decay in long radial networks.

---

## 2. Hardened Core Runtime Stack (v2.13)

The v2.13 layer manages the solver components within a robust, fault-tolerant execution container:

### 2.1 Canonical Hashing & Exact Cache
- **Task Hashing (`task_hashing.py`)**: Computes SHA-256 hashes of circuit configurations. Sorts nodes alphabetically, sequences edges, and rounds floating-point values to 8 significant figures to guarantee that isomorphic circuit structures yield identical keys.
- **Exact Cache (`exact_cache.py`)**: Intercepts solver calls via `ExactMatchCache` to instantly serve historical matches, bypassing iteration overhead completely.

### 2.2 Execution Policy & Error Recovery
- **Execution Policy (`execution_policy.py`)**: Enforces timeout, retry, and iteration budget parameters (`ExecutionPolicy`).
- **Recovery Handler (`execution_policy.py`)**: Intercepts NaNs, time-limit expirations, surrogate instabilities (where predictions wildly deviate from physical outputs), and projection divergences, registering them under explicit degradation flags rather than failing silently.

### 2.3 Confidence Routing
- **Confidence Estimation (`confidence_runtime.py`)**: Evaluates task complexity (dynamic range, graph size, topological family, raw KCL residual) to determine a deterministic confidence rating.
- **Capability Router (`capability_router.py`)**: Maps confidence to five execution pathways (`cache_hit`, `standard` with low budget, `increased_budget`, `ood_escalation`, or full `oracle_verification`).

### 2.4 Atomic Memory Persistence (`memory_runtime.py`)
- **Atomic Writes**: Eliminates file truncation risks. Writes data to a temporary file, calls `os.fsync()` to force disk flush, and invokes `os.replace()` for an atomic rename.
- **Compaction**: Employs `compact_memory_store.py` to prune obsolete or redundant transaction records.

---

## 3. Structural Curriculum and Failure Taxonomy

- **Topological Curriculum (`topology_curriculum.py`)**: Governs gradient progression using a structured curriculum (Trivial trees $\rightarrow$ Simple loops $\rightarrow$ Medium cycles $\rightarrow$ Dense meshes).
- **Failure Taxonomy (`failure_analysis.py`)**: Diagnoses physical anomalies by structural root cause (`cycle_drift_failure`, `dense_mesh_leakage`, `bridge_node_instability`) rather than plain MSE.
