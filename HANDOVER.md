# CPT Simulator v2.13 — Executive Handover

> **Current Status (May 2026)**: 🚀 **MILESTONE ACHIEVED: v2.13 Core Runtime Resilience, Exact Cache & Confidence Routing.**
> The system has transitioned from a basic physics projection pipeline to a fully production-grade, fault-tolerant execution environment with zero regression (141/141 passed).

---

## 🏗️ Architecture: The Resilient Solver Stack

The CPT Simulator is a hybrid, fault-tolerant execution engine that couples neural pre-conditioning with analytical solvers, managed by a deterministic routing runtime.

### 1. Ground Truth Oracle & GNN Pre-conditioning
- **Path**: `backend/circuits/dc_solver.py` & `backend/neural/models/circuit_gnn.py`
- **Logic**: A reference analytical MNA solver paired with a PINN-GNN surrogate. GNN estimates serve as a fast warm-start to drastically reduce iterations.

### 2. Physics Projection & True Global Virtual Node
- **Path**: `backend/circuits/physics_projection.py`
- **Logic**: Enforces strict physical invariants (KCL/KVL). Features a global virtual node that aggregates and distributes residual drift, avoiding the spectral radius convergence bottleneck.

### 3. Resilient Core Runtime (v2.13 Layer)
- **Path**: `backend/core_runtime/`
- **Modules**:
  - `task_hashing.py`: Creates canonical SHA-256 hashes of task inputs by sorting topology nodes/edges and normalizing floats to 8 significant figures.
  - `exact_cache.py`: Intercepts tasks with matching hashes to instantly return analytical results, avoiding redundant evaluation.
  - `execution_policy.py`: Governs timeout constraints, retry logic, and degradation monitoring (`RecoveryHandler`).
  - `confidence_runtime.py`: Computes deterministic heuristic confidence scores (OOD flag, dynamic range, graph size, residual history) without stochastic variance.
  - `capability_router.py`: Selects one of five routing paths based on task confidence (`cache_hit`, `standard`, `increased_budget`, `ood_escalation`, `oracle_verification`).
  - `memory_runtime.py`: Implements crash-safe persistence (temp write -> `fsync()` -> atomic `os.replace()`).

---

## 📈 Progression & Verification Metrics (v2.13)

- **Total Unit & Integration Tests:** 141/141 PASSED.
- **Cache Hit Integrity:** Equivalent circuit topologies successfully produce identical hashes; cache retrieval completely bypasses solver iteration overhead.
- **Fail-Safe Recovery:** No silent failures occur. All timeouts, NaNs, surrogate instabilities, and projection divergences are intercepted by the `RecoveryHandler` and registered as degraded states.
- **Atomic Operations:** Crash-safe memory writes eliminate file truncation risk during sudden execution interruptions.

| Metric | Baseline (v2.9) | Resilient Runtime (v2.13) |
|:---|:---:|:---:|
| **Test Coverage (Global)** | 16 Tests | **141 Tests** |
| **Silent Failures** | Possible | **0 (Strictly Blocked)** |
| **Persistence Method** | Standard File Write | **Atomic Temp-to-Replace with Fsync** |
| **OOD Safety** | High-Budget Iteration | **Router Escalation to Oracle Verification** |

---

## 🔮 Roadmap: Immediate Objectives

Incoming agents and developers should focus on the following core objectives:

1. **Vector-Based Retrieval Memory (FAISS)**:
   Implement fuzzy match retrieval using FAISS vectors to complement the deterministic exact cache (`exact_cache.py`), allowing nearest-neighbor initialization for sub-threshold confidence tasks.
2. **Dynamic LoRA Expert Scaling**:
   Utilize the `ConfidenceEstimate` and `RoutingDecision` outputs to route tasks to specialized low-rank adaptation GNN experts tailored to specific topological profiles.
3. **Continuous Online Replay Adaptation**:
   Configure the system to leverage degraded task registers to build experience replay datasets for offline training batches, correcting recurring edge cases.
4. **KiCad and EDA Plugin Suite**:
   Create structural importers using the Oracle SDK reference (`examples/oracle_template.py`) to permit physical PCB schematic netlist extraction directly into the GNN runtime.

---

## 📂 Comprehensive Documentation Base

To acquire the deep technical context required for development, **MUST READ**:
- 🛡️ [Runtime Resilience Deep Dive (v2.13)](docs/V213_RUNTIME_RESILIENCE.md)
- 📖 [Oracle SDK Integration Guide (v2.13)](docs/ORACLE_SDK_GUIDE.md)
- 🔬 [Scientific Report: Virtual Node Projection (v2.9F)](docs/V29F_VIRTUAL_NODE_PROJECTION.md)
- 📊 [Scientific Report: Topological Curriculum (v2.9E)](docs/V29E_TOPOLOGY_AWARE_SURROGATE.md)
