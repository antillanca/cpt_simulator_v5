# CPT Simulator v2.9F — Executive Handover

> **Current Status (May 2026)**: 🚀 **MILESTONE ACHIEVED: v2.9F True Global Virtual Node Projection.**
> The system has transitioned from an isolated neural regressor to a fully integrated **Hybrid Neuro-Symbolic Iterative Solver**.

---

## 🏗️ Architecture: The Hybrid Neuro-Symbolic Stack

The CPT Simulator is a layered intelligence system that fuses the sub-millisecond inference speed of Graph Neural Networks (GNNs) with the analytical accuracy of traditional mathematical solvers.

### 1. Layer 0: Core Truth (Analytical Oracle)
- **Path**: `backend/circuits/dc_solver.py`
- **Logic**: A reference solver utilizing exact Modified Nodal Analysis (MNA). Due to its $O(N^3)$ computational cost, it is employed strictly for ground-truth dataset generation and final error validation, bypassing inference latency.

### 2. Layer 1: GNN Surrogate (Pre-Conditioner)
- **Path**: `scripts/train_circuit_gnn.py` & `backend/neural/models/circuit_gnn.py`
- **Logic**: A Physics-Informed Graph Neural Network (PINN). In the v2.9F paradigm, this network acts as a highly optimized **Warm-Start** pre-conditioner rather than a standalone solver, estimating initial node voltages near-instantaneously.

### 3. Layer 2: Physics Projection (Deterministic Corrector)
- **Path**: `backend/circuits/physics_projection.py`
- **Logic**: A Jacobi-style iterative matrix solver that projects the surrogate's output onto the exact mathematical manifold required by Kirchhoff's laws (KCL/KVL).
- **v2.9F Innovation**: Implements the **True Global Virtual Node**, an aggregated mathematical construct that redistributes global residual error simultaneously. This structural modification circumvents spectral radius limitations, ensuring rapid convergence across high-diameter graphs (e.g., extensive radial chains).

---

## 📈 Progression & Metrics (v2.9F)

- **Topological Curriculum:** Circuits are injected into the training pipeline according to a rigorous structural difficulty scheduler (Trivial, Simple, Medium, Dense).
- **Failure Taxonomy:** The analytical engine diagnoses physical anomalies based on topological root causes (e.g., `cycle_drift_failure`, `dense_mesh_leakage`, `bridge_node_instability`).
- **OOD Stress Suite:** Deterministic generators deploy massive, highly interconnected meshes and extreme ladder networks to empirically evaluate the network's boundary limits (`ood_stress_suite.py`).

| Metric | Hybrid Pipeline (Projection + Virtual Node) |
|:---|:---:|
| **In-Dist MAE** | ~0.0 V |
| **KCL Max Residual (A)** | $< 1e-6$ |
| **Solver Iterations** | Minimal |

---

## 🔮 Roadmap: Immediate Objectives

Incoming agents and engineers should prioritize the following developmental vectors:

1. **Differentiable Newton-Physics Loss**: 
   Integrate the self-correcting physical projection heads directly into the residual layers of the GNN during training, enforcing analytical KCL compliance explicitly within the forward pass.
2. **Temporal Receptive Field Scaling**:
   Investigate the inclusion of global virtual nodes at the GNN message-passing level to combat signal attenuation in extreme, high-stage ladder networks ($>100$ stages).
3. **Autonomous Active Learning**:
   Configure the supervisory agent to continuously parse Circuit Arena topological metrics, automatically orchestrating targeted re-training sessions for structurally weak circuit families.

---

## 📂 Comprehensive Documentation Base

To acquire the complete, deep technical context required for development, **MUST READ**:
- 📖 [Comprehensive AI Handover Guide (v2.9F)](docs/AGENT_HANDOVER_V29F_COMPREHENSIVE.md)
- 🔬 [Official Scientific Report: Virtual Node Projection (v2.9F)](docs/V29F_VIRTUAL_NODE_PROJECTION.md)
- 📊 [Scientific Report: Topological Ablation (v2.9E)](docs/V29E_TOPOLOGY_AWARE_SURROGATE.md)
