# CPT Simulator v2.9F — Master Handover Document

> **Purpose**: Transfer comprehensive project context to any incoming developer or AI agent. This document outlines the system architecture, design rationale, current state, and immediate next steps for the CPT Simulator v5.

---

## 🎯 Project Scope

The CPT Simulator is a hybrid iterative solver designed for DC electrical circuit simulation. It implements a Physics-Informed Graph Neural Network (PINN-GNN) as a fast pre-conditioner (warm-start) for a deterministic analytical solver, effectively bypassing the $O(N^3)$ computational cost of traditional Modified Nodal Analysis (MNA) while guaranteeing strict adherence to physical conservation laws.

### The Problem

Traditional iterative solvers (such as Jacobi or Successive Over-Relaxation) suffer from catastrophic slowdowns on high-diameter graphs (e.g., long radial chains or extensive ladder networks) because their spectral radius approaches 1. Conversely, pure neural network regressors cannot guarantee strict physical invariants (KCL/KVL) without extensive and often unstable penalty tuning.

### The Solution (v2.9F Paradigm)

The v2.9F architecture resolves this dichotomy through a hybrid approach:
1. **Neural Pre-Conditioning**: The GNN predicts a highly accurate initial voltage state in sub-millisecond time.
2. **Deterministic Physics Projection**: An iterative correction layer forces the neural prediction to comply with Kirchhoff's laws.
3. **True Global Virtual Node**: A mathematical construct injected during projection that simultaneously aggregates and redistributes global residual error, reducing the effective graph communication diameter to 1 and ensuring rapid convergence regardless of the underlying topology.

---

## 🏗️ Architecture Stack

The system implements a strict, multi-stage resolution pipeline:

### 1. Ground Truth Oracle (`backend/circuits/dc_solver.py`)
- **Role**: The absolute baseline for correctness. It uses exact analytical MNA to solve circuits.
- **Usage**: Used exclusively during dataset generation and final validation, not during inference.

### 2. GNN Surrogate (`backend/neural/models/circuit_gnn.py`)
- **Role**: An `EdgeAwareCircuitGNN` trained under PINN constraints.
- **Features**: Processes dynamic topological features and applies logarithmic normalization to resistance values to ensure numerical stability and prevent gradient explosion in Out-Of-Distribution (OOD) scenarios.

### 3. Physics Projection Layer (`backend/circuits/physics_projection.py`)
- **Role**: A deterministic, Jacobi-style iterative corrector.
- **Mechanism**: Takes the surrogate's output and iteratively minimizes KCL/KVL residuals.
- **Virtual Node Integration**: Employs the `VirtualNodeProjection` to guarantee uniform global convergence.

---

## 🧠 Topological Curriculum and Diagnostics

### Structural Progression (`topology_curriculum.py`)
Training data is not uniformly sampled. The system follows a deterministic topological curriculum to prevent learning collapse:
- **Trivial**: Tree structures, $\le 4$ nodes, 0 independent cycles.
- **Simple**: 1 independent cycle, $\le 6$ nodes.
- **Medium**: 2-3 cycles, $\le 10$ nodes.
- **Dense**: $> 3$ cycles, $> 10$ nodes.

### Failure Taxonomy (`failure_analysis.py`)
Residual errors are structurally classified rather than simply aggregated as Mean Squared Error (MSE):
- `cycle_drift_failure`: KCL violations localized within closed loops.
- `dense_mesh_leakage`: Signal attenuation across highly interconnected nodes.
- `bridge_node_instability`: Iterative divergence across critical path bottlenecks.

---

## 📊 Current State (May 2026)

The project has successfully transitioned from Phase 1 (Symbolic domain knowledge acquisition) to Phase 2 (Structural topological mastery).

| Metric | Baseline GNN | Hybrid Pipeline (GNN + Virtual Node) |
| :--- | :---: | :---: |
| **In-Dist MAE** | ~15.44 V | **~0.0 V** |
| **KCL Max Residual** | ~0.275 A | **$< 1e-6$ A** |
| **Solver Iterations** | High | **Minimal** |

---

## 🗺️ Roadmap & Immediate Next Steps

Incoming agents and developers should focus on the following core challenges:

1. **Differentiable Newton-Physics Loss**:
   Embed the physics correction heads directly into the GNN's residual layers during the forward pass. This forces the network to learn exact analytical derivatives during training, moving the correction from a post-processing step to an intrinsic neural property.

2. **Temporal Receptive Field Scaling**:
   Investigate the injection of global virtual nodes *within the GNN message-passing phase* (not just the projection layer) to combat signal attenuation in extreme ladder networks ($>100$ stages).

3. **Autonomous Active Learning via Hermes**:
   Integrate the topological failure taxonomy output with the dataset generator. The Hermes agent should monitor the `run_circuit_arena.py` metrics and automatically orchestrate targeted training sessions for the weakest topological families.

---

## 📂 Critical File Map

```
cpt_simulator_v5/
├── backend/
│   ├── circuits/
│   │   ├── dc_solver.py                ← Analytical oracle
│   │   ├── graph_dataset.py            ← Feature engineering & log-normalization
│   │   ├── physics_projection.py       ← Virtual Node Projection layer
│   │   ├── topology_curriculum.py      ← Difficulty scheduler
│   │   ├── failure_analysis.py         ← Topological error classifier
│   │   └── warmstart_eval.py           ← Hybrid solver evaluation
│   └── neural/
│       └── models/
│           └── circuit_gnn.py          ← PINN-GNN Architecture
│
├── scripts/
│   ├── train_circuit_gnn.py            ← Training orchestrator
│   └── run_circuit_arena.py            ← Scientific benchmarking suite
│
├── docs/
│   ├── AGENT_HANDOVER_V29F_COMPREHENSIVE.md  ← Extended context for AI
│   └── V29F_VIRTUAL_NODE_PROJECTION.md       ← Official scientific report
│
└── tests/
    ├── test_v29f_virtual_projection.py ← Projection stability verification
    └── test_v29f_warmstart.py          ← Warm-start reduction tests
```

---

## ⚙️ Operational Commands

**Execute validation and stability test suites:**
```bash
pytest tests/test_v29e_*.py tests/test_v29f_*.py -v
```

**Run the warm-start scientific evaluation:**
```bash
python -m backend.circuits.warmstart_eval --steps 5 --perturbation 1.5
```

**Execute the segregated Circuit Arena benchmark:**
```bash
python scripts/run_circuit_arena.py
```
