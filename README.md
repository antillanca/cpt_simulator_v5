# CPT Simulator v2.9F

A hybrid iterative solver for DC electrical circuit simulation. This project combines a Physics-Informed Graph Neural Network (PINN-GNN) with a deterministic Jacobi-style projection layer to achieve analytical accuracy while bypassing traditional $O(N^3)$ Modified Nodal Analysis (MNA) computational costs.

---

## 🚀 System Architecture

### 1. Hybrid Solver Pipeline
The architecture employs a two-stage resolution pipeline:

1.  **GNN Surrogate (`EdgeAwareCircuitGNN`):** 
    Predicts initial node voltages from circuit graph representations. Implements dynamic topological feature extraction and logarithmic resistance normalization to prevent gradient explosion in Out-Of-Distribution (OOD) resistance ranges.
2.  **Physics Projection:** 
    A deterministic iterative layer (Jacobi/SOR variant) that corrects the surrogate's predictions to enforce Kirchhoff's Current Law (KCL) and Kirchhoff's Voltage Law (KVL).

### 2. True Global Virtual Node
To overcome the spectral radius bottleneck inherent in purely local iterative solvers (such as Jacobi on long radial chains), the projection layer injects a virtual mathematical node. This node aggregates the global residual error and redistributes it simultaneously, reducing the effective graph communication diameter to 1 and enabling rapid global convergence.

---

## 🧠 Topological Curriculum and Failure Taxonomy

Training relies on structural progression rather than random sampling. Circuit graphs are evaluated and scheduled via `topology_curriculum.py`:
*   **Trivial:** Trees, $\le 4$ nodes, 0 cycles.
*   **Simple:** 1 independent cycle, $\le 6$ nodes.
*   **Medium:** 2-3 cycles, $\le 10$ nodes.
*   **Dense:** $> 3$ cycles, $> 10$ nodes (high interconnectivity acts as a natural regularizer, yielding lower baseline MAE).

### Structural Failure Taxonomy
Anomalies that survive projection are classified by their topological root cause:
*   `cycle_drift_failure`: KCL divergence within closed loops.
*   `dense_mesh_leakage`: Signal attenuation in highly connected meshes.
*   `bridge_node_instability`: Propagation drift across tree bottlenecks.

---

## 📂 Workspace Structure (v2.9F)

```
cpt_simulator_v5/
├── docs/
│   ├── AGENT_HANDOVER_V29F_COMPREHENSIVE.md  ← Detailed AI handover context
│   └── V29F_VIRTUAL_NODE_PROJECTION.md       ← Scientific report on Virtual Node
├── backend/
│   ├── circuits/
│   │   ├── dc_solver.py                ← Baseline analytical MNA oracle
│   │   ├── graph_dataset.py            ← Graph conversion and log-normalization
│   │   ├── physics_projection.py       ← Iterative layer with Virtual Node
│   │   ├── topology_curriculum.py      ← Structural complexity scheduler
│   │   ├── failure_analysis.py         ← Topological failure classification
│   │   ├── warmstart_eval.py           ← Warm-start efficiency experiment
│   │   └── ood_stress_suite.py         ← Deterministic OOD topology generators
├── scripts/
│   ├── train_circuit_gnn.py            ← Training pipeline with physics loss
│   └── run_circuit_arena.py            ← Topological benchmark and evaluation
└── tests/
    ├── test_v29f_virtual_projection.py ← Residual monotonic reduction tests
    └── test_v29f_warmstart.py          ← Warm-start solver reduction tests
```

---

## 🎯 Current Performance (v2.9F)

| Configuration | In-Dist MAE | KCL Max (A) | OOD KCL Max (A) | Solver Iterations |
| :--- | :---: | :---: | :---: | :---: |
| **Pure GNN (Baseline)** | 15.44 V | 0.275 | 4.82 | High (Cold Start) |
| **GNN + Curriculum** | 14.16 V | 0.163 | 1.10 | - |
| **Hybrid (GNN + Virtual Node)**| **~0.0 V** | **$< 1e-6$** | **$< 1e-6$** | **Significantly Reduced** |

---

## 🛠️ Usage

Execute the physical validation and determinism test suite:
```bash
pytest tests/test_v29e_*.py tests/test_v29f_*.py -v
```

Run the warm-start scientific evaluation (iteration count reduction):
```bash
python -m backend.circuits.warmstart_eval --steps 5 --perturbation 1.5
```

Execute the Circuit Arena benchmark (segregated by topological families):
```bash
python scripts/run_circuit_arena.py
```
