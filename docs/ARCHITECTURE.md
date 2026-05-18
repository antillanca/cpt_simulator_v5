# System Architecture (v2.9F)

This document outlines the core architecture of the CPT Simulator v5, a hybrid iterative solver combining Physics-Informed Graph Neural Networks (PINN-GNN) with deterministic analytical projection.

---

## 1. Hybrid Solver Stack

The system resolves DC circuits by cascading neural pre-conditioning with mathematical projection.

### 1.1 Ground Truth Oracle (`backend/circuits/dc_solver.py`)
- **Role**: Reference analytical solver.
- **Mechanism**: Implements exact Modified Nodal Analysis (MNA).
- **Complexity**: $O(N^3)$. Used strictly for dataset generation and final validation, circumventing latency constraints during inference.

### 1.2 GNN Surrogate Pre-conditioner (`backend/neural/models/circuit_gnn.py`)
- **Role**: High-speed initial state estimator.
- **Architecture**: `EdgeAwareCircuitGNN` optimized via physics-informed loss (KCL/KVL/Power).
- **Features**: Utilizes topological feature extraction and logarithmic resistance normalization, ensuring stability across Out-Of-Distribution (OOD) resistance ranges ($0.1\Omega$ to $1M\Omega$).

### 1.3 Deterministic Physics Projection (`backend/circuits/physics_projection.py`)
- **Role**: Iterative Jacobi-style corrector.
- **Mechanism**: Computes exact residuals for Kirchhoff's laws and applies corrective updates to the surrogate's output until convergence limits are met.
- **Key Innovation (Virtual Node)**: Integrates a mathematical global node that aggregates the mean system residual and redistributes it universally. This structural modification reduces the effective communication diameter of any graph topology to 1, neutralizing the spectral radius degradation typical of local iterative solvers on high-diameter graphs.

---

## 2. Topological Curriculum (`topology_curriculum.py`)

Training data is curated through a deterministic difficulty scheduler rather than uniform random sampling.

- **Trivial**: Tree structures (0 independent cycles, $\le 4$ nodes).
- **Simple**: Single-loop circuits (1 cycle, $\le 6$ nodes).
- **Medium**: Moderately coupled loops (2-3 cycles, $\le 10$ nodes).
- **Dense**: Highly interconnected meshes ($> 3$ cycles). *Note: The dense interconnectivity acts as a natural graph regularizer, often yielding lower Mean Absolute Error (MAE) compared to sparse topologies.*

---

## 3. Structural Failure Taxonomy (`failure_analysis.py`)

Anomalies are diagnosed topologically to guide structural improvements:

- `cycle_drift_failure`: Non-convergent KCL residuals within closed loops.
- `dense_mesh_leakage`: Signal attenuation across high-degree nodes.
- `bridge_node_instability`: Convergence drift across critical bottlenecks in tree structures.

---

## 4. Development Roadmap

1.  **Differentiable Physics Layers**: Transition the post-processing Newton/Jacobi iterations into the computational graph during training, allowing the network to backpropagate through the physical correction steps.
2.  **GNN-level Virtual Nodes**: Implement the global virtual node directly within the message-passing phase of the GNN to counteract signal attenuation in extreme ladder topologies ($>100$ stages).
3.  **Active Learning Loop**: Couple the topological failure taxonomy output with the dataset generator to autonomously synthesize and over-sample weak topological configurations.
