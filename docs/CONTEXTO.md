# Glossary and Design Constraints (v2.9F)

This document formalizes the terminology and design constraints underlying the CPT Simulator v5 ecosystem.

---

## 1. Core Terminology

### Physical Invariants (Anchor Laws)
Mathematical principles that must be strictly satisfied by the final simulation output. The primary invariants are:
- **Kirchhoff's Current Law (KCL)**: Net current divergence at any node must equal zero.
- **Kirchhoff's Voltage Law (KVL)**: Directed sum of potential differences across any closed cycle must equal zero.
- **Power Conservation**: Total supplied power must precisely match total dissipated power.

### Analytical Oracle
The exact Modified Nodal Analysis (MNA) solver (`dc_solver.py`). Due to its $O(N^3)$ complexity, it is utilized exclusively for generating training targets and evaluating empirical error, rather than inference.

### Neural Surrogate
A Physics-Informed Graph Neural Network (`EdgeAwareCircuitGNN`) trained to estimate node voltages. Due to its probabilistic nature, raw predictions generally exhibit non-zero residual violations of the physical invariants.

### Physics Projection
A deterministic iterative matrix solver (Jacobi/SOR variant) that corrects the surrogate's predictions, projecting them onto the physical invariant manifold to enforce strict KCL/KVL compliance.

### True Global Virtual Node
A mathematical augmentation injected exclusively during the Physics Projection phase. It computes the global mean residual and applies a uniform scalar correction to all nodes simultaneously, reducing the effective graph diameter and ensuring rapid global convergence across adverse topologies (e.g., radial chains).

### Hybrid Warm-Start
The v2.9F architectural paradigm. It leverages the Neural Surrogate not as the final output generator, but as an optimal pre-conditioner for the Physics Projection solver, drastically reducing the required iteration count.

---

## 2. Architectural Design Decisions

### 1. Neuro-Symbolic Decoupling
The architecture strictly isolates neural estimation from mathematical verification. The neural network provides a fast, approximate initialization, while the deterministic projection layer enforces absolute physical constraints.

### 2. Topological Curriculum
Training is governed by an explicit mathematical progression (`topology_curriculum.py`) rather than stochastic sampling. The model must master trivial tree structures before progressing to high-density meshes, ensuring stable gradient convergence.

### 3. Logarithmic Feature Normalization
To prevent gradient overflow when evaluating Out-Of-Distribution (OOD) scenarios involving extreme resistance magnitudes ($0.1\Omega$ to $1M\Omega$), all structural resistance features undergo strict logarithmic normalization before entering the computational graph.

### 4. Root-Cause Topological Diagnosis
Evaluation extends beyond aggregated Mean Squared Error (MSE). The `failure_analysis.py` module classifies non-convergent instances by their underlying structural topology (e.g., `cycle_drift_failure`, `dense_mesh_leakage`), directing targeted dataset generation.
