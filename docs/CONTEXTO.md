# Glossary and Design Constraints (v2.13)

This document formalizes the terminology, constraints, and runtime rules of the CPT Simulator v5 environment.

---

## 1. Core Terminology

### Physical Invariants (Anchor Laws)
Mathematical principles that must be strictly satisfied by the simulation output:
- **Kirchhoff's Current Law (KCL)**: Net current divergence at any node must equal zero.
- **Kirchhoff's Voltage Law (KVL)**: Directed sum of potential differences across any closed cycle must equal zero.
- **Power Conservation**: Total supplied power must match total dissipated power.

### Analytical Oracle
The exact Modified Nodal Analysis (MNA) solver (`dc_solver.py`). Due to its $O(N^3)$ complexity, it is used for ground-truth synthesis, test validation, and OOD escalation, rather than primary inference.

### Neural Surrogate
A Physics-Informed Graph Neural Network (`EdgeAwareCircuitGNN`) trained to estimate node voltages, serving as a high-speed pre-conditioner.

### Physics Projection
A deterministic iterative solver (Jacobi/SOR variant) that corrects the surrogate's predictions to enforce KCL/KVL compliance.

### True Global Virtual Node
A mathematical augmentation injected during the Physics Projection phase. It computes the global mean residual and applies a uniform scalar correction to all nodes simultaneously, reducing the effective graph diameter and accelerating convergence.

### Hybrid Warm-Start
The architectural paradigm of leveraging the Neural Surrogate as an optimal pre-conditioner for the Physics Projection solver, minimizing the required iteration count.

### Resilient Core Runtime
The v2.13 execution environment. It wraps the surrogate, projection, and oracle components with deterministic hashing, exact caching, safety policies, dynamic capability routing, and crash-safe atomic persistence.

---

## 2. Hardened Runtime Design Decisions

### 1. Neuro-Symbolic Decoupling
Isolates probabilistic neural estimations from exact mathematical constraints. The GNN provides a near-instantaneous estimate, while the deterministic projection layer and analytical oracle enforce physical invariants.

### 2. Isomorphic Circuit Hashing
Enforces exact match caching (`task_hashing.py`). Equivalent circuit structures (independent of node alphabetical labelling or float representation formats) yield identical SHA-256 keys, bypassing redundant simulation costs.

### 3. Fail-Safe Degradation Handling
Forbids silent runtime failures. Under simulated stress (NaN values, timeout limits, numeric divergence, or surrogate instability), the `RecoveryHandler` catches exceptions, records the exact failure state, and assigns a specific degradation code rather than crashing the execution thread.

### 4. Crash-Safe Memory Operations
To guarantee data store integrity against sudden runtime interruptions or power loss, the persistence layer writes to a temporary file, calls `os.fsync()` to force disk flush, and invokes `os.replace()` for an atomic rename.

### 5. Deterministic Confidence Routing
Execution is directed by the `CapabilityRouter` according to rule-based confidence heuristics (dynamic range, graph size, topological family, residual history) without stochastic operations, establishing predictable and reproducible performance profiles.
