# Curriculum Status — CPT Simulator v5

> **Executive Summary**: The CPT Simulator v5 curriculum operates in two distinct phases. Phase 1 (Symbolic Domain Acquisition) is 100% complete. The system is currently executing Phase 2 (Topological and Structural Physics), focusing on the hybrid resolution of complex electrical graphs.

---

## 🎓 Phase 1: Symbolic Domain Curriculum (v2.5) — ✅ COMPLETED

This phase involved the acquisition of fundamental mathematical and physical principles via generated, deterministically verified Lua logic.

| Metric | Value | Status |
|:---|:---:|:---:|
| Total theoretical modules | **43** | ✅ Assimilated |
| Invariant verifications | **43** (100%) | ✅ Confirmed |
| Pending modules | **0** (0%) | - |

### Acquired Domains
- **Classical Mathematics**: Arithmetic, Algebra, Euclidean Geometry, Trigonometry, Linear Algebra, Numerical Calculus.
- **Classical Physics**: Kinematics, Newtonian Dynamics, Oscillators, Energy Conservation.
- **Electromagnetism**: Ohm's Law, Lorentz Force, Maxwell's Equations.
- **Modern Physics**: Relativity (Special/General), Quantum Mechanics (Wavefunction, Double Slit), QFT.
- **Systems Analysis**: Chaos Theory, Thermodynamics, Entropy.

*Note: All foundational modules reside in `backend/core_truth/` and were strictly validated by the analytical engine without probabilistic LLM interference.*

---

## ⚡ Phase 2: Topological Graph Curriculum (v2.9F) — 🔄 ACTIVE

Having theoretically assimilated circuit laws, the system currently trains a **Graph Neural Network (GNN)** coupled with a **Physics Projection** layer to resolve arbitrary circuit graphs in real-time.

To prevent training collapse, data is structured via a rigorous `CurriculumLevel` topological scheduler (`topology_curriculum.py`).

### Topological Progression Status

| Level | Definition | Structural Parameters | GNN Pre-conditioning Status | Hybrid Solver Behavior |
|:---:|:---|:---|:---:|:---|
| **L0** | **Trivial** | Tree structures, $\le 4$ nodes, $0$ cycles. | ✅ **Mastered** | Instantaneous convergence. |
| **L1** | **Simple** | 1 independent cycle, $\le 6$ nodes. | ✅ **Mastered** | High precision initial estimation. |
| **L2** | **Medium** | 2-3 cycles, $\le 10$ nodes. | 🟡 **Advanced** | Stable; GNN acts as an optimal warm-start. |
| **L3** | **Dense** | $>3$ cycles, $>10$ nodes (complex meshes). | 🟢 **Excellent** | High interconnectivity enforces dense physical constraints, regularizing the network to exceptionally low MAE. |
| **L4** | **Extreme (OOD)**| Radial chains ($+50$ nodes), $1M\Omega$ resistors. | 🟡 **Active** | The **True Global Virtual Node** (v2.9F) is critical here to ensure mathematical convergence by mitigating spectral radius decay. |

### Phase 2 (v2.9F) Milestones
- [x] Paradigm shift from pure regression to **Hybrid Iterative Solver**.
- [x] Implementation of deterministic post-GNN Physics Projection.
- [x] Integration of the **Virtual Node** to reduce the communication diameter in high-length radial networks.
- [x] Implementation of automated Structural Failure Taxonomy diagnostics.

---

## 🔮 Phase 3: Autonomous Active Learning (Roadmap)

The next evolutionary phase involves delegating topological curriculum management to the supervisory autonomous agent.
- The agent will continuously monitor structural metrics via the Circuit Arena.
- It will identify topologically weak families (e.g., Extreme Ladder Networks).
- It will autonomously synthesize and inject targeted synthetic batches into the training pipeline, closing the Active Learning loop.
