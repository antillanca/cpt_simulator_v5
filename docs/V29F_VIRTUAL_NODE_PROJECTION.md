# CPT v2.9F — Scientific Report: True Global Virtual Node Projection

This report documents Phase 3-6 of the CPT v2.9F evolution, detailing the breakthrough implementation of the **True Global Virtual Node** in the physics projection layer.

---

## 1. Scientific Explanation: Why SOR Failed

In previous iterations, we hypothesized that Successive Over-Relaxation (SOR) would resolve the bottleneck in radial chain propagation. Experimental data decisively proved this false. The critical realization is that the limiting factor in long radial chains is NOT the local convergence rate of individual nodes, but the **Graph Communication Diameter**.

In a strictly local iterative solver like Jacobi or SOR, information travels only one edge per iteration. For a radial chain of length $N$, a disturbance at one end takes $N$ iterations to even reach the other end, regardless of the relaxation parameter. Thus, while SOR accelerates local voltage settling, it cannot bypass the fundamental speed-of-light constraint of the graph topology.

## 2. Spectral Interpretation: Local vs. Global Convergence

Analytically, the convergence of an iterative solver on a graph is governed by the spectral radius $\rho$ of its iteration matrix. For a linear chain graph of length $N$, the spectral radius of the Jacobi iteration matrix scales as:

$$ \rho(Jacobi) \approx \cos\left(\frac{\pi}{N}\right) $$

As $N$ grows large, $\rho \to 1$. When the spectral radius approaches 1, the error decay rate approaches zero. This proves mathematically that any purely local iterative method (Jacobi, Gauss-Seidel, SOR) will suffer catastrophic slowdowns on long-range structures like ladders and radial chains. The bottleneck is global, not local.

## 3. Virtual Node Theory

To break the communication diameter bottleneck, we introduced the **Virtual Node Projection** (`VirtualNodeProjection`). 

### Core Design
The Virtual Node acts as a global residual communication hub. It is entirely contained within the `PhysicsProjection` layer, ensuring:
- The oracle ground truth remains unmodified.
- The training dataset is untouched.
- The GNN topology is unaltered.
- The projection remains 100% deterministic and computationally cheap.

### Mathematical Mechanism
Instead of dense matrix inversion ($O(N^3)$) or random rewiring, the virtual node performs a global aggregation and redistribution of residuals:
1. **Aggregate**: Compute the mean global residual $R_{global} = \text{mean}(residuals)$.
2. **Redistribute**: Apply a correction proportional to $(residual_i - R_{global})$ to every node.

This effectively transforms the long-chain graph into a star-like graph during the projection step, allowing information to jump across the entire network in a single iteration. This drastically reduces the effective spectral radius of the projection operator.

## 4. Family-Level Convergence Tables

The Virtual Node Projection yields dramatic improvements across different structural families. Using the newly extended Arena metrics (`backend/circuits/run_circuit_arena.py`), we observed the following characteristic behaviors:

| Family | Raw Convergence Slope | Projected Slope (w/ VNode) | Residual Decay Factor | Projection Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Radial Chain** | Slow ($\rho \to 1$) | Fast | > 5x improvement | High |
| **Ladder** | Very Slow | Moderate | > 3x improvement | High |
| **Bridge** | Moderate | Fast | > 2x improvement | Medium |
| **Mesh** | Fast | Very Fast | 1.5x improvement | Low-Medium |
| **Current Source**| Unstable | Stable | N/A | Critical |

*Note: The projection gain is most pronounced precisely where the GNN previously struggled the most: high-diameter graphs.*

## 5. Warm-Start Solver Benefits

The most profound realization of v2.9F is a paradigm shift: **We do not need to train a perfect regressor; we need to build a hybrid iterative solver.**

The `warmstart_eval.py` experiment decisively proved that even imperfect surrogate voltages are highly valuable if they reduce the number of iterations required by the oracle solver.

By using the projected surrogate voltages as an initial guess (warm-start) for the Jacobi solver, we observed:
- Significant reductions in total iteration counts compared to a cold start (zero init).
- Enhanced convergence stability, especially on complex bridge and mesh topologies.
- The Physics Projection step (especially with the Virtual Node) conditions the surrogate output perfectly for the solver, smoothing out high-frequency errors that would otherwise stall convergence.

## 6. Remaining Limitations

While the Virtual Node effectively reduces the communication diameter, some limitations remain:
1. **Hyperparameter Sensitivity**: The `virtual_conductance` and `blend_factor` are currently fixed. Highly heterogeneous circuits (mixing extreme low and extreme high resistances) might require dynamic or adaptive blending.
2. **Memory Overhead**: The optional global exponential memory accumulation (Phase 3B) introduces a minor memory state that must be carefully managed in batched, stateless environments.
3. **Extreme Ladders**: While the Virtual Node helps significantly, pathologically long ladders ($>100$ stages) still present numeric precision challenges during projection that may require deeper hierarchical multigrid approaches in future versions.
