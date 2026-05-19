# CORE Paper Positioning

## Core Contributions

### 1. Adaptive Scheduling Without Correctness Sacrifice

The central claim: adaptive budget allocation and trajectory-aware
scheduling can reduce runtime without increasing residual error.

Evidence:
- Adaptive final_residual <= fixed final_residual * 1.01 (benchmark)
- Adaptive iterations < fixed iterations (average)
- Adaptive runtime_ms < fixed runtime_ms (average)
- Scheduler efficiency ratio: 230.8x (overhead 0.026ms/decision)

### 2. Trajectory Classification for Runtime Optimization

The trajectory analyzer classifies convergence behavior into:
- fast_converging (exponential decay)
- oscillatory (overshooting with net improvement)
- stalled (flat residual)
- divergence_risk (increasing residual)

Each class triggers different scheduling policies:
- fast_converging: reduce budget, no escalation
- oscillatory: increase budget, monitor divergence
- stalled: consider warmstart or escalation
- divergence_risk: immediate escalation to oracle

### 3. Retrieval-Assisted Warmstart

FAISS-based retrieval of similar past solutions provides warmstart
initializations that reduce convergence effort.

Evidence:
- Warmstart iterations < coldstart iterations (statistically significant)
- p < 0.0001 (synthetic validation)
- Initial residual delta consistently negative (warmstart starts closer)

### 4. Operational Experience Dataset

300+ execution traces with full metadata:
- Topology families, convergence classes, iteration distributions
- Warmstart effectiveness, escalation frequencies
- Runtime distributions, degraded execution rates
- Available as JSONL + CSV for reproducibility

### 5. Domain-Agnostic Runtime Framework

The CORE SDK proves that the same scheduling/routing/caching
infrastructure serves multiple domains:
- Circuits: MNA + KCL/KVL projection (first validated domain)
- Linear System: Ax = b with gradient descent projection (canary domain)

Future domains (KiCad, FreeCAD, mathematics, logic, programming)
can plug into the same runtime without modifying core code.

## Benchmark Methodology

1. **Correctness Preservation Benchmark**
   - Compare fixed vs adaptive scheduling on identical task sets
   - Measure: residual, KCL/KVL violations, iterations, runtime
   - Accept: adaptive residual <= 1.01x fixed residual

2. **Oscillatory Convergence Validation**
   - Synthetic trajectories with known classifications
   - Verify: oscillatory != divergence_risk, stalled detection, etc.

3. **Retrieval Effectiveness Validation**
   - Warmstart vs coldstart on identical task sets
   - Statistical significance test on iteration reduction

4. **Scheduler Overhead Validation**
   - Measure scheduler decision time vs projection time saved
   - Accept: efficiency ratio > 5.0x (measured: 230.8x)

## Reproducibility

All experiments are reproducible:
- Deterministic hashing ensures same input = same execution
- Release manifest captures exact environment (Python, numpy, torch)
- Benchmark seed is recorded
- Operational experience exports include full metadata
- Paper figures are generated from CSV source tables
