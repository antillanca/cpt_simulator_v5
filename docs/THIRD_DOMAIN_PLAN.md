# Third Domain Plan: Propositional Logic Verification

## Purpose

Prove CORE is ready for symbolic/non-circuit expansion by planning
a third domain that differs fundamentally from both circuits and
linear systems.

## Recommended Domain: Propositional Logic Verification

### Why This Domain

1. **Symbolic, not numeric** -- circuits and linear systems are both
   continuous/numeric. Propositional logic is discrete/symbolic. This
   exercises a completely different SDK surface.

2. **Oracle is trivial** -- SAT solvers or truth-table enumeration
   provide exact verification. No numerical approximation.

3. **Surrogate is meaningful** -- heuristic assignment (e.g., unit
   propagation, DPLL-style partial assignment) provides an approximate
   solution that projection can refine.

4. **Projection makes sense** -- resolving unsatisfied clauses is a
   constraint projection operation: given an assignment that violates
   some clauses, project onto the satisfying subspace by flipping
   variables.

5. **Well-studied problem** -- enormous literature, clear metrics
   (satisfied clause ratio, assignment flip count).

### OracleProtocol

```python
class LogicOracle:
    """Exact verification via truth-table or SAT solver."""
    
    def compute(self, formula: LogicFormula) -> LogicOracleResult:
        # Returns exact satisfying assignment or UNSAT
        pass
    
    def cost_estimate(self, formula: LogicFormula) -> CostEstimate:
        # Based on variable count and clause density
        pass
```

### Surrogate

A heuristic partial-assignment builder:

1. **Unit propagation** -- assign forced variables
2. **Pure literal elimination** -- assign pure literals
3. **Greedy variable selection** -- pick variable appearing in most
   unsatisfied clauses

This produces an approximate assignment that may violate some clauses.

### Projection

Clause-satisfaction projection:

1. **Identify violated clauses** -- clauses not satisfied by current assignment
2. **Flip highest-impact variable** -- variable whose flip satisfies the most
   violated clauses
3. **Validate** -- check if all clauses are now satisfied
4. **Residual** = violated_clause_ratio (analogous to KCL violation in circuits)

### What Would Be Reused From CORE

- **ExactMatchCache** -- hash the formula, return cached satisfying assignment
- **RetrievalMemory** -- find structurally similar formulas (similar clause
  density, variable count, clause-to-variable ratio)
- **CapabilityRouter** -- route: cache_hit, retrieval_warmstart, standard,
  oracle_escalation, degraded
- **ProjectionScheduler** -- budget allocation, trajectory analysis, warmstart
- **TrajectoryAnalyzer** -- classify: fast_converging (few flips), oscillatory
  (flip back and forth), stalled (no progress), divergence_risk (more violations)
- **DomainTaskBase** -- generic task interface
- **ExecutionTracer** -- full trace of flip decisions and clause satisfaction

### What Must Remain Domain-Specific

- **LogicFormula** -- clause-variable representation (not MNA matrix)
- **Oracle** -- SAT solver or truth-table (not numpy.linalg.solve)
- **Surrogate** -- unit propagation + greedy (not Jacobi preconditioner)
- **Projection** -- variable flipping (not residual correction)
- **Evaluator** -- clause satisfaction check (not KCL/KVL check)
- **Confidence** -- based on clause satisfaction ratio decay

### Implementation Status: COMPLETE (v3.2)

The propositional logic domain has been implemented and verified.

File: `core_runtime/domains/propositional_logic/__init__.py`

#### What Was Implemented

| Component | Lines | Details |
|-----------|-------|---------|
| CNFFormula + Clause + Literal | ~60 | Frozen dataclasses, hashable, serializable |
| PropositionalLogicTask | ~20 | DomainTaskBase subclass, formula in metadata |
| PropositionalLogicOracle | ~40 | Brute-force truth table, O(2^n) |
| PropositionalLogicSurrogate | ~50 | Unit clause shortcut, greedy heuristic |
| PropositionalLogicProjection | ~120 | Unit propagation + pure literal + brute-force fill |
| PropositionalLogicEvaluator | ~30 | Oracle vs projected comparison |
| execute_propositional_logic_pipeline | ~80 | Full pipeline with trace |
| Tests | ~345 | 28 tests in test_propositional_logic_domain.py |
| **Total** | **~745** | |

#### Key Design Decisions

1. **No external solver dependency** — oracle uses brute-force truth table,
   keeping the domain self-contained and deterministic.

2. **Projection uses hybrid approach** — starts from surrogate assignment,
   applies unit propagation + pure literal elimination, then brute-forces
   remaining variables. This mirrors the iterative refinement pattern in
   linear_system and circuits.

3. **Residual = unsatisfied_clause_ratio** — analogous to KCL violation
   in circuits and Ax-b residual in linear_system.

4. **Formula stored as serializable dict** — in task metadata, enabling
   exact cache hashing and retrieval memory embedding.

5. **Pipeline-compatible** — `execute_propositional_logic_pipeline()` returns
   the same structure as linear_system, enabling cross-domain runtime use.

#### Cross-Domain Verification

The domain was verified to:
- Implement all four DomainSDK protocols (Oracle, Surrogate, Projection, Evaluator)
- Run through `execute_propositional_logic_pipeline()` end-to-end
- Produce deterministic results for SAT and UNSAT formulas
- Generate valid execution traces
- Pass 28 domain-specific tests + principle regression tests

#### Success Criterion: MET

The propositional logic domain runs through the full CORE pipeline
(task -> oracle -> surrogate -> projection -> evaluation -> trace).
CORE is proven domain-agnostic for both continuous and symbolic domains.

### Alternative Domains Considered

| Domain | Rejected Because |
|--------|-----------------|
| Algebraic identity verification | Too close to linear_system |
| ODE solving | Too close to circuits (numeric) |
| Graph coloring | Good candidate, but SAT is more fundamental |
| Type checking | Requires language runtime dependency |
