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

### Implementation Scope (v3.2+)

This is a plan only. No implementation in v3.1. Estimated effort:

- Oracle: 50 lines (truth-table for small formulas, z3 for large)
- Surrogate: 80 lines
- Projection: 60 lines
- Evaluator: 30 lines
- Confidence: 20 lines
- Domain adapters: 40 lines
- Tests: 200 lines
- **Total: ~480 lines**

### Alternative Domains Considered

| Domain | Rejected Because |
|--------|-----------------|
| Algebraic identity verification | Too close to linear_system |
| ODE solving | Too close to circuits (numeric) |
| Graph coloring | Good candidate, but SAT is more fundamental |
| Type checking | Requires language runtime dependency |

### Success Criterion

If a propositional logic domain runs through the full CORE pipeline
(task -> oracle -> surrogate -> projection -> memory -> trace), then
CORE is proven domain-agnostic for both continuous and symbolic domains.
