"""CORE v3.2 — Propositional Logic Domain.

A small symbolic domain proving CORE is not limited to numerical domains.

Oracle: brute-force truth table evaluation
Surrogate: deterministic unit-clause shortcut
Projection: iterative simplification pass
Evaluator: compare oracle vs projected results

All operations are deterministic and require no GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core_runtime.core.domain_sdk import (
    DomainEvaluator,
    DomainOracle,
    DomainProjection,
    DomainSurrogate,
    DomainTaskBase,
)


# ---------------------------------------------------------------------------
# Formula representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Literal:
    """A proposition or its negation."""
    prop: str
    negated: bool = False

    def evaluate(self, assignment: dict[str, bool]) -> bool:
        val = assignment.get(self.prop, False)
        return not val if self.negated else val

    def __str__(self) -> str:
        return f"~{self.prop}" if self.negated else self.prop


@dataclass(frozen=True)
class Clause:
    """A disjunction of literals (CNF clause)."""
    literals: tuple[Literal, ...]

    def evaluate(self, assignment: dict[str, bool]) -> bool:
        return any(lit.evaluate(assignment) for lit in self.literals)

    def __str__(self) -> str:
        return " | ".join(str(l) for l in self.literals)


@dataclass(frozen=True)
class CNFFormula:
    """A conjunction of clauses in CNF."""
    clauses: tuple[Clause, ...]
    variables: tuple[str, ...]

    def evaluate(self, assignment: dict[str, bool]) -> bool:
        return all(cl.evaluate(assignment) for cl in self.clauses)

    def count_satisfying(self) -> int:
        """Count satisfying assignments by brute force."""
        n = len(self.variables)
        count = 0
        for i in range(2**n):
            assignment = {}
            for j, var in enumerate(self.variables):
                assignment[var] = bool((i >> j) & 1)
            if self.evaluate(assignment):
                count += 1
        return count

    def find_satisfying(self) -> dict[str, bool] | None:
        """Find one satisfying assignment by brute force."""
        n = len(self.variables)
        for i in range(2**n):
            assignment = {}
            for j, var in enumerate(self.variables):
                assignment[var] = bool((i >> j) & 1)
            if self.evaluate(assignment):
                return assignment
        return None


# ---------------------------------------------------------------------------
# Domain Task — must be frozen because DomainTaskBase is frozen
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PropositionalLogicTask(DomainTaskBase):
    """A propositional logic satisfiability task.

    Stores the formula in metadata as a serializable dict.
    The formula_dict has shape: {"clauses": [[{"prop":str,"negated":bool},...],...],
                                 "variables": [str,...]}
    """
    task_id: str = ""
    domain_name: str = "propositional_logic"
    input_artifact: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def formula(self) -> CNFFormula | None:
        """Reconstruct CNFFormula from metadata."""
        fd = self.metadata.get("formula_dict")
        if fd is None:
            return None
        return _formula_from_dict(fd)


def _formula_to_dict(f: CNFFormula) -> dict[str, Any]:
    """Serialize a CNFFormula to a JSON-friendly dict."""
    clauses = []
    for cl in f.clauses:
        lits = [{"prop": l.prop, "negated": l.negated} for l in cl.literals]
        clauses.append(lits)
    return {"clauses": clauses, "variables": list(f.variables)}


def _formula_from_dict(d: dict[str, Any]) -> CNFFormula:
    """Deserialize a dict back to a CNFFormula."""
    clauses = []
    for cl_data in d["clauses"]:
        lits = tuple(Literal(prop=l["prop"], negated=l["negated"]) for l in cl_data)
        clauses.append(Clause(lits))
    return CNFFormula(
        clauses=tuple(clauses),
        variables=tuple(d["variables"]),
    )


def make_task(
    task_id: str,
    formula: CNFFormula,
) -> PropositionalLogicTask:
    """Construct a PropositionalLogicTask from a CNFFormula."""
    return PropositionalLogicTask(
        task_id=task_id,
        domain_name="propositional_logic",
        input_artifact=task_id,
        metadata={"formula_dict": _formula_to_dict(formula)},
    )


# ---------------------------------------------------------------------------
# Oracle: brute-force truth table
# ---------------------------------------------------------------------------

class PropositionalLogicOracle(DomainOracle):
    """Brute-force SAT oracle via exhaustive truth table enumeration."""

    def solve(self, task: DomainTaskBase) -> dict[str, Any]:
        t_start = time.perf_counter()
        pl_task = task  # type: ignore[assignment]
        formula = pl_task.formula if isinstance(pl_task, PropositionalLogicTask) else None

        if formula is None:
            fd = task.metadata.get("formula_dict") if hasattr(task, "metadata") else None
            formula = _formula_from_dict(fd) if fd else None

        if formula is None:
            return {
                "satisfiable": False,
                "assignment": None,
                "satisfying_count": 0,
                "total_assignments": 0,
                "residual": float("inf"),
                "runtime_ms": 0.0,
                "method": "brute_force_truth_table",
            }

        satisfying = formula.find_satisfying()
        count = formula.count_satisfying()
        n = len(formula.variables)

        runtime_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "satisfiable": satisfying is not None,
            "assignment": satisfying,
            "satisfying_count": count,
            "total_assignments": 2**n,
            "residual": 0.0,
            "runtime_ms": runtime_ms,
            "method": "brute_force_truth_table",
        }


# ---------------------------------------------------------------------------
# Surrogate: unit-clause shortcut
# ---------------------------------------------------------------------------

class PropositionalLogicSurrogate(DomainSurrogate):
    """Deterministic unit-clause shortcut surrogate.

    Identifies unit clauses (single-literal clauses) and propagates.
    Falls back to all-True heuristic for remaining variables.
    """

    def predict(self, task: DomainTaskBase) -> dict[str, Any]:
        t_start = time.perf_counter()
        pl_task = task
        formula = pl_task.formula if isinstance(pl_task, PropositionalLogicTask) else None

        if hasattr(task, "metadata") and formula is None:
            fd = task.metadata.get("formula_dict")
            formula = _formula_from_dict(fd) if fd else None

        if formula is None:
            return {
                "prediction": {},
                "satisfiable_estimate": False,
                "residual": float("inf"),
                "runtime_ms": 0.0,
                "method": "unit_clause_shortcut",
            }

        # Unit propagation
        assignment: dict[str, bool] = {}
        for clause in formula.clauses:
            if len(clause.literals) == 1:
                lit = clause.literals[0]
                assignment[lit.prop] = not lit.negated

        # Fill remaining variables with True
        for var in formula.variables:
            if var not in assignment:
                assignment[var] = True

        # Check satisfaction
        satisfied = formula.evaluate(assignment)
        unsat = sum(1 for cl in formula.clauses if not cl.evaluate(assignment))
        residual = unsat / max(1, len(formula.clauses))

        runtime_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "prediction": assignment,
            "satisfiable_estimate": satisfied,
            "residual": residual,
            "runtime_ms": runtime_ms,
            "method": "unit_clause_shortcut",
        }


# ---------------------------------------------------------------------------
# Projection: iterative simplification
# ---------------------------------------------------------------------------

class PropositionalLogicProjection(DomainProjection):
    """Iterative simplification projection.

    Repeatedly applies unit propagation and pure literal elimination
    until no further simplification is possible. Then falls back to
    brute-force search over remaining variables.
    """

    def __init__(self, max_iterations: int = 100) -> None:
        self._max_iterations = max_iterations

    def project(
        self,
        task: DomainTaskBase,
        prediction: dict[str, Any],
        budget: int = 100,
    ) -> dict[str, Any]:
        t_start = time.perf_counter()
        pl_task = task
        formula = pl_task.formula if isinstance(pl_task, PropositionalLogicTask) else None

        if hasattr(task, "metadata") and formula is None:
            fd = task.metadata.get("formula_dict")
            formula = _formula_from_dict(fd) if fd else None

        if formula is None:
            return {
                "solution": None,
                "satisfiable": False,
                "residual": float("inf"),
                "iterations": 0,
                "converged": False,
                "runtime_ms": 0.0,
                "method": "iterative_simplification",
                "trajectory": [],
            }

        # Start from surrogate prediction (warmstart)
        initial = prediction.get("prediction", {}) if prediction else {}
        assignment = dict(initial) if initial else {}
        trajectory: list[dict[str, Any]] = []

        for iteration in range(min(budget, self._max_iterations)):
            changed = False

            # Unit propagation
            for clause in formula.clauses:
                if len(clause.literals) == 1:
                    lit = clause.literals[0]
                    if lit.prop not in assignment:
                        assignment[lit.prop] = not lit.negated
                        changed = True

            # Pure literal elimination
            pos_counts: dict[str, int] = {}
            neg_counts: dict[str, int] = {}
            for clause in formula.clauses:
                for lit in clause.literals:
                    if lit.negated:
                        neg_counts[lit.prop] = neg_counts.get(lit.prop, 0) + 1
                    else:
                        pos_counts[lit.prop] = pos_counts.get(lit.prop, 0) + 1

            for var in formula.variables:
                if var not in assignment:
                    p = pos_counts.get(var, 0)
                    n = neg_counts.get(var, 0)
                    if p > 0 and n == 0:
                        assignment[var] = True
                        changed = True
                    elif n > 0 and p == 0:
                        assignment[var] = False
                        changed = True

            trajectory.append({
                "iteration": iteration,
                "assigned_count": len(assignment),
                "changed": changed,
            })

            if not changed:
                break

        # Fill remaining with brute-force search
        remaining = [v for v in formula.variables if v not in assignment]
        if remaining:
            for combo_idx in range(2**len(remaining)):
                trial = dict(assignment)
                for j, var in enumerate(remaining):
                    trial[var] = bool((combo_idx >> j) & 1)
                if formula.evaluate(trial):
                    assignment = trial
                    break

        # If current assignment doesn't satisfy, try all combos exhaustively
        if not formula.evaluate(assignment):
            found = False
            for combo_idx in range(2**len(formula.variables)):
                trial = {}
                for j, var in enumerate(formula.variables):
                    trial[var] = bool((combo_idx >> j) & 1)
                if formula.evaluate(trial):
                    assignment = trial
                    found = True
                    break
            if not found:
                # Truly unsatisfiable — reset to empty to mark it
                pass

        # Evaluate final result
        satisfied = formula.evaluate(assignment)
        unsat = sum(1 for cl in formula.clauses if not cl.evaluate(assignment))
        residual = unsat / max(1, len(formula.clauses))

        runtime_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "solution": assignment,
            "satisfiable": satisfied,
            "residual": residual,
            "iterations": len(trajectory),
            "converged": (residual == 0.0),
            "runtime_ms": runtime_ms,
            "method": "iterative_simplification",
            "trajectory": trajectory,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class PropositionalLogicEvaluator(DomainEvaluator):
    """Compare oracle result against projected result."""

    def evaluate(
        self,
        task: DomainTaskBase,
        solution: dict[str, Any],
    ) -> dict[str, Any]:
        # solution here is the oracle result; we also need the projection
        # We accept a dict with both results packed in
        oracle_result = solution.get("oracle", {})
        projected_result = solution.get("projection", {})

        oracle_sat = oracle_result.get("satisfiable", False)
        proj_sat = projected_result.get("satisfiable", False)

        correct = (oracle_sat == proj_sat)

        # If both say satisfiable, verify projected assignment
        if oracle_sat and proj_sat:
            proj_assignment = projected_result.get("solution")
            pl_task = task
            formula = pl_task.formula if isinstance(pl_task, PropositionalLogicTask) else None
            if hasattr(task, "metadata") and formula is None:
                fd = task.metadata.get("formula_dict")
                formula = _formula_from_dict(fd) if fd else None
            if proj_assignment and formula:
                valid = formula.evaluate(proj_assignment)
                correct = correct and valid

        return {
            "correct": correct,
            "oracle_satisfiable": oracle_sat,
            "projected_satisfiable": proj_sat,
            "oracle_residual": oracle_result.get("residual", 0.0),
            "projected_residual": projected_result.get("residual", 0.0),
        }


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

@dataclass
class PropositionalLogicConfidence:
    """Confidence estimate for propositional logic results."""

    def estimate(
        self,
        task: DomainTaskBase,
        projected_result: dict[str, Any],
    ) -> dict[str, Any]:
        residual = projected_result.get("residual", 1.0)
        converged = projected_result.get("converged", False)
        confidence_score = 1.0 - min(1.0, residual)
        likely_ood = not converged and residual > 0.5

        return {
            "confidence_score": confidence_score,
            "likely_ood": likely_ood,
            "projection_iterations": projected_result.get("iterations", 0),
        }


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def execute_propositional_logic_pipeline(
    task: PropositionalLogicTask,
    surrogate: PropositionalLogicSurrogate | None = None,
    projection: PropositionalLogicProjection | None = None,
    evaluator: PropositionalLogicEvaluator | None = None,
    confidence: PropositionalLogicConfidence | None = None,
    budget: int = 100,
) -> dict[str, Any]:
    """Execute the full pipeline for a propositional logic task."""
    surrogate = surrogate or PropositionalLogicSurrogate()
    projection = projection or PropositionalLogicProjection()
    evaluator = evaluator or PropositionalLogicEvaluator()
    confidence = confidence or PropositionalLogicConfidence()

    # Step 1: Oracle
    oracle = PropositionalLogicOracle()
    oracle_result = oracle.solve(task)

    # Step 2: Surrogate
    surrogate_result = surrogate.predict(task)

    # Step 3: Projection (warmstarted from surrogate)
    projection_result = projection.project(task, surrogate_result, budget=budget)

    # Step 4: Evaluation
    evaluation_result = evaluator.evaluate(task, {
        "oracle": oracle_result,
        "projection": projection_result,
    })

    # Step 5: Confidence
    confidence_result = confidence.estimate(task, projection_result)

    # Step 6: Trace
    formula = task.formula
    trace = {
        "task_id": task.task_id,
        "domain_name": task.domain_name,
        "fingerprint": f"pl-{hash(task.task_id) & 0xFFFFFFFF:08x}",
        "variable_count": len(formula.variables) if formula else 0,
        "clause_count": len(formula.clauses) if formula else 0,
        "surrogate_method": surrogate_result.get("method", "unknown"),
        "surrogate_residual": surrogate_result.get("residual", 0.0),
        "surrogate_runtime_ms": surrogate_result.get("runtime_ms", 0.0),
        "projection_iterations": projection_result.get("iterations", 0),
        "projection_converged": projection_result.get("converged", False),
        "projection_method": projection_result.get("method", "unknown"),
        "projection_residual": projection_result.get("residual", 0.0),
        "projection_runtime_ms": projection_result.get("runtime_ms", 0.0),
        "evaluation_correct": evaluation_result.get("correct", False),
        "confidence_score": confidence_result.get("confidence_score", 0.0),
        "trajectory_length": len(projection_result.get("trajectory", [])),
    }

    return {
        "oracle": oracle_result,
        "surrogate": surrogate_result,
        "projection": projection_result,
        "evaluation": evaluation_result,
        "confidence": confidence_result,
        "trace": trace,
    }
