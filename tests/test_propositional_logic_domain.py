"""CORE v3.2 — Propositional Logic Domain Tests.

Verify that the propositional_logic domain runs through the same
runtime interface as circuits and linear_system.
"""

from __future__ import annotations

import pytest

from core_runtime.domains.propositional_logic import (
    CNFFormula,
    Clause,
    Literal,
    PropositionalLogicEvaluator,
    PropositionalLogicOracle,
    PropositionalLogicProjection,
    PropositionalLogicSurrogate,
    PropositionalLogicTask,
    execute_propositional_logic_pipeline,
    make_task,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sat_formula() -> CNFFormula:
    """(A | B) & (~A | C) & (~B | ~C) — satisfiable."""
    return CNFFormula(
        clauses=(
            Clause((Literal("A"), Literal("B"))),
            Clause((Literal("A", negated=True), Literal("C"))),
            Clause((Literal("B", negated=True), Literal("C", negated=True))),
        ),
        variables=("A", "B", "C"),
    )


def _unsat_formula() -> CNFFormula:
    """A & ~A — unsatisfiable."""
    return CNFFormula(
        clauses=(
            Clause((Literal("A"),)),
            Clause((Literal("A", negated=True),)),
        ),
        variables=("A",),
    )


def _tautology_formula() -> CNFFormula:
    """(A | ~A) — always satisfiable."""
    return CNFFormula(
        clauses=(
            Clause((Literal("A"), Literal("A", negated=True))),
        ),
        variables=("A",),
    )


def _larger_formula() -> CNFFormula:
    """4-variable formula with mixed clauses."""
    return CNFFormula(
        clauses=(
            Clause((Literal("A"), Literal("B"))),
            Clause((Literal("B", negated=True), Literal("C"))),
            Clause((Literal("C", negated=True), Literal("D"))),
            Clause((Literal("D", negated=True), Literal("A", negated=True))),
            Clause((Literal("A", negated=True), Literal("C", negated=True))),
        ),
        variables=("A", "B", "C", "D"),
    )


# ---------------------------------------------------------------------------
# Formula tests
# ---------------------------------------------------------------------------

class TestCNFFormula:
    def test_sat_formula_is_satisfiable(self):
        f = _sat_formula()
        assert f.find_satisfying() is not None
        assert f.count_satisfying() > 0

    def test_unsat_formula_is_not_satisfiable(self):
        f = _unsat_formula()
        assert f.find_satisfying() is None
        assert f.count_satisfying() == 0

    def test_tautology_is_satisfiable(self):
        f = _tautology_formula()
        assert f.find_satisfying() is not None
        assert f.count_satisfying() == 2

    def test_evaluate_specific_assignment(self):
        f = _sat_formula()
        assert f.evaluate({"A": True, "B": False, "C": True})
        assert f.evaluate({"A": False, "B": True, "C": False})
        assert not f.evaluate({"A": True, "B": True, "C": True})


# ---------------------------------------------------------------------------
# Oracle tests
# ---------------------------------------------------------------------------

class TestOracle:
    def test_oracle_finds_sat(self):
        task = make_task("sat_01", _sat_formula())
        oracle = PropositionalLogicOracle()
        result = oracle.solve(task)
        assert result["satisfiable"] is True
        assert result["assignment"] is not None
        assert result["residual"] == 0.0

    def test_oracle_detects_unsat(self):
        task = make_task("unsat_01", _unsat_formula())
        oracle = PropositionalLogicOracle()
        result = oracle.solve(task)
        assert result["satisfiable"] is False
        assert result["assignment"] is None

    def test_oracle_total_assignments(self):
        task = make_task("taut_01", _tautology_formula())
        oracle = PropositionalLogicOracle()
        result = oracle.solve(task)
        assert result["total_assignments"] == 2


# ---------------------------------------------------------------------------
# Surrogate tests
# ---------------------------------------------------------------------------

class TestSurrogate:
    def test_surrogate_returns_prediction(self):
        task = make_task("sat_01", _sat_formula())
        surrogate = PropositionalLogicSurrogate()
        result = surrogate.predict(task)
        assert "prediction" in result
        assert "residual" in result
        assert result["method"] == "unit_clause_shortcut"

    def test_surrogate_unit_clause(self):
        # Formula with unit clause: A & (A | B)
        f = CNFFormula(
            clauses=(
                Clause((Literal("A"),)),
                Clause((Literal("A"), Literal("B"))),
            ),
            variables=("A", "B"),
        )
        task = make_task("unit_01", f)
        surrogate = PropositionalLogicSurrogate()
        result = surrogate.predict(task)
        assert result["prediction"]["A"] is True


# ---------------------------------------------------------------------------
# Projection tests
# ---------------------------------------------------------------------------

class TestProjection:
    def test_projection_finds_sat(self):
        task = make_task("sat_01", _sat_formula())
        surrogate = PropositionalLogicSurrogate()
        surr_result = surrogate.predict(task)
        projection = PropositionalLogicProjection()
        result = projection.project(task, surr_result, budget=50)
        assert result["satisfiable"] is True
        assert result["converged"] is True

    def test_projection_detects_unsat(self):
        task = make_task("unsat_01", _unsat_formula())
        surrogate = PropositionalLogicSurrogate()
        surr_result = surrogate.predict(task)
        projection = PropositionalLogicProjection()
        result = projection.project(task, surr_result, budget=50)
        assert result["satisfiable"] is False

    def test_projection_iterations(self):
        task = make_task("sat_01", _sat_formula())
        surrogate = PropositionalLogicSurrogate()
        surr_result = surrogate.predict(task)
        projection = PropositionalLogicProjection()
        result = projection.project(task, surr_result, budget=50)
        assert isinstance(result["iterations"], int)
        assert result["iterations"] >= 0


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

class TestEvaluator:
    def test_evaluator_correct_sat(self):
        task = make_task("sat_01", _sat_formula())
        oracle = PropositionalLogicOracle()
        surrogate = PropositionalLogicSurrogate()
        projection = PropositionalLogicProjection()

        oracle_result = oracle.solve(task)
        surr_result = surrogate.predict(task)
        proj_result = projection.project(task, surr_result, budget=50)

        evaluator = PropositionalLogicEvaluator()
        eval_result = evaluator.evaluate(task, {
            "oracle": oracle_result,
            "projection": proj_result,
        })
        assert eval_result["correct"] is True

    def test_evaluator_correct_unsat(self):
        task = make_task("unsat_01", _unsat_formula())
        oracle = PropositionalLogicOracle()
        surrogate = PropositionalLogicSurrogate()
        projection = PropositionalLogicProjection()

        oracle_result = oracle.solve(task)
        surr_result = surrogate.predict(task)
        proj_result = projection.project(task, surr_result, budget=50)

        evaluator = PropositionalLogicEvaluator()
        eval_result = evaluator.evaluate(task, {
            "oracle": oracle_result,
            "projection": proj_result,
        })
        assert eval_result["correct"] is True


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_pipeline_sat(self):
        task = make_task("sat_01", _sat_formula())
        result = execute_propositional_logic_pipeline(task, budget=50)
        assert result["oracle"]["satisfiable"] is True
        assert result["projection"]["satisfiable"] is True
        assert result["evaluation"]["correct"] is True
        assert result["trace"]["evaluation_correct"] is True

    def test_pipeline_unsat(self):
        task = make_task("unsat_01", _unsat_formula())
        result = execute_propositional_logic_pipeline(task, budget=50)
        assert result["oracle"]["satisfiable"] is False
        assert result["projection"]["satisfiable"] is False
        assert result["evaluation"]["correct"] is True

    def test_pipeline_tautology(self):
        task = make_task("taut_01", _tautology_formula())
        result = execute_propositional_logic_pipeline(task, budget=50)
        assert result["oracle"]["satisfiable"] is True
        assert result["projection"]["satisfiable"] is True
        assert result["evaluation"]["correct"] is True

    def test_pipeline_larger_formula(self):
        task = make_task("large_01", _larger_formula())
        result = execute_propositional_logic_pipeline(task, budget=50)
        assert result["evaluation"]["correct"] is True

    def test_pipeline_deterministic(self):
        """Same task + same budget -> same result."""
        task = make_task("det_01", _sat_formula())
        r1 = execute_propositional_logic_pipeline(task, budget=50)
        r2 = execute_propositional_logic_pipeline(task, budget=50)
        assert r1["oracle"]["satisfiable"] == r2["oracle"]["satisfiable"]
        assert r1["projection"]["satisfiable"] == r2["projection"]["satisfiable"]
        assert r1["evaluation"]["correct"] == r2["evaluation"]["correct"]
        assert r1["trace"]["evaluation_correct"] == r2["trace"]["evaluation_correct"]

    def test_pipeline_trace_keys(self):
        task = make_task("trace_01", _sat_formula())
        result = execute_propositional_logic_pipeline(task, budget=50)
        trace = result["trace"]
        required_keys = [
            "task_id", "domain_name", "fingerprint",
            "variable_count", "clause_count",
            "surrogate_method", "surrogate_residual",
            "projection_iterations", "projection_converged",
            "evaluation_correct", "confidence_score",
        ]
        for key in required_keys:
            assert key in trace, f"Missing trace key: {key}"

    def test_pipeline_output_structure(self):
        task = make_task("struct_01", _sat_formula())
        result = execute_propositional_logic_pipeline(task, budget=50)
        assert "oracle" in result
        assert "surrogate" in result
        assert "projection" in result
        assert "evaluation" in result
        assert "confidence" in result
        assert "trace" in result


# ---------------------------------------------------------------------------
# Serialization roundtrip tests
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_formula_dict_roundtrip(self):
        from core_runtime.domains.propositional_logic import _formula_to_dict, _formula_from_dict
        original = _sat_formula()
        d = _formula_to_dict(original)
        restored = _formula_from_dict(d)
        assert len(restored.clauses) == len(original.clauses)
        assert restored.variables == original.variables
        assert restored.evaluate({"A": True, "B": False, "C": True})

    def test_task_metadata_roundtrip(self):
        formula = _sat_formula()
        task = make_task("rt_01", formula)
        restored_formula = task.formula
        assert restored_formula is not None
        assert len(restored_formula.clauses) == len(formula.clauses)
        assert restored_formula.variables == formula.variables


# ---------------------------------------------------------------------------
# Cross-domain runtime compatibility
# ---------------------------------------------------------------------------

class TestRuntimeCompat:
    def test_task_is_domain_task_base(self):
        task = make_task("compat_01", _sat_formula())
        from core_runtime.core.domain_sdk import DomainTaskBase
        assert isinstance(task, DomainTaskBase)

    def test_oracle_is_domain_oracle(self):
        from core_runtime.core.domain_sdk import DomainOracle
        assert isinstance(PropositionalLogicOracle(), DomainOracle)

    def test_surrogate_is_domain_surrogate(self):
        from core_runtime.core.domain_sdk import DomainSurrogate
        assert isinstance(PropositionalLogicSurrogate(), DomainSurrogate)

    def test_projection_is_domain_projection(self):
        from core_runtime.core.domain_sdk import DomainProjection
        assert isinstance(PropositionalLogicProjection(), DomainProjection)

    def test_evaluator_is_domain_evaluator(self):
        from core_runtime.core.domain_sdk import DomainEvaluator
        assert isinstance(PropositionalLogicEvaluator(), DomainEvaluator)
