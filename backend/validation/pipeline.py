"""Validation pipeline for comparing neural predictions against the sandbox."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from backend.core_truth.sandbox import sandbox_manager
from backend.verifiers import verify_simulation


@dataclass
class ValidationReport:
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    violations: list[dict[str, Any]] = field(default_factory=list)


class ValidationPipeline:
    def __init__(self, invariant_set: list[str] | None = None, violation_threshold: float = 0.0):
        self.invariant_set = invariant_set or ["logic_basic"]
        self.violation_threshold = violation_threshold

    def evaluate(self, simulation_cases: list[dict[str, Any]], model_predictor: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> ValidationReport:
        metrics: dict[str, Any] = {"cases": len(simulation_cases), "prediction_error": 0.0, "invariant_violation_rate": 0.0,
                                   "symbolic_consistency": 0.0, "trajectory_deviation": 0.0}
        violations: list[dict[str, Any]] = []
        prediction_errors = []
        symbolic_matches = 0
        trajectory_deviations = []
        invariant_failures = 0

        for case in simulation_cases:
            rule = case["rule"]
            state = case.get("initial_state", {})
            expected = sandbox_manager.run_rule(rule, state, frames=case.get("frames", 1), collect_trace=True)
            invariant_result = verify_simulation(expected.get("trace", expected), self.invariant_set)
            if not invariant_result["passed"]:
                invariant_failures += 1
                violations.extend(invariant_result["violations"])

            if model_predictor is not None:
                predicted = model_predictor(case)
                actual = expected.get("particle", {})
                prediction_errors.append(self._state_distance(predicted, actual))
                if self._symbolic_consistent(predicted, actual):
                    symbolic_matches += 1
                trajectory_deviations.append(self._trajectory_distance(predicted, actual))

        n = max(len(simulation_cases), 1)
        metrics["prediction_error"] = sum(prediction_errors) / max(len(prediction_errors), 1)
        metrics["invariant_violation_rate"] = invariant_failures / n
        metrics["symbolic_consistency"] = symbolic_matches / max(len(simulation_cases), 1)
        metrics["trajectory_deviation"] = sum(trajectory_deviations) / max(len(trajectory_deviations), 1)

        passed = metrics["invariant_violation_rate"] <= self.violation_threshold
        return ValidationReport(passed=passed, metrics=metrics, violations=violations)

    @staticmethod
    def _state_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
        keys = {"x", "y", "vx", "vy"} | set(a.keys()) | set(b.keys())
        distance = 0.0
        for key in keys:
            av = float(a.get(key, 0.0) or 0.0)
            bv = float(b.get(key, 0.0) or 0.0)
            distance += abs(av - bv)
        return distance

    @staticmethod
    def _symbolic_consistent(predicted: dict[str, Any], actual: dict[str, Any]) -> bool:
        for key in ("x", "y", "vx", "vy"):
            if key in predicted and key in actual:
                if abs(float(predicted.get(key, 0.0) or 0.0) - float(actual.get(key, 0.0) or 0.0)) > 1e-6:
                    return False
        return True

    @staticmethod
    def _trajectory_distance(predicted: dict[str, Any], actual: dict[str, Any]) -> float:
        return ValidationPipeline._state_distance(predicted, actual)

