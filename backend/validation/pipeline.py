"""Validation pipeline for comparing neural predictions against the sandbox."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from backend.core_truth.sandbox import sandbox_manager
from backend.verifiers import verify_simulation
from backend.validation.thresholds import InvariantThresholds, invariant_family


@dataclass
class ValidationReport:
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    violations: list[dict[str, Any]] = field(default_factory=list)
    rejected: bool = False


@dataclass
class ValidationCaseResult:
    case: dict[str, Any]
    sandbox_result: dict[str, Any]
    model_prediction: dict[str, Any] | None
    invariant_result: dict[str, Any]
    prediction_error: float
    trajectory_deviation: float
    symbolic_consistency: float
    passed: bool
    violations: list[dict[str, Any]]


class ValidationPipeline:
    def __init__(
        self,
        invariant_set: list[str] | None = None,
        violation_threshold: float = 0.0,
        thresholds: InvariantThresholds | None = None,
    ):
        self.invariant_set = invariant_set or ["logic_basic"]
        self.violation_threshold = violation_threshold
        self.thresholds = thresholds or InvariantThresholds.from_env()

    def sandbox_vs_model(
        self,
        case: dict[str, Any],
        model_predictor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> ValidationCaseResult:
        rule = case["rule"]
        state = case.get("initial_state", {})
        frames = case.get("frames", 1)
        sandbox_result = sandbox_manager.run_rule(rule, state, frames=frames, collect_trace=True)
        if sandbox_result.get("status") != "ok":
            violation = {"reason": sandbox_result.get("message", "sandbox failure")}
            return ValidationCaseResult(
                case=case,
                sandbox_result=sandbox_result,
                model_prediction=model_predictor(case) if model_predictor is not None else None,
                invariant_result={"passed": False, "violations": [violation], "metrics": {}},
                prediction_error=0.0,
                trajectory_deviation=0.0,
                symbolic_consistency=0.0,
                passed=False,
                violations=[violation],
            )

        invariant_result = verify_simulation(sandbox_result.get("trace", {}), self.invariant_set)

        prediction = model_predictor(case) if model_predictor is not None else None
        actual = sandbox_result.get("particle", {})
        prediction_error = self._state_distance(prediction or {}, actual) if prediction is not None else 0.0
        trajectory_deviation = self._trajectory_distance(prediction or {}, actual) if prediction is not None else 0.0
        symbolic_consistency = self._symbolic_consistency_score(prediction or {}, actual) if prediction is not None else 0.0

        violations = list(invariant_result.get("violations", []))
        passed = bool(invariant_result.get("passed", False))
        return ValidationCaseResult(
            case=case,
            sandbox_result=sandbox_result,
            model_prediction=prediction,
            invariant_result=invariant_result,
            prediction_error=prediction_error,
            trajectory_deviation=trajectory_deviation,
            symbolic_consistency=symbolic_consistency,
            passed=passed,
            violations=violations,
        )

    def evaluate(self, simulation_cases: list[dict[str, Any]], model_predictor: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> ValidationReport:
        metrics: dict[str, Any] = {
            "cases": len(simulation_cases),
            "prediction_error": 0.0,
            "invariant_violation_rate": 0.0,
            "symbolic_consistency": 0.0,
            "trajectory_deviation": 0.0,
            "exact_match_rate": 0.0,
            "causal_consistency": 0.0,
            "family_violation_rates": {},
            "thresholds": self.thresholds.to_dict(),
        }
        violations: list[dict[str, Any]] = []
        prediction_errors: list[float] = []
        symbolic_matches = 0
        trajectory_deviations: list[float] = []
        invariant_failures = 0
        exact_matches = 0
        causal_matches = 0
        case_results: list[ValidationCaseResult] = []
        family_totals: dict[str, int] = {}
        family_failures: dict[str, int] = {}

        for case in simulation_cases:
            case_result = self.sandbox_vs_model(case, model_predictor=model_predictor)
            case_results.append(case_result)
            if not case_result.passed:
                invariant_failures += 1
                violations.extend(case_result.violations)

            case_invariants = list(case.get("invariants", self.invariant_set)) or list(self.invariant_set)
            families = {invariant_family(name) for name in case_invariants}
            for family in families:
                family_totals[family] = family_totals.get(family, 0) + 1
                if not case_result.passed:
                    family_failures[family] = family_failures.get(family, 0) + 1

            if model_predictor is not None:
                predicted = case_result.model_prediction or {}
                actual = case_result.sandbox_result.get("particle", {})
                prediction_errors.append(case_result.prediction_error)
                if case_result.symbolic_consistency >= 1.0:
                    symbolic_matches += 1
                trajectory_deviations.append(case_result.trajectory_deviation)
                if self._exact_match(predicted, actual):
                    exact_matches += 1
                if self._causal_consistent(case, predicted, actual):
                    causal_matches += 1

        n = max(len(simulation_cases), 1)
        metrics["prediction_error"] = sum(prediction_errors) / max(len(prediction_errors), 1)
        metrics["invariant_violation_rate"] = invariant_failures / n
        metrics["symbolic_consistency"] = symbolic_matches / max(len(simulation_cases), 1)
        metrics["trajectory_deviation"] = sum(trajectory_deviations) / max(len(trajectory_deviations), 1)
        metrics["exact_match_rate"] = exact_matches / max(len(simulation_cases), 1)
        metrics["causal_consistency"] = causal_matches / max(len(simulation_cases), 1)
        family_rates: dict[str, float] = {}
        for family, total in family_totals.items():
            family_rates[family] = family_failures.get(family, 0) / max(total, 1)
        metrics["family_violation_rates"] = family_rates
        metrics["threshold_profile"] = self.thresholds.to_dict()

        passed = metrics["invariant_violation_rate"] <= self.violation_threshold
        for family, rate in family_rates.items():
            if rate > self.thresholds.threshold_for_family(family):
                passed = False
        rejected = not passed
        return ValidationReport(passed=passed, metrics=metrics, violations=violations, rejected=rejected)

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
    def _symbolic_consistency_score(predicted: dict[str, Any], actual: dict[str, Any]) -> float:
        keys = [key for key in ("x", "y", "vx", "vy") if key in predicted or key in actual]
        if not keys:
            return 0.0
        matches = 0
        for key in keys:
            if abs(float(predicted.get(key, 0.0) or 0.0) - float(actual.get(key, 0.0) or 0.0)) <= 1e-6:
                matches += 1
        return matches / len(keys)

    @staticmethod
    def _trajectory_distance(predicted: dict[str, Any], actual: dict[str, Any]) -> float:
        return ValidationPipeline._state_distance(predicted, actual)

    @staticmethod
    def _exact_match(predicted: dict[str, Any], actual: dict[str, Any]) -> bool:
        keys = set(predicted.keys()) | set(actual.keys())
        for key in keys:
            if predicted.get(key) != actual.get(key):
                return False
        return True

    @staticmethod
    def _causal_consistent(case: dict[str, Any], predicted: dict[str, Any], actual: dict[str, Any]) -> bool:
        expected_state = case.get("expected_state", {})
        if not isinstance(expected_state, dict) or not expected_state:
            return ValidationPipeline._exact_match(predicted, actual)
        for key, value in expected_state.items():
            if actual.get(key) != value:
                return False
        return True
