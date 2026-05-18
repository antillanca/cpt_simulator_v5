"""Model evaluation harness for CPT v2.7 distillation readiness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.traces.schema import ReasoningTrace, assert_replay_consistency
from backend.validation.failure_taxonomy import classify_failure
from backend.validation.oracle_arena import ArenaExample, ArenaResult, aggregate_arena_results, compare_oracle_vs_model


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def exact_match_rate(predictions: list[dict], oracle_answers: list[dict]) -> float:
    if not predictions:
        return 0.0
    matches = sum(1 for p, o in zip(predictions, oracle_answers) if p == o)
    return matches / len(predictions)


def token_or_struct_match_rate(predictions: list[dict], oracle_answers: list[dict]) -> float:
    if not predictions:
        return 0.0
    matches = 0
    for p, o in zip(predictions, oracle_answers):
        if set(p.keys()) == set(o.keys()) and all(type(p.get(k)) == type(o.get(k)) for k in p if k in o):
            matches += 1
    return matches / len(predictions)


def invariant_violation_rate(predictions: list[dict], invariants_list: list[list[str]]) -> float:
    if not predictions:
        return 0.0
    violations = 0
    for pred, _invariants in zip(predictions, invariants_list):
        verification = pred.get("verification_status", {})
        if not verification.get("passed", True):
            violations += 1
    return violations / len(predictions)


def trace_consistency(predictions: list[dict], oracle_traces: list[list[dict]]) -> float:
    if not predictions:
        return 0.0
    consistent = 0
    for pred, oracle_steps in zip(predictions, oracle_traces):
        pred_trace = pred.get("reasoning_trace", [])
        if len(pred_trace) == len(oracle_steps):
            consistent += 1
    return consistent / len(predictions)


def answer_consistency(predictions: list[dict], oracle_answers: list[dict]) -> float:
    if not predictions:
        return 0.0
    consistent = 0
    for pred, oracle in zip(predictions, oracle_answers):
        pred_answer = pred.get("final_answer", {})
        oracle_answer = oracle.get("final_answer", {})
        if isinstance(pred_answer, dict) and isinstance(oracle_answer, dict):
            shared_keys = set(pred_answer.keys()) & set(oracle_answer.keys())
            if shared_keys:
                key_match = sum(1 for k in shared_keys if pred_answer[k] == oracle_answer[k])
                if key_match == len(shared_keys):
                    consistent += 1
        elif pred_answer == oracle_answer:
            consistent += 1
    return consistent / len(predictions)


def trajectory_deviation(predictions: list[dict], oracle_traces: list[list[dict]]) -> float:
    if not predictions:
        return float("inf")
    deviations = []
    for pred, oracle_steps in zip(predictions, oracle_traces):
        pred_len = len(pred.get("reasoning_trace", []))
        oracle_len = len(oracle_steps)
        deviations.append(abs(pred_len - oracle_len))
    return sum(deviations) / len(deviations)


def replay_consistency(predictions: list[dict]) -> float:
    if not predictions:
        return 0.0
    consistent = 0
    for pred in predictions:
        trace_data = pred.get("trace_export", pred.get("reasoning_trace", {}))
        try:
            if isinstance(trace_data, dict) and "steps" in trace_data:
                trace = ReasoningTrace.from_dict(trace_data)
                initial = pred.get("structured_state", {}).get("initial_state", {})
                final = pred.get("final_answer", {})
                result = assert_replay_consistency(trace, initial_state=initial, expected_final=final)
                if result.passed:
                    consistent += 1
        except Exception:
            pass
    return consistent / len(predictions)


def invariant_retention_rate(predictions: list[dict], oracle_records: list[dict]) -> float:
    if not predictions:
        return 0.0
    retained = 0
    for pred, oracle in zip(predictions, oracle_records):
        oracle_inv = set(map(str, oracle.get("invariants_checked", [])))
        pred_inv = set(map(str, pred.get("invariants_checked", [])))
        verification = pred.get("verification_status", {})
        if verification.get("passed", False) and oracle_inv.issubset(pred_inv):
            retained += 1
    return retained / len(predictions)


def causal_fidelity(predictions: list[dict], oracle_records: list[dict]) -> float:
    if not predictions:
        return 0.0
    matches = 0
    for pred, oracle in zip(predictions, oracle_records):
        oracle_trace = oracle.get("reasoning_trace", [])
        pred_trace = pred.get("reasoning_trace", [])
        if len(oracle_trace) == len(pred_trace) and len(oracle_trace) > 0:
            matches += 1
    return matches / len(predictions)


def replay_stability(predictions: list[dict]) -> float:
    return replay_consistency(predictions)


def extrapolation_robustness(results: list[ArenaResult]) -> float:
    if not results:
        return 0.0
    failures = sum(1 for result in results if result.failure_type == "extrapolation_collapse")
    return 1.0 - (failures / len(results))


@dataclass
class ModelEvaluationResult:
    model_type: str = ""
    dataset_version: str = "2.7.0"
    total_samples: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    by_layer: dict[int, dict[str, float]] = field(default_factory=dict)
    by_module: dict[str, dict[str, float]] = field(default_factory=dict)
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    failure_breakdown: dict[str, int] = field(default_factory=dict)
    timestamp: str = ""
    fingerprint: str = ""

    def compute_fingerprint(self) -> str:
        payload = {
            "model_type": self.model_type,
            "dataset_version": self.dataset_version,
            "total_samples": self.total_samples,
            "metrics": dict(sorted(self.metrics.items())),
            "by_layer": {str(k): dict(sorted(v.items())) for k, v in sorted(self.by_layer.items())},
            "by_module": dict(sorted(self.by_module.items())),
            "by_category": dict(sorted(self.by_category.items())),
            "failure_breakdown": dict(sorted(self.failure_breakdown.items())),
        }
        return _stable_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        self.fingerprint = self.compute_fingerprint()
        return {
            "model_type": self.model_type,
            "dataset_version": self.dataset_version,
            "total_samples": self.total_samples,
            "metrics": dict(sorted(self.metrics.items())),
            "by_layer": {str(k): dict(sorted(v.items())) for k, v in sorted(self.by_layer.items())},
            "by_module": dict(sorted(self.by_module.items())),
            "by_category": dict(sorted(self.by_category.items())),
            "failure_breakdown": dict(sorted(self.failure_breakdown.items())),
            "timestamp": self.timestamp,
            "fingerprint": self.fingerprint,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path


def _failure_breakdown(results: list[ArenaResult]) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for result in results:
        key = result.failure_type or "pass"
        breakdown[key] = breakdown.get(key, 0) + 1
    return breakdown


class ModelEvaluator:
    """Evaluate model predictions against oracle truth."""

    def __init__(self, model_type: str = "unknown"):
        self.model_type = model_type

    def evaluate(self, predictions: list[dict], oracle_records: list[dict]) -> ModelEvaluationResult:
        import time as _time

        oracle_answers = [r.get("final_answer", {}) for r in oracle_records]
        oracle_traces = [r.get("reasoning_trace", []) for r in oracle_records]
        invariants_list = [r.get("invariants_checked", []) for r in oracle_records]
        pred_answers = [p.get("final_answer", {}) for p in predictions]

        arena_results: list[ArenaResult] = []
        for index, (pred, oracle) in enumerate(zip(predictions, oracle_records)):
            arena_results.append(
                compare_oracle_vs_model(
                    ArenaExample(
                        sample_id=str(oracle.get("sample_id", index)),
                        question=str(oracle.get("question", "")),
                        oracle=oracle,
                        model_output=pred,
                        metadata={
                            "module_source": oracle.get("module_source", ""),
                            "curriculum_layer": oracle.get("curriculum_layer", -1),
                            "category": oracle.get("module_source", ""),
                            "initial_state": oracle.get("structured_state", {}).get("initial_state", {}),
                            "is_ood": bool(oracle.get("structured_state", {}).get("ood", False)),
                        },
                    )
                )
            )

        metrics = {
            "exact_match_rate": exact_match_rate(pred_answers, oracle_answers),
            "token_or_struct_match_rate": token_or_struct_match_rate(pred_answers, oracle_answers),
            "invariant_violation_rate": invariant_violation_rate(predictions, invariants_list),
            "trace_consistency": trace_consistency(predictions, oracle_traces),
            "answer_consistency": answer_consistency(predictions, oracle_records),
            "trajectory_deviation": trajectory_deviation(predictions, oracle_traces),
            "replay_consistency": replay_consistency(predictions),
            "invariant_retention_rate": invariant_retention_rate(predictions, oracle_records),
            "causal_fidelity": causal_fidelity(predictions, oracle_records),
            "replay_stability": replay_stability(predictions),
            "extrapolation_robustness": extrapolation_robustness(arena_results),
        }

        result = ModelEvaluationResult(
            model_type=self.model_type,
            total_samples=min(len(predictions), len(oracle_records)),
            metrics=metrics,
            failure_breakdown=_failure_breakdown(arena_results),
            timestamp=_time.strftime("%Y-%m-%dT%H:%M:%S", _time.gmtime()),
        )

        result.by_layer = {int(key): value for key, value in aggregate_arena_results(arena_results, group_by="layer").items()}
        result.by_module = aggregate_arena_results(arena_results, group_by="module")
        result.by_category = aggregate_arena_results(arena_results, group_by="category")
        return result
