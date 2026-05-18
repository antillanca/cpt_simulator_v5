"""Oracle-vs-model arena for deterministic distillation experiments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from backend.traces.schema import ReasoningTrace, assert_replay_consistency


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _stable_dump(value: Any) -> str:
    return json.dumps(_normalize(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _as_trace(payload: Any) -> ReasoningTrace | None:
    if isinstance(payload, ReasoningTrace):
        return payload
    if isinstance(payload, list):
        try:
            return ReasoningTrace.from_dict({"steps": payload, "metadata": {}})
        except Exception:
            return None
    if isinstance(payload, dict) and payload.get("steps") is not None:
        try:
            return ReasoningTrace.from_dict(payload)
        except Exception:
            return None
    return None


def _answer_value(payload: dict[str, Any]) -> Any:
    if "final_answer" in payload:
        return payload.get("final_answer")
    if "prediction" in payload:
        return payload.get("prediction")
    return payload


def _structural_match(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        if set(left) != set(right):
            return False
        return all(_structural_match(left[key], right[key]) for key in left)
    if isinstance(left, list):
        if len(left) != len(right):
            return False
        return all(_structural_match(a, b) for a, b in zip(left, right))
    return True


def _trace_similarity(oracle_trace: Any, model_trace: Any) -> float:
    oracle = _as_trace(oracle_trace)
    model = _as_trace(model_trace)
    if oracle is None or model is None:
        return 0.0
    oracle_steps = oracle.ordered_steps()
    model_steps = model.ordered_steps()
    if not oracle_steps and not model_steps:
        return 1.0
    if not oracle_steps or not model_steps:
        return 0.0

    shared = 0
    for oracle_step, model_step in zip(oracle_steps, model_steps):
        if oracle_step.equation == model_step.equation and oracle_step.operation == model_step.operation:
            shared += 1
    return shared / max(len(oracle_steps), len(model_steps))


@dataclass(frozen=True)
class ArenaExample:
    sample_id: str
    question: str
    oracle: dict[str, Any]
    model_output: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArenaResult:
    exact_match: bool
    struct_match: bool
    invariant_violation: bool
    trace_consistency: float
    answer_consistency: float
    trajectory_deviation: float
    replay_consistency: bool
    failure_type: str | None = None
    sample_id: str = ""
    module_source: str = ""
    curriculum_layer: int = -1
    category: str = ""
    is_ood: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_match": self.exact_match,
            "struct_match": self.struct_match,
            "invariant_violation": self.invariant_violation,
            "trace_consistency": self.trace_consistency,
            "answer_consistency": self.answer_consistency,
            "trajectory_deviation": self.trajectory_deviation,
            "replay_consistency": self.replay_consistency,
            "failure_type": self.failure_type,
            "sample_id": self.sample_id,
            "module_source": self.module_source,
            "curriculum_layer": self.curriculum_layer,
            "category": self.category,
            "is_ood": self.is_ood,
        }


def compare_oracle_vs_model(example: ArenaExample) -> ArenaResult:
    """Compare model output to oracle truth and compute deterministic metrics."""

    oracle = example.oracle
    model = example.model_output

    oracle_answer = _answer_value(oracle)
    model_answer = _answer_value(model)
    exact_match = _normalize(oracle_answer) == _normalize(model_answer)
    struct_match = _structural_match(oracle_answer, model_answer)

    oracle_verification = oracle.get("verification_status", {})
    model_verification = model.get("verification_status", {})
    invariant_violation = not bool(model_verification.get("passed", oracle_verification.get("passed", True)))

    oracle_trace = oracle.get("trace_export", oracle.get("reasoning_trace", []))
    model_trace = model.get("trace_export", model.get("reasoning_trace", []))
    trace_consistency = _trace_similarity(oracle_trace, model_trace)

    oracle_trace_obj = _as_trace(oracle_trace)
    model_trace_obj = _as_trace(model_trace)
    oracle_steps = len(oracle_trace_obj.ordered_steps()) if oracle_trace_obj else len(oracle.get("reasoning_trace", []))
    model_steps = len(model_trace_obj.ordered_steps()) if model_trace_obj else len(model.get("reasoning_trace", []))
    trajectory_deviation = float(abs(oracle_steps - model_steps))

    answer_consistency = 1.0 if exact_match else (1.0 if struct_match and _stable_dump(oracle_answer) == _stable_dump(model_answer) else 0.0)

    initial_state = example.metadata.get("initial_state")
    if initial_state is None:
        initial_state = oracle.get("structured_state", {}).get("initial_state", {})
    expected_final = oracle.get("final_answer", oracle_answer)
    replay_ok = False
    try:
        if model_trace_obj is not None:
            replay_ok = bool(assert_replay_consistency(model_trace_obj, initial_state=initial_state, expected_final=expected_final).passed)
    except Exception:
        replay_ok = False

    if exact_match and replay_ok:
        failure_type = None
    else:
        from backend.validation.failure_taxonomy import classify_failure

        failure_type = classify_failure(
            ArenaResult(
                exact_match=exact_match,
                struct_match=struct_match,
                invariant_violation=invariant_violation,
                trace_consistency=trace_consistency,
                answer_consistency=answer_consistency,
                trajectory_deviation=trajectory_deviation,
                replay_consistency=replay_ok,
                sample_id=example.sample_id,
                module_source=str(example.metadata.get("module_source", oracle.get("module_source", ""))),
                curriculum_layer=int(example.metadata.get("curriculum_layer", oracle.get("curriculum_layer", -1))),
                category=str(example.metadata.get("category", oracle.get("module_source", ""))),
            )
        )

    return ArenaResult(
        exact_match=exact_match,
        struct_match=struct_match,
        invariant_violation=invariant_violation,
        trace_consistency=trace_consistency,
        answer_consistency=answer_consistency,
        trajectory_deviation=trajectory_deviation,
        replay_consistency=replay_ok,
        failure_type=failure_type,
        sample_id=example.sample_id,
        module_source=str(example.metadata.get("module_source", oracle.get("module_source", ""))),
        curriculum_layer=int(example.metadata.get("curriculum_layer", oracle.get("curriculum_layer", -1))),
        category=str(example.metadata.get("category", oracle.get("module_source", ""))),
        is_ood=bool(example.metadata.get("is_ood", oracle.get("structured_state", {}).get("ood", False))),
    )


def aggregate_arena_results(results: list[ArenaResult], *, group_by: str = "module") -> dict[str, dict[str, float]]:
    """Aggregate arena results by module, layer, or category."""

    buckets: dict[str, list[ArenaResult]] = {}
    for result in results:
        if group_by == "layer":
            key = str(result.curriculum_layer)
        elif group_by == "category":
            key = result.category or "unknown"
        else:
            key = result.module_source or "unknown"
        buckets.setdefault(key, []).append(result)

    summary: dict[str, dict[str, float]] = {}
    for key, bucket in sorted(buckets.items(), key=lambda item: item[0]):
        total = float(len(bucket)) or 1.0
        summary[key] = {
            "sample_count": float(len(bucket)),
            "exact_match_rate": sum(1 for item in bucket if item.exact_match) / total,
            "struct_match_rate": sum(1 for item in bucket if item.struct_match) / total,
            "invariant_violation_rate": sum(1 for item in bucket if item.invariant_violation) / total,
            "trace_consistency": sum(item.trace_consistency for item in bucket) / total,
            "answer_consistency": sum(item.answer_consistency for item in bucket) / total,
            "trajectory_deviation": sum(item.trajectory_deviation for item in bucket) / total,
            "replay_consistency": sum(1 for item in bucket if item.replay_consistency) / total,
        }
    return summary
