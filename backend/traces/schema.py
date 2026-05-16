"""Canonical structured trace schema for deterministic reasoning.

The sandbox remains the source of truth. This layer only normalizes and
serializes replayable reasoning traces extracted from sandbox execution.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


class TraceValidationError(ValueError):
    """Raised when a trace does not satisfy the canonical schema."""


def _sorted_mapping(value: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in sorted(value):
        normalized[key] = _normalize(value[key])
    return normalized


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return _sorted_mapping(value)
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    return value


def _ensure_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TraceValidationError(f"{field_name} must be a mapping.")
    return value


def _ensure_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TraceValidationError(f"{field_name} must be a list.")
    return value


@dataclass(frozen=True)
class TraceStep:
    step_id: int
    rule: str
    equation: str
    inputs: dict[str, Any]
    operation: str
    intermediate_result: dict[str, Any]
    invariants_checked: list[str] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=lambda: {"passed": False, "violations": []})
    timestamp: float = field(default_factory=lambda: time.time())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TraceStep":
        payload = _ensure_dict(payload, "trace step")
        verification = payload.get("verification") or {}
        if not isinstance(verification, dict):
            raise TraceValidationError("verification must be a mapping.")
        return cls(
            step_id=int(payload["step_id"]),
            rule=str(payload["rule"]),
            equation=str(payload["equation"]),
            inputs=_ensure_dict(payload.get("inputs", {}), "inputs"),
            operation=str(payload["operation"]),
            intermediate_result=_ensure_dict(payload.get("intermediate_result", {}), "intermediate_result"),
            invariants_checked=[str(item) for item in _ensure_list(payload.get("invariants_checked", []), "invariants_checked")],
            verification={
                "passed": bool(verification.get("passed", False)),
                "violations": [_normalize(item) for item in _ensure_list(verification.get("violations", []), "verification.violations")],
            },
            timestamp=float(payload.get("timestamp", time.time())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "rule": self.rule,
            "equation": self.equation,
            "inputs": _normalize(self.inputs),
            "operation": self.operation,
            "intermediate_result": _normalize(self.intermediate_result),
            "invariants_checked": list(self.invariants_checked),
            "verification": _normalize(self.verification),
            "timestamp": float(self.timestamp),
        }

    def validate(self) -> None:
        if not isinstance(self.step_id, int):
            raise TraceValidationError("step_id must be an integer.")
        if self.step_id < 0:
            raise TraceValidationError("step_id must be non-negative.")
        for field_name, value in (
            ("rule", self.rule),
            ("equation", self.equation),
            ("operation", self.operation),
        ):
            if not isinstance(value, str) or not value.strip():
                raise TraceValidationError(f"{field_name} must be a non-empty string.")
        if not isinstance(self.inputs, dict):
            raise TraceValidationError("inputs must be a mapping.")
        if not isinstance(self.intermediate_result, dict):
            raise TraceValidationError("intermediate_result must be a mapping.")
        if not isinstance(self.invariants_checked, list):
            raise TraceValidationError("invariants_checked must be a list.")
        if not isinstance(self.verification, dict):
            raise TraceValidationError("verification must be a mapping.")
        if "passed" not in self.verification or "violations" not in self.verification:
            raise TraceValidationError("verification must contain passed and violations.")
        if not isinstance(self.verification.get("violations"), list):
            raise TraceValidationError("verification.violations must be a list.")
        if not isinstance(self.timestamp, (int, float)):
            raise TraceValidationError("timestamp must be numeric.")


@dataclass
class TraceReplayResult:
    """Result of replaying a structured trace."""

    final_state: dict[str, Any]
    states: list[dict[str, Any]]
    passed: bool
    violations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_state": _normalize(self.final_state),
            "states": _normalize(self.states),
            "passed": self.passed,
            "violations": _normalize(self.violations),
        }


@dataclass
class ReasoningTrace:
    """Canonical structured trace with deterministic ordering."""

    steps: list[TraceStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReasoningTrace":
        payload = _ensure_dict(payload, "trace")
        steps = [TraceStep.from_dict(step) for step in _ensure_list(payload.get("steps", []), "steps")]
        metadata = _ensure_dict(payload.get("metadata", {}), "metadata")
        return cls(steps=steps, metadata=metadata)

    @classmethod
    def from_json(cls, payload: str) -> "ReasoningTrace":
        return cls.from_dict(json.loads(payload))

    def ordered_steps(self) -> list[TraceStep]:
        return sorted(self.steps, key=lambda step: (step.step_id, float(step.timestamp), step.rule, step.equation))

    def validate(self) -> None:
        ordered = self.ordered_steps()
        seen_ids: set[int] = set()
        previous_id: int | None = None
        for step in ordered:
            step.validate()
            if step.step_id in seen_ids:
                raise TraceValidationError(f"Duplicate step_id detected: {step.step_id}")
            seen_ids.add(step.step_id)
            if previous_id is not None and step.step_id < previous_id:
                raise TraceValidationError("Trace steps must be deterministically ordered by step_id.")
            previous_id = step.step_id

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "steps": [step.to_dict() for step in self.ordered_steps()],
            "metadata": _normalize(self.metadata),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def replay(self, initial_state: dict[str, Any] | None = None) -> TraceReplayResult:
        """Replay the trace by applying each recorded step deterministically.

        The replay does not execute Lua. It verifies the recorded before/after
        chain and returns the final state observed in the trace.
        """

        ordered = self.ordered_steps()
        current = dict(initial_state or {})
        states: list[dict[str, Any]] = [dict(current)]
        violations: list[dict[str, Any]] = []

        for step in ordered:
            before = _normalize(step.inputs.get("before", current)) if isinstance(step.inputs, dict) else dict(current)
            after = _normalize(step.intermediate_result.get("after", step.intermediate_result))
            if before != _normalize(current):
                violations.append(
                    {
                        "step_id": step.step_id,
                        "reason": "step input state does not match replay state",
                    }
                )
            current = dict(after) if isinstance(after, dict) else dict(current)
            states.append(dict(current))

        passed = len(violations) == 0
        return TraceReplayResult(final_state=dict(current), states=states, passed=passed, violations=violations)


def canonicalize_trace(trace: dict[str, Any] | ReasoningTrace) -> ReasoningTrace:
    if isinstance(trace, ReasoningTrace):
        return trace
    return ReasoningTrace.from_dict(trace)


def replay_trace(trace: dict[str, Any] | ReasoningTrace, initial_state: dict[str, Any] | None = None) -> TraceReplayResult:
    return canonicalize_trace(trace).replay(initial_state=initial_state)

