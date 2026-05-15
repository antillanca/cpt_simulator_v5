"""Basic logical invariants."""

from __future__ import annotations

from math import isfinite


def _iterate_states(trace):
    if isinstance(trace, dict) and "steps" in trace:
        for step in trace["steps"]:
            yield step.get("before", {})
            yield step.get("after", {})
        return
    if isinstance(trace, dict):
        yield trace.get("initial_state", {})
        yield trace.get("final_state", trace.get("state", {}))
        return
    if isinstance(trace, list):
        yield from trace


def logic_basic(trace):
    violations = []
    checks = 0
    for state in _iterate_states(trace):
        checks += 1
        for key, value in (state or {}).items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                if not isfinite(float(value)):
                    violations.append({"reason": f"non-finite numeric value for {key}", "key": key})
            elif value is None:
                violations.append({"reason": f"null value for {key}", "key": key})

    passed = len(violations) == 0
    return {
        "passed": passed,
        "violations": violations,
        "metrics": {"logic_checked_states": checks, "logic_violation_count": len(violations)},
    }

