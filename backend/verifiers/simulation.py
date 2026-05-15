"""Simulation trace verification entry point."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from backend.verifiers.invariants import get_invariant


def verify_simulation(trace, invariant_set) -> dict:
    """Verify a simulation trace against a set of invariant names."""

    violations = []
    metrics = {}
    passed = True

    names = list(invariant_set or [])
    for name in names:
        invariant = get_invariant(name)
        if invariant is None:
            violations.append({"invariant": name, "reason": "unknown invariant"})
            passed = False
            continue

        result = invariant(trace)
        metrics.update(result.get("metrics", {}))
        if not result.get("passed", False):
            passed = False
            for violation in result.get("violations", []):
                violations.append({"invariant": name, **violation})

    return {"passed": passed, "violations": violations, "metrics": metrics}

