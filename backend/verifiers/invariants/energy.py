"""Energy invariants."""

from __future__ import annotations

from math import isfinite


def _extract_states(trace):
    if isinstance(trace, dict):
        if "steps" in trace and trace["steps"]:
            steps = trace["steps"]
            first = steps[0].get("before") or trace.get("initial_state") or {}
            last = steps[-1].get("after") or trace.get("final_state") or {}
            return first, last
        return trace.get("initial_state", {}), trace.get("final_state", trace.get("state", {}))
    if isinstance(trace, list) and trace:
        return trace[0], trace[-1]
    return {}, {}


def _kinetic_energy(state):
    mass = float(state.get("mass", 1.0) or 1.0)
    vx = float(state.get("vx", 0.0) or 0.0)
    vy = float(state.get("vy", 0.0) or 0.0)
    return 0.5 * mass * (vx * vx + vy * vy)


def energy_conservation(trace):
    initial, final = _extract_states(trace)
    ei = _kinetic_energy(initial)
    ef = _kinetic_energy(final)
    drift = abs(ef - ei)
    baseline = max(abs(ei), 1e-9)
    relative_drift = drift / baseline
    passed = isfinite(ei) and isfinite(ef) and relative_drift <= 0.05
    return {
        "passed": passed,
        "violations": [] if passed else [{"reason": "energy drift exceeds tolerance", "drift": relative_drift}],
        "metrics": {"energy_initial": ei, "energy_final": ef, "energy_drift": relative_drift},
    }

