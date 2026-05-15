"""Momentum invariants."""

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


def _momentum(state):
    mass = float(state.get("mass", 1.0) or 1.0)
    vx = float(state.get("vx", 0.0) or 0.0)
    vy = float(state.get("vy", 0.0) or 0.0)
    return mass * vx, mass * vy


def momentum_conservation(trace):
    initial, final = _extract_states(trace)
    px0, py0 = _momentum(initial)
    pxf, pyf = _momentum(final)
    dx = abs(pxf - px0)
    dy = abs(pyf - py0)
    scale = max(abs(px0) + abs(py0), 1e-9)
    drift = (dx + dy) / scale
    passed = isfinite(px0) and isfinite(py0) and isfinite(pxf) and isfinite(pyf) and drift <= 0.05
    return {
        "passed": passed,
        "violations": [] if passed else [{"reason": "momentum drift exceeds tolerance", "drift": drift}],
        "metrics": {"momentum_initial": [px0, py0], "momentum_final": [pxf, pyf], "momentum_drift": drift},
    }

