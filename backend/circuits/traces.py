"""Oracle trace generation for CPT v2.8 Circuit Oracle Core.

Generates a deterministic step-by-step trace that mirrors the MNA solution
process. Each step is a dictionary with 'action' and relevant values.
The trace is fingerprintable (SHA-256 of canonical JSON).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple

from backend.circuits.models import Circuit, CircuitSolution

_ROUND = 9


def _sr(v: float) -> float:
    """Stable round."""
    return round(float(v), _ROUND)


def generate_oracle_trace(circuit: Circuit, solution: CircuitSolution) -> Tuple[Dict[str, Any], ...]:
    """Generate a deterministic step-by-step oracle trace of the MNA solution.

    Steps:
      1. init: circuit metadata
      2. stamp_resistor: one per resistor, sorted
      3. stamp_voltage_source: one per VS, sorted
      4. stamp_current_source: one per CS, sorted
      5. solve_linear_system: matrix size and result
      6. compute_currents: resistor currents
      7. compute_power: resistor power dissipation
      8. summary: final voltages and invariants-ready data
    """
    steps: list[Dict[str, Any]] = []

    # Step 1: init
    steps.append({
        "action": "init",
        "circuit_name": circuit.name,
        "num_nodes": len(circuit.all_nodes),
        "num_resistors": len(circuit.resistors),
        "num_voltage_sources": len(circuit.voltage_sources),
        "num_current_sources": len(circuit.current_sources),
        "nodes": list(circuit.all_nodes),
        "ground_node": circuit.ground_node,
    })

    # Step 2: stamp resistors
    for r in circuit.resistors:
        g = 1.0 / r.resistance_ohm
        steps.append({
            "action": "stamp_resistor",
            "name": r.name,
            "node_a": r.node_a,
            "node_b": r.node_b,
            "resistance_ohm": r.resistance_ohm,
            "conductance_siemens": _sr(g),
        })

    # Step 3: stamp voltage sources
    for vs in circuit.voltage_sources:
        steps.append({
            "action": "stamp_voltage_source",
            "name": vs.name,
            "positive": vs.positive,
            "negative": vs.negative,
            "voltage": vs.voltage,
        })

    # Step 4: stamp current sources
    for cs in circuit.current_sources:
        steps.append({
            "action": "stamp_current_source",
            "name": cs.name,
            "positive": cs.positive,
            "negative": cs.negative,
            "current": cs.current,
        })

    # Step 5: solve
    steps.append({
        "action": "solve_linear_system",
        "matrix_size": len(circuit.all_nodes) + len(circuit.voltage_sources),
        "node_voltages": dict(sorted(solution.node_voltages.items())),
    })

    # Step 6: compute currents
    current_data: Dict[str, float] = {}
    for r in circuit.resistors:
        i_r = solution.branch_currents.get(r.name, 0.0)
        current_data[r.name] = _sr(i_r)
    for vs in circuit.voltage_sources:
        current_data[vs.name] = _sr(solution.branch_currents.get(vs.name, 0.0))
    for cs in circuit.current_sources:
        current_data[cs.name] = _sr(cs.current)
    steps.append({
        "action": "compute_currents",
        "branch_currents": dict(sorted(current_data.items())),
    })

    # Step 7: compute power
    power_data: Dict[str, float] = {}
    for r in circuit.resistors:
        power_data[r.name] = _sr(solution.power_dissipation.get(r.name, 0.0))
    steps.append({
        "action": "compute_power",
        "power_dissipation": dict(sorted(power_data.items())),
    })

    # Step 8: summary
    steps.append({
        "action": "summary",
        "total_power_dissipated": _sr(sum(solution.power_dissipation.values())),
        "all_node_voltages": dict(sorted(solution.node_voltages.items())),
    })

    return tuple(steps)


def trace_fingerprint(trace: Tuple[Dict[str, Any], ...]) -> str:
    """Compute SHA-256 fingerprint of a trace's canonical JSON."""
    canonical = json.dumps(
        [dict(sorted(step.items())) for step in trace],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
