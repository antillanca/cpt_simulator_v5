"""Deterministic DC solver using Modified Nodal Analysis (MNA).

Solves linear resistive circuits with independent DC sources.
Uses numpy for matrix operations. All operations are deterministic:
nodes in alphabetical order, components sorted by name before stamping,
stable rounding to 9 decimal places.
"""

from __future__ import annotations

import numpy as np
from backend.circuits.models import Circuit, CircuitSolution

# Stable rounding to kill floating-point noise
_ROUND = 9


def _stable_round(value: float) -> float:
    return round(float(value), _ROUND)


def solve_dc_circuit(circuit: Circuit) -> CircuitSolution:
    """Solve a linear DC circuit using MNA. Returns deterministic CircuitSolution.

    MNA formulation:
      [G  B] [v]   [i]
      [C  D] [j] = [e]

    Where:
      v = node voltages (excluding ground)
      j = voltage source currents
      G = conductance matrix
      B/C = voltage source connectivity
      D = zero matrix
      i = current source contributions
      e = voltage source values
    """
    nodes = list(circuit.all_nodes)  # already sorted, ground excluded
    n = len(nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}

    m = len(circuit.voltage_sources)
    vs_index = {vs.name: idx for idx, vs in enumerate(circuit.voltage_sources)}

    size = n + m
    A = np.zeros((size, size), dtype=np.float64)
    b = np.zeros(size, dtype=np.float64)

    # Stamp resistors (sorted by name — circuit already ensures this)
    for r in circuit.resistors:
        g = 1.0 / r.resistance_ohm
        a_idx = node_index.get(r.node_a)
        b_idx = node_index.get(r.node_b)

        if a_idx is not None:
            A[a_idx, a_idx] += g
        if b_idx is not None:
            A[b_idx, b_idx] += g
        if a_idx is not None and b_idx is not None:
            A[a_idx, b_idx] -= g
            A[b_idx, a_idx] -= g
        elif a_idx is not None:
            # b is ground — already handled by not adding off-diagonal
            pass
        elif b_idx is not None:
            # a is ground
            pass

    # Stamp voltage sources
    for vs in circuit.voltage_sources:
        j_idx = n + vs_index[vs.name]
        pos_idx = node_index.get(vs.positive)
        neg_idx = node_index.get(vs.negative)

        if pos_idx is not None:
            A[pos_idx, j_idx] += 1.0
            A[j_idx, pos_idx] += 1.0
        if neg_idx is not None:
            A[neg_idx, j_idx] -= 1.0
            A[j_idx, neg_idx] -= 1.0

        b[j_idx] = vs.voltage

    # Stamp current sources
    for cs in circuit.current_sources:
        # Current flows from positive to negative through external circuit
        # So current enters the positive node from the source
        pos_idx = node_index.get(cs.positive)
        neg_idx = node_index.get(cs.negative)

        if pos_idx is not None:
            b[pos_idx] += cs.current
        if neg_idx is not None:
            b[neg_idx] -= cs.current

    # Solve the system
    x = np.linalg.solve(A, b)

    # Extract node voltages
    node_voltages: dict[str, float] = {"0": 0.0}
    for node in nodes:
        idx = node_index[node]
        node_voltages[node] = _stable_round(x[idx])

    # Extract voltage source currents
    branch_currents: dict[str, float] = {}
    for vs in circuit.voltage_sources:
        j_idx = n + vs_index[vs.name]
        branch_currents[vs.name] = _stable_round(x[j_idx])

    # Compute resistor currents and power
    power_dissipation: dict[str, float] = {}
    for r in circuit.resistors:
        va = node_voltages.get(r.node_a, 0.0)
        vb = node_voltages.get(r.node_b, 0.0)
        v_drop = va - vb
        current = v_drop / r.resistance_ohm
        branch_currents[r.name] = _stable_round(current)
        power = v_drop * current
        power_dissipation[r.name] = _stable_round(power)

    # Compute current source "current" — the current through the source
    # For a current source, the current is just the specified value
    for cs in circuit.current_sources:
        branch_currents[cs.name] = _stable_round(cs.current)

    return CircuitSolution(
        node_voltages=dict(sorted(node_voltages.items())),
        branch_currents=dict(sorted(branch_currents.items())),
        power_dissipation=dict(sorted(power_dissipation.items())),
    )
