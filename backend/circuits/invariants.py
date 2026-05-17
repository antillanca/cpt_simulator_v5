"""Invariant validation for CPT v2.8 Circuit Oracle Core.

Checks:
  1. KCL at every non-ground node (sum of currents = 0, tol 1e-9)
  2. KVL for every voltage source loop (V_source = V_pos - V_neg)
  3. Power conservation: total supplied = total dissipated (tol 1e-6)

All traversals are deterministic (sorted nodes, sorted components).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from backend.circuits.models import Circuit, CircuitSolution

KCL_TOLERANCE = 1e-9
POWER_TOLERANCE = 1e-6


@dataclass(frozen=True)
class InvariantResult:
    passed: bool
    max_error: float
    details: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "max_error": self.max_error,
            "details": list(self.details),
        }


def validate_invariants(circuit: Circuit, solution: CircuitSolution) -> InvariantResult:
    """Validate KCL, KVL, and power conservation invariants."""
    errors: list[str] = []
    max_err = 0.0

    # === KCL at every non-ground node ===
    all_nodes_with_ground = list(circuit.all_nodes) + ["0"]
    for node in sorted(all_nodes_with_ground):
        current_sum = 0.0

        # Resistor currents: current flows from node_a to node_b (I = (Va - Vb) / R)
        for r in circuit.resistors:
            if r.node_a == node:
                # Current leaving node_a through resistor
                current_sum -= solution.branch_currents.get(r.name, 0.0)
            elif r.node_b == node:
                # Current entering node_b through resistor
                current_sum += solution.branch_currents.get(r.name, 0.0)

        # Voltage source: MNA current variable = current flowing from positive to
        # negative *through the source*. At the positive node, this current leaves
        # the node (enters the source). At the negative node, it arrives.
        for vs in circuit.voltage_sources:
            i_vs = solution.branch_currents.get(vs.name, 0.0)
            if vs.positive == node:
                current_sum -= i_vs
            elif vs.negative == node:
                current_sum += i_vs

        # Current source: injects current at positive, extracts at negative
        for cs in circuit.current_sources:
            if cs.positive == node:
                current_sum += cs.current
            elif cs.negative == node:
                current_sum -= cs.current

        kcl_err = abs(current_sum)
        if kcl_err > max_err:
            max_err = kcl_err
        if kcl_err > KCL_TOLERANCE:
            errors.append(f"KCL violated at node {node}: sum={current_sum:.2e} (tol={KCL_TOLERANCE:.0e})")

    # === KVL: for each voltage source, V_pos - V_neg = V_source ===
    for vs in circuit.voltage_sources:
        v_pos = solution.node_voltages.get(vs.positive, 0.0)
        v_neg = solution.node_voltages.get(vs.negative, 0.0)
        v_diff = v_pos - v_neg
        kvl_err = abs(v_diff - vs.voltage)
        if kvl_err > max_err:
            max_err = kvl_err
        if kvl_err > KCL_TOLERANCE:
            errors.append(f"KVL violated for {vs.name}: V_diff={v_diff:.9f}, expected={vs.voltage:.9f}")

    # === Power conservation ===
    # Total power delivered by all sources must equal total power dissipated.
    # Using consistent sign convention from KCL (sum of currents at a node = 0):
    #   At any node: sum of currents entering from sources = sum leaving through loads.
    #   Power delivered = sum over source nodes of (V_node * I_entering_node).
    #
    # Voltage source: MNA current = current flowing from positive to negative
    # through the source. At the positive node this current LEAVES (→ enters
    # the source), so I_entering = -I_mna at positive node. At the negative
    # node, I_entering = +I_mna.
    #   P_delivered = V_pos * (-I_mna) + V_neg * (+I_mna) = (V_neg - V_pos) * I_mna
    #
    # Current source: current INJECTS at positive node, EXTRACTS at negative.
    #   P_delivered = V_pos * I_source + V_neg * (-I_source) = (V_pos - V_neg) * I_source
    p_delivered = 0.0
    for vs in circuit.voltage_sources:
        i_vs = solution.branch_currents.get(vs.name, 0.0)
        v_pos = solution.node_voltages.get(vs.positive, 0.0)
        v_neg = solution.node_voltages.get(vs.negative, 0.0)
        p_delivered += (v_neg - v_pos) * i_vs

    for cs in circuit.current_sources:
        v_pos = solution.node_voltages.get(cs.positive, 0.0)
        v_neg = solution.node_voltages.get(cs.negative, 0.0)
        p_delivered += (v_pos - v_neg) * cs.current

    # Power dissipated by resistors
    p_dissipated = sum(solution.power_dissipation.values())

    power_err = abs(p_delivered - p_dissipated)
    if power_err > max_err:
        max_err = power_err
    if power_err > POWER_TOLERANCE:
        errors.append(
            f"Power conservation violated: delivered={p_delivered:.9f}, dissipated={p_dissipated:.9f}, "
            f"diff={power_err:.2e} (tol={POWER_TOLERANCE:.0e})"
        )

    details = tuple(sorted(errors)) if errors else ("All invariants satisfied",)
    return InvariantResult(
        passed=len(errors) == 0,
        max_error=_round_err(max_err),
        details=details,
    )


def _round_err(v: float) -> float:
    return round(v, 12)
