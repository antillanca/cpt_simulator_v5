"""CPT Core Runtime — Oracle Protocol & MNA Adapter.

Defines OracleProtocol (generic) and MNAOracleAdapter (circuit domain).
Future domains (symbolic math, logic, geometry) implement OracleProtocol.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.models import Circuit, CircuitSolution


# ---------------------------------------------------------------------------
# OracleProtocol — re-exported from task_runtime for convenience
# ---------------------------------------------------------------------------

@runtime_checkable
class OracleProtocol(Protocol):
    """Any oracle that solves a domain problem exactly."""

    def solve(self, task_or_graph: Any) -> dict[str, Any]:
        ...

    def name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# MNAOracleAdapter — circuit-domain oracle
# ---------------------------------------------------------------------------

class MNAOracleAdapter:
    """Adapts the existing MNA DC solver to OracleProtocol.

    solve() accepts either:
      - A RuntimeTask (uses input_artifact to look up circuit)
      - A Circuit object directly
      - A dict with 'circuit' key
    Returns dict with 'voltages', 'solution', 'latency_ms'.
    """

    def __init__(self, circuit_lookup: dict[str, Circuit] | None = None) -> None:
        """circuit_lookup: mapping from artifact fingerprint → Circuit."""
        self._circuit_lookup = circuit_lookup or {}

    def register_circuit(self, fingerprint: str, circuit: Circuit) -> None:
        self._circuit_lookup[fingerprint] = circuit

    def solve(self, task_or_graph: Any) -> dict[str, Any]:
        """Solve circuit via MNA. Returns deterministic solution dict."""
        circuit = self._resolve_circuit(task_or_graph)
        t0 = time.perf_counter()
        solution: CircuitSolution = solve_dc_circuit(circuit)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Convert to ordered voltage list (sorted by node name for determinism)
        node_order = sorted(solution.node_voltages.keys())
        voltages = [solution.node_voltages[n] for n in node_order]

        import torch
        return {
            "voltages": torch.tensor(voltages, dtype=torch.float32),
            "solution": solution,
            "node_order": node_order,
            "latency_ms": latency_ms,
            "oracle_name": self.name(),
        }

    def name(self) -> str:
        return "mna_dc_solver"

    # -- internal --

    def _resolve_circuit(self, task_or_graph: Any) -> Circuit:
        if isinstance(task_or_graph, Circuit):
            return task_or_graph
        if isinstance(task_or_graph, dict) and "circuit" in task_or_graph:
            return task_or_graph["circuit"]
        # RuntimeTask has input_artifact
        if hasattr(task_or_graph, "input_artifact"):
            fp = task_or_graph.input_artifact
            if fp in self._circuit_lookup:
                return self._circuit_lookup[fp]
        raise ValueError(f"Cannot resolve circuit from {type(task_or_graph)}")
