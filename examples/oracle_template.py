"""CPT Oracle SDK Example — Ohm's Law Oracle.

A minimal working oracle that demonstrates:
1. Implementing OracleProtocol
2. Integrating with RuntimeExecutor
3. Deterministic execution

This is the reference implementation for external contributors who want
to add new domains (KiCad, FreeCAD, symbolic math, etc.) to the CPT
runtime.

Usage:
    python examples/oracle_template.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
from backend.core_runtime.task_runtime import RuntimeTask, RuntimeExecutor, OracleProtocol, SurrogateProtocol


# ---------------------------------------------------------------------------
# ExampleOhmsLawOracle — implements OracleProtocol
# ---------------------------------------------------------------------------

class ExampleOhmsLawOracle:
    """Minimal oracle that solves V=IR for a single resistor circuit.

    This demonstrates the OracleProtocol interface:
    - solve(graph) -> dict with 'voltages' key
    - name() -> str
    """

    def solve(self, task_or_graph) -> dict:
        """Solve a simple V=IR circuit.

        Accepts either:
        - A RuntimeTask (the executor passes the task object)
        - A dict with voltage_source_v, resistance_ohm, nodes

        Returns:
        {
            "voltages": torch.Tensor,  # Node voltages (source, junction, gnd)
            "current_a": float,        # Branch current
            "oracle_name": str,
            "latency_ms": float,
        }
        """
        import time

        # Extract params from RuntimeTask.metadata or dict
        if isinstance(task_or_graph, dict):
            graph = task_or_graph
        else:
            graph = getattr(task_or_graph, "metadata", {})

        t0 = time.perf_counter()
        vs = graph.get("voltage_source_v", 10.0)
        r = graph.get("resistance_ohm", 1000.0)
        nodes = graph.get("nodes", 3)

        # Solve: I = V/R, node voltages
        current = vs / r
        voltages = torch.zeros(nodes, dtype=torch.float32)
        if nodes >= 2:
            voltages[0] = vs       # Source node
            voltages[1] = vs       # Through R1 (direct)
        # Ground stays 0.0

        latency = (time.perf_counter() - t0) * 1000.0

        return {
            "voltages": voltages,
            "current_a": current,
            "oracle_name": self.name(),
            "latency_ms": latency,
        }

    def name(self) -> str:
        return "example_ohms_law"


# ---------------------------------------------------------------------------
# ExampleZeroSurrogate — implements SurrogateProtocol
# ---------------------------------------------------------------------------

class ExampleZeroSurrogate:
    """Zero-baseline surrogate for demonstration."""

    def predict(self, task_or_graph) -> torch.Tensor:
        nodes = 3
        if isinstance(task_or_graph, dict):
            nodes = task_or_graph.get("nodes", 3)
        else:
            meta = getattr(task_or_graph, "metadata", {})
            nodes = meta.get("nodes", 3)
        return torch.zeros(nodes, dtype=torch.float32)

    def name(self) -> str:
        return "example_zero_surrogate"


# ---------------------------------------------------------------------------
# Demo: run the oracle through the CPT runtime
# ---------------------------------------------------------------------------

def main() -> None:
    oracle = ExampleOhmsLawOracle()
    surrogate = ExampleZeroSurrogate()

    executor = RuntimeExecutor(
        oracle=oracle,
        surrogate=surrogate,
        projection=None,
        evaluator=None,
        memory_sink=None,
    )

    task = RuntimeTask(
        task_id="demo_001",
        domain="example",
        input_artifact="ohms_law_simple",
        oracle_name="example_ohms_law",
        surrogate_name="example_zero_surrogate",
        projection_enabled=False,
        metadata={"voltage_source_v": 10.0, "resistance_ohm": 1000.0, "nodes": 3},
    )

    result = executor.execute(task)

    print("=" * 60)
    print("CPT Oracle SDK — ExampleOhmsLawOracle Demo")
    print("=" * 60)
    print(f"Task ID:         {result.task_id}")
    print(f"Task fingerprint: {result.task_fingerprint[:16]}...")
    print(f"Oracle voltages: {result.oracle_voltages.tolist()}")
    print(f"Surrogate output: {result.surrogate_voltages.tolist()}")
    print(f"Oracle runtime:   {result.oracle_runtime_ms:.2f} ms")
    print(f"Total runtime:    {result.total_runtime_ms:.2f} ms")
    print("=" * 60)
    print("\nTo create your own oracle:")
    print("1. Implement solve(graph) -> dict with 'voltages' key")
    print("2. Implement name() -> str")
    print("3. Pass to RuntimeExecutor(oracle=YourOracle(), ...)")
    print("See: docs/ORACLE_SDK_GUIDE.md")


if __name__ == "__main__":
    main()
