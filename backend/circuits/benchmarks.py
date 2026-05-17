"""Benchmark harness for CPT v2.8 Circuit Oracle Core.

Hand-crafted test circuits for benchmarking the solver, invariants,
and trace determinism.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.invariants import validate_invariants
from backend.circuits.models import Circuit, CircuitSolution
from backend.circuits.parser import parse_netlist
from backend.circuits.traces import generate_oracle_trace, trace_fingerprint


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    success: bool
    solve_time_ms: float
    invariant_passed: bool
    max_error: float
    trace_deterministic: bool
    details: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "success": self.success,
            "solve_time_ms": self.solve_time_ms,
            "invariant_passed": self.invariant_passed,
            "max_error": self.max_error,
            "trace_deterministic": self.trace_deterministic,
            "details": self.details,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    total: int
    passed: int
    failed: int
    invariant_pass_rate: float
    trace_determinism_rate: float
    avg_solve_time_ms: float
    results: Tuple[BenchmarkResult, ...]

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "invariant_pass_rate": self.invariant_pass_rate,
            "trace_determinism_rate": self.trace_determinism_rate,
            "avg_solve_time_ms": self.avg_solve_time_ms,
            "results": [r.to_dict() for r in self.results],
        }

    def to_markdown(self) -> str:
        lines = [
            "# CPT v2.8 Circuit Oracle Core — Benchmark Report",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total circuits | {self.total} |",
            f"| Passed | {self.passed} |",
            f"| Failed | {self.failed} |",
            f"| Invariant pass rate | {self.invariant_pass_rate:.1%} |",
            f"| Trace determinism rate | {self.trace_determinism_rate:.1%} |",
            f"| Avg solve time | {self.avg_solve_time_ms:.3f} ms |",
            "",
            "## Per-Circuit Results",
            "",
            "| Circuit | Success | Invariants | Trace Det. | Solve ms | Details |",
            "|---------|---------|------------|------------|----------|---------|",
        ]
        for r in self.results:
            inv = "PASS" if r.invariant_passed else "FAIL"
            det = "YES" if r.trace_deterministic else "NO"
            ok = "OK" if r.success else "FAIL"
            lines.append(f"| {r.name} | {ok} | {inv} | {det} | {r.solve_time_ms:.3f} | {r.details} |")
        lines.append("")
        return "\n".join(lines)


# === Hand-crafted benchmark circuits ===

BENCHMARK_CIRCUITS: List[Tuple[str, str]] = [
    # 1. Simple voltage divider
    (
        "voltage_divider",
        "# Voltage divider\nV1 VIN 0 5\nR1 VIN N1 1000\nR2 N1 0 2000\n",
    ),
    # 2. Single resistor + voltage source
    (
        "single_resistor",
        "V1 A 0 10\nR1 A 0 1000\n",
    ),
    # 3. Three-resistor series
    (
        "series_3r",
        "V1 A 0 12\nR1 A B 100\nR2 B C 200\nR3 C 0 300\n",
    ),
    # 4. Parallel resistors
    (
        "parallel_2r",
        "V1 A 0 9\nR1 A 0 1000\nR2 A 0 2000\n",
    ),
    # 5. Current source + resistor
    (
        "current_source_simple",
        "I1 A 0 0.01\nR1 A 0 1000\n",
    ),
    # 6. Voltage divider with 3 resistors
    (
        "divider_3r",
        "V1 IN 0 15\nR1 IN A 1000\nR2 A B 2000\nR3 B 0 3000\n",
    ),
    # 7. Two voltage sources
    (
        "two_vsources",
        "V1 A 0 5\nV2 B 0 3\nR1 A B 1000\n",
    ),
    # 8. Wheatstone bridge
    (
        "wheatstone",
        "V1 A 0 10\nR1 A B 100\nR2 A C 200\nR3 B D 300\nR4 C D 400\nR5 B C 1000\n",
    ),
    # 9. Current source with parallel resistor
    (
        "current_parallel",
        "I1 A 0 0.005\nR1 A 0 500\nR2 A 0 1500\n",
    ),
    # 10. Ladder network
    (
        "ladder_4r",
        "V1 IN 0 20\nR1 IN N1 100\nR2 N1 N2 200\nR3 N2 N3 300\nR4 N3 0 400\n",
    ),
    # 11. Mixed sources
    (
        "mixed_sources",
        "V1 A 0 6\nI1 A B 0.002\nR1 A B 500\nR2 B 0 1000\n",
    ),
    # 12. Ground alias GND
    (
        "gnd_alias",
        "V1 VIN GND 3.3\nR1 VIN OUT 100\nR2 OUT GND 200\n",
    ),
]


def run_benchmark(circuit_name: str, netlist_text: str) -> BenchmarkResult:
    """Run a single benchmark: parse, solve, validate invariants, check trace determinism."""
    try:
        circuit = parse_netlist(netlist_text, name=circuit_name)

        t0 = time.perf_counter()
        solution = solve_dc_circuit(circuit)
        t1 = time.perf_counter()
        solve_ms = (t1 - t0) * 1000.0

        inv = validate_invariants(circuit, solution)

        # Trace determinism: generate twice, compare fingerprints
        trace1 = generate_oracle_trace(circuit, solution)
        trace2 = generate_oracle_trace(circuit, solution)
        fp1 = trace_fingerprint(trace1)
        fp2 = trace_fingerprint(trace2)
        det = fp1 == fp2

        success = inv.passed and det
        details = f"invariants={'PASS' if inv.passed else 'FAIL'}, trace_det={'YES' if det else 'NO'}"

        return BenchmarkResult(
            name=circuit_name,
            success=success,
            solve_time_ms=round(solve_ms, 4),
            invariant_passed=inv.passed,
            max_error=inv.max_error,
            trace_deterministic=det,
            details=details,
        )
    except Exception as exc:
        return BenchmarkResult(
            name=circuit_name,
            success=False,
            solve_time_ms=0.0,
            invariant_passed=False,
            max_error=float("inf"),
            trace_deterministic=False,
            details=f"ERROR: {exc}",
        )


def run_all_benchmarks() -> BenchmarkReport:
    """Run all benchmark circuits and produce a report."""
    results: List[BenchmarkResult] = []
    for name, netlist in BENCHMARK_CIRCUITS:
        results.append(run_benchmark(name, netlist))

    total = len(results)
    passed = sum(1 for r in results if r.success)
    inv_passed = sum(1 for r in results if r.invariant_passed)
    det_ok = sum(1 for r in results if r.trace_deterministic)
    avg_ms = sum(r.solve_time_ms for r in results) / max(total, 1)

    return BenchmarkReport(
        total=total,
        passed=passed,
        failed=total - passed,
        invariant_pass_rate=inv_passed / max(total, 1),
        trace_determinism_rate=det_ok / max(total, 1),
        avg_solve_time_ms=round(avg_ms, 4),
        results=tuple(sorted(results, key=lambda r: r.name)),
    )
