"""Tests for CPT v2.8 Circuit Oracle Core.

Covers: parsing, solving, invariants, traces, dataset generation, benchmarks.
All tests deterministic — no random seeds in test logic.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from backend.circuits.benchmarks import run_all_benchmarks, run_benchmark, BENCHMARK_CIRCUITS
from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.invariants import validate_invariants
from backend.circuits.models import (
    Circuit,
    CircuitSolution,
    CurrentSource,
    Resistor,
    VoltageSource,
    _normalize_ground,
)
from backend.circuits.parser import parse_netlist
from backend.circuits.traces import generate_oracle_trace, trace_fingerprint


# ============================================================
# PHASE 1 — Domain Models
# ============================================================

class TestDomainModels:

    def test_resistor_frozen(self):
        r = Resistor(name="R1", node_a="A", node_b="0", resistance_ohm=1000)
        with pytest.raises(AttributeError):
            r.name = "R2"  # type: ignore[misc]

    def test_resistor_positive_resistance(self):
        with pytest.raises(ValueError, match="resistance must be > 0"):
            Resistor(name="R1", node_a="A", node_b="0", resistance_ohm=0)

    def test_resistor_negative_resistance(self):
        with pytest.raises(ValueError, match="resistance must be > 0"):
            Resistor(name="R1", node_a="A", node_b="0", resistance_ohm=-100)

    def test_voltage_source_frozen(self):
        v = VoltageSource(name="V1", positive="A", negative="0", voltage=5.0)
        with pytest.raises(AttributeError):
            v.voltage = 10.0  # type: ignore[misc]

    def test_current_source_zero_rejected(self):
        with pytest.raises(ValueError, match="non-zero"):
            CurrentSource(name="I1", positive="A", negative="0", current=0)

    def test_normalize_ground_zero(self):
        assert _normalize_ground("0") == "0"

    def test_normalize_ground_gnd(self):
        assert _normalize_ground("GND") == "0"
        assert _normalize_ground("gnd") == "0"

    def test_normalize_ground_other(self):
        assert _normalize_ground("VIN") == "VIN"

    def test_circuit_all_nodes_sorted(self):
        c = Circuit(
            resistors=(
                Resistor(name="R1", node_a="C", node_b="A", resistance_ohm=100),
                Resistor(name="R2", node_a="A", node_b="0", resistance_ohm=200),
            ),
            voltage_sources=(VoltageSource(name="V1", positive="C", negative="0", voltage=5),),
        )
        assert c.all_nodes == ("A", "C")

    def test_circuit_ground_normalized(self):
        c = Circuit(
            resistors=(Resistor(name="R1", node_a="VIN", node_b="GND", resistance_ohm=100),),
        )
        assert c.ground_node == "0"
        assert c.resistors[0].node_b == "0"

    def test_circuit_components_sorted_by_name(self):
        c = Circuit(
            resistors=(
                Resistor(name="R3", node_a="A", node_b="0", resistance_ohm=300),
                Resistor(name="R1", node_a="B", node_b="0", resistance_ohm=100),
                Resistor(name="R2", node_a="C", node_b="0", resistance_ohm=200),
            ),
        )
        assert c.resistors[0].name == "R1"
        assert c.resistors[1].name == "R2"
        assert c.resistors[2].name == "R3"


# ============================================================
# PHASE 2 — Parser
# ============================================================

class TestParser:

    def test_parse_voltage_divider(self):
        text = "# Voltage divider\nV1 VIN 0 5\nR1 VIN N1 1000\nR2 N1 0 2000\n"
        c = parse_netlist(text, name="vd")
        assert c.name == "vd"
        assert len(c.resistors) == 2
        assert len(c.voltage_sources) == 1
        assert c.resistors[0].name == "R1"
        assert c.resistors[0].resistance_ohm == 1000
        assert c.voltage_sources[0].voltage == 5.0

    def test_parse_gnd_normalized(self):
        text = "V1 VIN GND 3.3\nR1 VIN OUT 100\nR2 OUT GND 200\n"
        c = parse_netlist(text)
        assert c.ground_node == "0"
        assert c.voltage_sources[0].negative == "0"
        assert c.resistors[1].node_b == "0"

    def test_parse_current_source(self):
        text = "I1 A 0 0.01\nR1 A 0 1000\n"
        c = parse_netlist(text)
        assert len(c.current_sources) == 1
        assert c.current_sources[0].current == 0.01

    def test_parse_empty_lines_and_comments(self):
        text = "\n# comment\n\nV1 A 0 5\n\n# another comment\nR1 A 0 1000\n"
        c = parse_netlist(text)
        assert len(c.resistors) == 1
        assert len(c.voltage_sources) == 1

    def test_parse_rejects_unknown_prefix(self):
        with pytest.raises(ValueError, match="unknown component prefix"):
            parse_netlist("C1 A 0 0.001\n")

    def test_parse_rejects_malformed_line(self):
        with pytest.raises(ValueError, match="expected at least 4 tokens"):
            parse_netlist("R1 A\n")

    def test_parse_rejects_non_numeric(self):
        with pytest.raises(ValueError):
            parse_netlist("R1 A 0 hello\n")

    def test_parse_rejects_zero_resistance(self):
        with pytest.raises(ValueError, match="resistance must be > 0"):
            parse_netlist("R1 A 0 0\n")

    def test_parse_rejects_empty_netlist(self):
        with pytest.raises(ValueError, match="no components"):
            parse_netlist("# just a comment\n")

    def test_parse_deterministic_ordering(self):
        text = "R3 A 0 300\nR1 A 0 100\nR2 A 0 200\n"
        c = parse_netlist(text)
        names = [r.name for r in c.resistors]
        assert names == ["R1", "R2", "R3"]


# ============================================================
# PHASE 3 — Solver
# ============================================================

class TestSolver:

    def test_single_resistor_ohms_law(self):
        """V=10, R=1000 → I=0.01A, V_node=10"""
        c = Circuit(
            resistors=(Resistor(name="R1", node_a="A", node_b="0", resistance_ohm=1000),),
            voltage_sources=(VoltageSource(name="V1", positive="A", negative="0", voltage=10),),
        )
        sol = solve_dc_circuit(c)
        assert sol.node_voltages["A"] == 10.0
        assert sol.branch_currents["R1"] == 0.01
        assert sol.power_dissipation["R1"] == 0.1  # P = V*I = 10*0.01

    def test_voltage_divider(self):
        """V1=5, R1=1k, R2=2k → V_N1 = 5*2/3 ≈ 3.333..."""
        c = Circuit(
            resistors=(
                Resistor(name="R1", node_a="VIN", node_b="N1", resistance_ohm=1000),
                Resistor(name="R2", node_a="N1", node_b="0", resistance_ohm=2000),
            ),
            voltage_sources=(VoltageSource(name="V1", positive="VIN", negative="0", voltage=5),),
        )
        sol = solve_dc_circuit(c)
        assert sol.node_voltages["VIN"] == 5.0
        assert round(sol.node_voltages["N1"], 6) == round(5 * 2000 / 3000, 6)
        assert round(sol.branch_currents["R1"], 6) == round(5 / 3000, 6)

    def test_parallel_resistors(self):
        """V=9, R1=1k, R2=2k in parallel → I_total = 9/1000 + 9/2000"""
        c = Circuit(
            resistors=(
                Resistor(name="R1", node_a="A", node_b="0", resistance_ohm=1000),
                Resistor(name="R2", node_a="A", node_b="0", resistance_ohm=2000),
            ),
            voltage_sources=(VoltageSource(name="V1", positive="A", negative="0", voltage=9),),
        )
        sol = solve_dc_circuit(c)
        assert sol.node_voltages["A"] == 9.0
        assert round(sol.branch_currents["R1"], 6) == 0.009
        assert round(sol.branch_currents["R2"], 6) == 0.0045

    def test_current_source_simple(self):
        """I=0.01A, R=1000Ω → V=10V"""
        c = Circuit(
            resistors=(Resistor(name="R1", node_a="A", node_b="0", resistance_ohm=1000),),
            current_sources=(CurrentSource(name="I1", positive="A", negative="0", current=0.01),),
        )
        sol = solve_dc_circuit(c)
        assert round(sol.node_voltages["A"], 6) == 10.0
        assert round(sol.branch_currents["R1"], 6) == 0.01

    def test_series_resistors(self):
        """V=12, R1=100, R2=200, R3=300 → I=12/600=0.02"""
        c = Circuit(
            resistors=(
                Resistor(name="R1", node_a="A", node_b="B", resistance_ohm=100),
                Resistor(name="R2", node_a="B", node_b="C", resistance_ohm=200),
                Resistor(name="R3", node_a="C", node_b="0", resistance_ohm=300),
            ),
            voltage_sources=(VoltageSource(name="V1", positive="A", negative="0", voltage=12),),
        )
        sol = solve_dc_circuit(c)
        assert sol.node_voltages["A"] == 12.0
        assert round(sol.branch_currents["R1"], 6) == 0.02
        assert round(sol.node_voltages["B"], 6) == round(12 - 0.02 * 100, 6)  # 10.0
        assert round(sol.node_voltages["C"], 6) == round(12 - 0.02 * 300, 6)  # 6.0

    def test_deterministic_across_calls(self):
        """Same circuit → identical solution across two calls."""
        c = Circuit(
            resistors=(
                Resistor(name="R1", node_a="VIN", node_b="N1", resistance_ohm=1000),
                Resistor(name="R2", node_a="N1", node_b="0", resistance_ohm=2000),
            ),
            voltage_sources=(VoltageSource(name="V1", positive="VIN", negative="0", voltage=5),),
        )
        sol1 = solve_dc_circuit(c)
        sol2 = solve_dc_circuit(c)
        assert sol1.node_voltages == sol2.node_voltages
        assert sol1.branch_currents == sol2.branch_currents
        assert sol1.power_dissipation == sol2.power_dissipation


# ============================================================
# PHASE 4 — Invariants
# ============================================================

class TestInvariants:

    def test_kcl_voltage_divider(self):
        c = Circuit(
            resistors=(
                Resistor(name="R1", node_a="VIN", node_b="N1", resistance_ohm=1000),
                Resistor(name="R2", node_a="N1", node_b="0", resistance_ohm=2000),
            ),
            voltage_sources=(VoltageSource(name="V1", positive="VIN", negative="0", voltage=5),),
        )
        sol = solve_dc_circuit(c)
        inv = validate_invariants(c, sol)
        assert inv.passed, f"Invariants failed: {inv.details}"

    def test_kcl_parallel(self):
        c = Circuit(
            resistors=(
                Resistor(name="R1", node_a="A", node_b="0", resistance_ohm=1000),
                Resistor(name="R2", node_a="A", node_b="0", resistance_ohm=2000),
            ),
            voltage_sources=(VoltageSource(name="V1", positive="A", negative="0", voltage=9),),
        )
        sol = solve_dc_circuit(c)
        inv = validate_invariants(c, sol)
        assert inv.passed, f"Invariants failed: {inv.details}"

    def test_power_conservation(self):
        c = Circuit(
            resistors=(Resistor(name="R1", node_a="A", node_b="0", resistance_ohm=1000),),
            voltage_sources=(VoltageSource(name="V1", positive="A", negative="0", voltage=10),),
        )
        sol = solve_dc_circuit(c)
        inv = validate_invariants(c, sol)
        assert inv.passed
        # P_supplied = V*I = 10*0.01 = 0.1
        # P_dissipated = R*I^2 = 1000*0.0001 = 0.1
        assert round(sol.power_dissipation["R1"], 6) == 0.1

    def test_invariant_result_serializable(self):
        c = Circuit(
            resistors=(Resistor(name="R1", node_a="A", node_b="0", resistance_ohm=1000),),
            voltage_sources=(VoltageSource(name="V1", positive="A", negative="0", voltage=5),),
        )
        sol = solve_dc_circuit(c)
        inv = validate_invariants(c, sol)
        d = inv.to_dict()
        assert isinstance(d, dict)
        assert "passed" in d
        json.dumps(d)  # must be JSON-serializable


# ============================================================
# PHASE 5 — Traces
# ============================================================

class TestTraces:

    def test_trace_has_all_actions(self):
        c = Circuit(
            resistors=(Resistor(name="R1", node_a="A", node_b="0", resistance_ohm=1000),),
            voltage_sources=(VoltageSource(name="V1", positive="A", negative="0", voltage=5),),
        )
        sol = solve_dc_circuit(c)
        trace = generate_oracle_trace(c, sol)
        actions = [step["action"] for step in trace]
        assert "init" in actions
        assert "stamp_resistor" in actions
        assert "stamp_voltage_source" in actions
        assert "solve_linear_system" in actions
        assert "compute_currents" in actions
        assert "compute_power" in actions
        assert "summary" in actions

    def test_trace_deterministic(self):
        c = Circuit(
            resistors=(
                Resistor(name="R1", node_a="VIN", node_b="N1", resistance_ohm=1000),
                Resistor(name="R2", node_a="N1", node_b="0", resistance_ohm=2000),
            ),
            voltage_sources=(VoltageSource(name="V1", positive="VIN", negative="0", voltage=5),),
        )
        sol = solve_dc_circuit(c)
        t1 = generate_oracle_trace(c, sol)
        t2 = generate_oracle_trace(c, sol)
        assert t1 == t2

    def test_trace_fingerprint_stable(self):
        c = Circuit(
            resistors=(Resistor(name="R1", node_a="A", node_b="0", resistance_ohm=1000),),
            voltage_sources=(VoltageSource(name="V1", positive="A", negative="0", voltage=5),),
        )
        sol = solve_dc_circuit(c)
        trace = generate_oracle_trace(c, sol)
        fp1 = trace_fingerprint(trace)
        fp2 = trace_fingerprint(trace)
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex

    def test_trace_json_serializable(self):
        c = Circuit(
            resistors=(Resistor(name="R1", node_a="A", node_b="0", resistance_ohm=500),),
            voltage_sources=(VoltageSource(name="V1", positive="A", negative="0", voltage=3.3),),
        )
        sol = solve_dc_circuit(c)
        trace = generate_oracle_trace(c, sol)
        # Must be JSON-serializable
        result = json.dumps(list(trace), sort_keys=True)
        assert isinstance(result, str)


# ============================================================
# PHASE 6 — Dataset Generation
# ============================================================

class TestDatasetGeneration:

    def test_seed_reproducibility(self):
        from scripts.generate_circuit_dataset import generate_dataset
        d1 = generate_dataset(seed=42, num_circuits=20, include_benchmarks=False)
        d2 = generate_dataset(seed=42, num_circuits=20, include_benchmarks=False)
        # Must be identical row-by-row
        for r1, r2 in zip(d1, d2):
            assert r1["fingerprint"] == r2["fingerprint"]

    def test_different_seed_different_output(self):
        from scripts.generate_circuit_dataset import generate_dataset
        d1 = generate_dataset(seed=42, num_circuits=10, include_benchmarks=False)
        d2 = generate_dataset(seed=99, num_circuits=10, include_benchmarks=False)
        fps1 = [r["fingerprint"] for r in d1]
        fps2 = [r["fingerprint"] for r in d2]
        # At least some rows should differ
        assert fps1 != fps2

    def test_dataset_row_has_required_fields(self):
        from scripts.generate_circuit_dataset import generate_dataset
        rows = generate_dataset(seed=1, num_circuits=5, include_benchmarks=False)
        for row in rows:
            assert "id" in row
            assert "netlist" in row
            assert "solution" in row
            assert "trace" in row
            assert "invariants" in row
            assert "fingerprint" in row


# ============================================================
# PHASE 7 — Benchmarks
# ============================================================

class TestBenchmarks:

    def test_benchmark_report_generated(self):
        report = run_all_benchmarks()
        assert report.total >= 10
        assert report.passed + report.failed == report.total

    def test_benchmark_report_json(self):
        report = run_all_benchmarks()
        d = report.to_dict()
        data = json.dumps(d)
        assert isinstance(data, str)

    def test_benchmark_report_markdown(self):
        report = run_all_benchmarks()
        md = report.to_markdown()
        assert "# CPT v2.8" in md
        assert "| Circuit |" in md

    def test_single_benchmark_runs(self):
        result = run_benchmark("voltage_divider", BENCHMARK_CIRCUITS[0][1])
        assert result.name == "voltage_divider"
        assert result.trace_deterministic is True

    def test_all_benchmarks_invariant_pass(self):
        report = run_all_benchmarks()
        assert report.invariant_pass_rate == 1.0, f"Some invariants failed: {report.results}"


# ============================================================
# Integration
# ============================================================

class TestIntegration:

    def test_full_pipeline_voltage_divider(self):
        """Parse → Solve → Validate → Trace → Fingerprint."""
        netlist = "# Voltage divider\nV1 VIN 0 5\nR1 VIN N1 1000\nR2 N1 0 2000\n"
        circuit = parse_netlist(netlist, name="vd")
        sol = solve_dc_circuit(circuit)
        inv = validate_invariants(circuit, sol)
        trace = generate_oracle_trace(circuit, sol)
        fp = trace_fingerprint(trace)

        assert inv.passed, f"Invariants: {inv.details}"
        assert len(fp) == 64
        assert round(sol.node_voltages["N1"], 4) == round(5 * 2000 / 3000, 4)

    def test_full_pipeline_current_source(self):
        """Full pipeline with current source."""
        netlist = "I1 A 0 0.005\nR1 A 0 500\nR2 A 0 1500\n"
        circuit = parse_netlist(netlist, name="cs")
        sol = solve_dc_circuit(circuit)
        inv = validate_invariants(circuit, sol)
        trace = generate_oracle_trace(circuit, sol)

        assert inv.passed, f"Invariants: {inv.details}"
        # Parallel R: 1/(1/500 + 1/1500) = 375 ohm
        # V = 0.005 * 375 = 1.875
        assert round(sol.node_voltages["A"], 4) == 1.875

    def test_wheatstone_bridge_pipeline(self):
        """Wheatstone bridge: parse, solve, validate."""
        netlist = (
            "V1 A 0 10\n"
            "R1 A B 100\n"
            "R2 A C 200\n"
            "R3 B D 300\n"
            "R4 C D 400\n"
            "R5 B C 1000\n"
        )
        circuit = parse_netlist(netlist, name="wheatstone")
        sol = solve_dc_circuit(circuit)
        inv = validate_invariants(circuit, sol)
        trace = generate_oracle_trace(circuit, sol)

        assert inv.passed, f"Invariants: {inv.details}"
        assert trace_fingerprint(trace) == trace_fingerprint(trace)

    def test_solution_serialization_roundtrip(self):
        c = Circuit(
            resistors=(Resistor(name="R1", node_a="A", node_b="0", resistance_ohm=1000),),
            voltage_sources=(VoltageSource(name="V1", positive="A", negative="0", voltage=5),),
        )
        sol = solve_dc_circuit(c)
        d = sol.to_dict()
        json_str = json.dumps(d, sort_keys=True)
        loaded = json.loads(json_str)
        assert loaded["node_voltages"]["A"] == 5.0
