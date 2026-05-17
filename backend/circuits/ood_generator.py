"""Out-of-distribution circuit generator for robustness testing.

Generates circuits with extreme values, unusual topologies, and
edge cases. All generation is deterministic (seed-based).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.graph_dataset import CircuitGraph, circuit_to_graph
from backend.circuits.invariants import validate_invariants
from backend.circuits.models import Circuit, CurrentSource, Resistor, VoltageSource
from backend.circuits.parser import parse_netlist
from backend.circuits.traces import generate_oracle_trace


def _extreme_resistor_circuit(rng: random.Random, idx: int) -> Tuple[str, str]:
    """Circuit with extreme resistor values (1e-6 to 1e9 ohm)."""
    n_nodes = rng.randint(2, 4)
    nodes = ["0"] + [f"N{i}" for i in range(1, n_nodes)]

    lines = [f"# OOD extreme resistor {idx}"]
    # Voltage source
    pos = rng.choice(nodes[1:])
    v_val = rng.choice([1, 5, 10, 100])
    lines.append(f"V1 {pos} 0 {v_val}")

    # Resistors with extreme values
    seen = set()
    for ri in range(rng.randint(2, 4)):
        for _ in range(50):
            a = rng.choice(nodes[1:])
            b = rng.choice(nodes)
            if a == b:
                continue
            pair = tuple(sorted([a, b]))
            if pair not in seen:
                seen.add(pair)
                break
        # Log-uniform from 1e-6 to 1e9
        exp = rng.uniform(-6, 9)
        r_val = 10.0 ** exp
        lines.append(f"R{ri+1} {a} {b} {r_val:.10g}")

    return f"ood_extreme_r_{idx}", "\n".join(lines) + "\n"


def _high_voltage_circuit(rng: random.Random, idx: int) -> Tuple[str, str]:
    """Circuit with high voltage values (up to 1e4 V)."""
    n_nodes = rng.randint(2, 5)
    nodes = ["0"] + [f"N{i}" for i in range(1, n_nodes)]

    lines = [f"# OOD high voltage {idx}"]
    # High voltage source
    pos = rng.choice(nodes[1:])
    v_val = rng.choice([100, 500, 1000, 5000, 10000])
    lines.append(f"V1 {pos} 0 {v_val}")

    # Normal resistors
    seen = set()
    for ri in range(rng.randint(1, 5)):
        for _ in range(50):
            a = rng.choice(nodes[1:])
            b = rng.choice(nodes)
            if a == b:
                continue
            pair = tuple(sorted([a, b]))
            if pair not in seen:
                seen.add(pair)
                break
        r_val = rng.choice([100, 500, 1000, 5000, 10000, 50000, 100000])
        lines.append(f"R{ri+1} {a} {b} {r_val}")

    return f"ood_high_v_{idx}", "\n".join(lines) + "\n"


def _star_topology_circuit(rng: random.Random, idx: int) -> Tuple[str, str]:
    """Star topology: one central node connected to all others."""
    n_spokes = rng.randint(3, 8)
    lines = [f"# OOD star topology {idx}"]

    # Voltage at center
    v_val = rng.choice([5, 10, 12, 24])
    lines.append(f"V1 N0 0 {v_val}")

    # Resistors from center to each spoke
    for i in range(n_spokes):
        r_val = rng.choice([100, 500, 1000, 2000, 5000, 10000])
        lines.append(f"R{i+1} N0 N{i+1} {r_val}")

    return f"ood_star_{idx}", "\n".join(lines) + "\n"


def _mesh_topology_circuit(rng: random.Random, idx: int) -> Tuple[str, str]:
    """Dense mesh: many nodes connected to many others."""
    n_nodes = rng.randint(3, 6)
    nodes = [f"N{i}" for i in range(n_nodes)]

    lines = [f"# OOD mesh topology {idx}"]

    # Voltage source
    v_val = rng.choice([5, 10, 12])
    lines.append(f"V1 {nodes[0]} 0 {v_val}")

    # Dense resistive mesh
    seen = set()
    r_count = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < 0.6:  # 60% chance of connection
                pair = tuple(sorted([nodes[i], nodes[j]]))
                if pair not in seen:
                    seen.add(pair)
                    r_val = rng.choice([100, 500, 1000, 2000, 5000])
                    r_count += 1
                    lines.append(f"R{r_count} {nodes[i]} {nodes[j]} {r_val}")

    return f"ood_mesh_{idx}", "\n".join(lines) + "\n"


def _chain_topology_circuit(rng: random.Random, idx: int) -> Tuple[str, str]:
    """Long chain topology: N1 - N2 - N3 - ... - Nk."""
    n = rng.randint(4, 10)
    lines = [f"# OOD chain topology {idx}"]

    # Voltage at start of chain
    v_val = rng.choice([5, 10, 12, 24])
    lines.append(f"V1 N1 0 {v_val}")

    # Chain resistors
    for i in range(1, n):
        r_val = rng.choice([100, 500, 1000, 2000, 5000])
        if i == n - 1:
            lines.append(f"R{i} N{i} 0 {r_val}")
        else:
            lines.append(f"R{i} N{i} N{i+1} {r_val}")

    return f"ood_chain_{idx}", "\n".join(lines) + "\n"


def _current_source_ood(rng: random.Random, idx: int) -> Tuple[str, str]:
    """Circuit with current sources and varied resistors."""
    n_nodes = rng.randint(2, 5)
    nodes = ["0"] + [f"N{i}" for i in range(1, n_nodes)]

    lines = [f"# OOD current source {idx}"]

    # Current source
    pos = rng.choice(nodes[1:])
    i_val = rng.choice([0.001, 0.01, 0.1, 1.0])
    lines.append(f"I1 {pos} 0 {i_val}")

    # Multiple resistors
    seen = set()
    for ri in range(rng.randint(2, 5)):
        for _ in range(50):
            a = rng.choice(nodes[1:])
            b = rng.choice(nodes)
            if a == b:
                continue
            pair = tuple(sorted([a, b]))
            if pair not in seen:
                seen.add(pair)
                break
        r_val = rng.choice([10, 50, 100, 500, 1000, 5000])
        lines.append(f"R{ri+1} {a} {b} {r_val}")

    return f"ood_current_{idx}", "\n".join(lines) + "\n"


def generate_ood_circuits(seed: int = 123, num_circuits: int = 1000) -> List[dict]:
    """Generate a deterministic list of OOD circuit dataset rows."""
    rng = random.Random(seed)
    rows: list[dict] = []

    generators = [
        _extreme_resistor_circuit,
        _high_voltage_circuit,
        _star_topology_circuit,
        _mesh_topology_circuit,
        _chain_topology_circuit,
        _current_source_ood,
    ]

    idx = 0
    while len(rows) < num_circuits and idx < num_circuits * 3:
        gen_fn = rng.choice(generators)
        name, netlist = gen_fn(rng, idx)
        try:
            circuit = parse_netlist(netlist, name=name)
            solution = solve_dc_circuit(circuit)
            inv = validate_invariants(circuit, solution)
            trace = generate_oracle_trace(circuit, solution)

            # Filter out extreme voltages
            max_abs_v = max((abs(v) for v in solution.node_voltages.values()), default=0.0)
            if max_abs_v > 1e8:
                idx += 1
                continue

            row = {
                "id": "",
                "netlist": netlist,
                "circuit_name": name,
                "solution": solution.to_dict(),
                "trace": [dict(sorted(step.items())) for step in trace],
                "invariants": inv.to_dict(),
                "ood_type": name.split("_")[1] if "_" in name else "unknown",
            }
            import hashlib
            row["id"] = hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:16]
            rows.append(row)
        except Exception:
            pass
        idx += 1

    return rows[:num_circuits]


def generate_ood_jsonl(
    seed: int = 123,
    num_circuits: int = 1000,
    output_path: str = "workspace/datasets/circuits/ood_circuits.jsonl",
) -> Path:
    """Generate OOD circuits and save as JSONL."""
    rows = generate_ood_circuits(seed=seed, num_circuits=num_circuits)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"OOD dataset: {len(rows)} rows → {path}")
    return path
