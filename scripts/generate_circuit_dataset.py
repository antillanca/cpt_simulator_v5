#!/usr/bin/env python3
"""Generate deterministic circuit oracle JSONL datasets.

Each row: id, netlist, solution, trace, invariants, fingerprint.
Reproducible: same seed → byte-for-byte identical output.
Supports sharding via backend.datasets.sharding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from backend.circuits.benchmarks import BENCHMARK_CIRCUITS
from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.invariants import validate_invariants
from backend.circuits.models import Circuit, CurrentSource, Resistor, VoltageSource
from backend.circuits.parser import parse_netlist
from backend.circuits.traces import generate_oracle_trace, trace_fingerprint


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _row_fingerprint(row: dict) -> str:
    """SHA-256 of canonical JSON excluding the fingerprint field itself."""
    copy = {k: v for k, v in row.items() if k != "fingerprint"}
    return _stable_hash(copy)


def _generate_resistor_circuit(rng: random.Random, idx: int) -> tuple[str, str]:
    """Generate a random resistive circuit netlist."""
    n_nodes = rng.randint(2, 5)
    nodes = ["0"] + [f"N{i}" for i in range(1, n_nodes)]

    n_resistors = rng.randint(1, min(6, n_nodes * 2))
    n_vsources = rng.randint(1, 2)

    lines: list[str] = [f"# Generated circuit {idx}"]

    # Add voltage sources
    for vi in range(n_vsources):
        pos = rng.choice(nodes[1:])  # non-ground
        v_val = rng.choice([1, 2, 3, 3.3, 5, 9, 10, 12, 15, 24])
        lines.append(f"V{vi+1} {pos} 0 {v_val}")

    # Add resistors
    seen_pairs: set[tuple[str, str]] = set()
    for ri in range(n_resistors):
        a, b = nodes[1], "0"  # defaults
        for _ in range(50):  # avoid duplicate edges
            a = rng.choice(nodes[1:])
            b = rng.choice(nodes)
            if a == b:
                continue
            pair = tuple(sorted([a, b]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)  # type: ignore[arg-type]
                break
        r_val = rng.choice([100, 200, 330, 470, 500, 680, 1000, 1500, 2000, 2200, 3300, 4700, 10000])
        lines.append(f"R{ri+1} {a} {b} {r_val}")

    return f"gen_{idx}", "\n".join(lines) + "\n"


def _generate_current_source_circuit(rng: random.Random, idx: int) -> tuple[str, str]:
    """Generate a random current-source circuit netlist."""
    n_nodes = rng.randint(2, 4)
    nodes = ["0"] + [f"N{i}" for i in range(1, n_nodes)]

    lines: list[str] = [f"# Generated current circuit {idx}"]

    # Add current source
    pos = rng.choice(nodes[1:])
    i_val = rng.choice([0.001, 0.002, 0.005, 0.01, 0.02, 0.05])
    lines.append(f"I1 {pos} 0 {i_val}")

    # Add resistors
    n_resistors = rng.randint(1, 4)
    for ri in range(n_resistors):
        a = rng.choice(nodes[1:])
        b = rng.choice(nodes)
        if a == b:
            b = "0"
        r_val = rng.choice([100, 200, 500, 1000, 2000, 5000, 10000])
        lines.append(f"R{ri+1} {a} {b} {r_val}")

    return f"gen_current_{idx}", "\n".join(lines) + "\n"


def generate_dataset(
    seed: int = 42,
    num_circuits: int = 100,
    include_benchmarks: bool = True,
) -> list[dict]:
    """Generate a deterministic list of circuit dataset rows."""
    rng = random.Random(seed)
    rows: list[dict] = []

    # Include hand-crafted benchmark circuits first
    if include_benchmarks:
        for bname, bnetlist in BENCHMARK_CIRCUITS:
            row = _process_netlist(bname, bnetlist)
            if row is not None:
                rows.append(row)

    # Generate random circuits
    remaining = num_circuits - len(rows)
    for i in range(max(0, remaining)):
        # Mix resistor and current-source circuits
        if rng.random() < 0.7:
            name, netlist = _generate_resistor_circuit(rng, i)
        else:
            name, netlist = _generate_current_source_circuit(rng, i)

        row = _process_netlist(name, netlist)
        if row is not None:
            rows.append(row)

    return rows


def _process_netlist(name: str, netlist: str) -> dict | None:
    """Parse, solve, validate, trace a netlist and return a dataset row."""
    try:
        circuit = parse_netlist(netlist, name=name)
        solution = solve_dc_circuit(circuit)
        inv = validate_invariants(circuit, solution)
        trace = generate_oracle_trace(circuit, solution)

        row = {
            "id": "",
            "netlist": netlist,
            "circuit_name": name,
            "solution": solution.to_dict(),
            "trace": [dict(sorted(step.items())) for step in trace],
            "invariants": inv.to_dict(),
        }
        row["id"] = _stable_hash(row)
        row["fingerprint"] = _row_fingerprint(row)
        return row
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CPT v2.8 circuit oracle JSONL dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    parser.add_argument("--num-circuits", type=int, default=100, help="Number of circuits to generate")
    parser.add_argument("--output-dir", default="workspace/datasets/circuits", help="Output directory")
    parser.add_argument("--shard-size", type=int, default=0, help="Shard size (0 = no sharding)")
    parser.add_argument("--no-benchmarks", action="store_true", help="Exclude benchmark circuits")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = generate_dataset(
        seed=args.seed,
        num_circuits=args.num_circuits,
        include_benchmarks=not args.no_benchmarks,
    )

    if args.shard_size > 0:
        # Write then shard
        raw_path = output_dir / "raw.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")

        from backend.datasets.sharding import shard_dataset
        manifest = shard_dataset(raw_path, output_dir / "shards", shard_size=args.shard_size, prefix="circuit")
        manifest_path = output_dir / "shard_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        raw_path.unlink()
        print(f"Sharded dataset: {len(rows)} rows → {output_dir / 'shards'}")
    else:
        out_path = output_dir / "circuits.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"Dataset: {len(rows)} rows → {out_path}")

    # Register artifact in governance
    try:
        from backend.governance.artifact_registry import ArtifactRegistry

        registry = ArtifactRegistry()
        dataset_fp = _stable_hash(rows)
        registry.register(
            artifact_type="circuit_oracle_dataset",
            schema_version="2.8.0",
            fingerprint=dataset_fp,
            metadata={"seed": args.seed, "num_circuits": args.num_circuits, "rows": len(rows)},
        )
        registry_path = output_dir / "artifact_registry.json"
        registry.save(registry_path)
        print(f"Artifact registry saved → {registry_path}")
    except Exception as exc:
        print(f"Warning: artifact registry skipped: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
