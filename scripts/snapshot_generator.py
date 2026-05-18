#!/usr/bin/env python3
"""CPT v2.6 Reproducible Snapshot Generator.

Creates a snapshot.json capturing the full system state so that any future
execution can be compared against this known-good reference.

Usage:
    PYTHONPATH=. python scripts/snapshot_generator.py [--output PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# -- project root -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        r = subprocess.run(
            ["git", "diff", "--quiet"], cwd=PROJECT_ROOT, capture_output=True
        )
        return r.returncode != 0
    except Exception:
        return True


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _dir_content_hash(directory: Path, glob: str = "**/*.json") -> str:
    """Deterministic hash of all matching files sorted by path."""
    files = sorted(directory.glob(glob))
    h = hashlib.sha256()
    for f in files:
        rel = f.relative_to(directory)
        h.update(str(rel).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _run_pytest() -> dict:
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=no", "-p", "no:timeout",
             "-k", "not snapshot"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        )
        output = r.stdout.strip()
        # Parse last line: "31 passed, 2 skipped, 1 warning in 0.76s"
        summary_line = output.split("\n")[-1] if output else ""
        return {
            "exit_code": r.returncode,
            "summary": summary_line,
            "passed": "passed" in summary_line,
        }
    except Exception as exc:
        return {"exit_code": -1, "summary": str(exc), "passed": False}


def _collect_config() -> dict:
    """Collect active configuration from env vars and defaults."""
    config = {}
    # Relevant env vars for CPT
    env_vars = [
        "PYTHONPATH", "CPT_SEED", "CPT_SANDBOX_FRAMES",
        "ENERGY_THRESHOLD", "MOMENTUM_THRESHOLD", "LOGIC_THRESHOLD",
        "QUANTUM_THRESHOLD", "NEURAL_TOLERANCE", "DEFAULT_THRESHOLD",
        "CPT_BENCH_VERSION",
    ]
    for var in env_vars:
        val = os.environ.get(var)
        if val is not None:
            config[var] = val
    return config


def generate_snapshot(output_path: Path | None = None) -> dict:
    snapshot: dict = {}

    # 1. Git metadata
    snapshot["git"] = {
        "commit": _git_head(),
        "branch": _git_branch(),
        "dirty": _git_dirty(),
    }

    # 2. Timestamps
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    snapshot["timestamp"] = now
    snapshot["timestamp_unix"] = int(time.time())

    # 3. Seeds (use deterministic defaults if not set)
    snapshot["seeds"] = {
        "default_oracle_seed": int(os.environ.get("CPT_SEED", "42")),
        "default_benchmark_seed": int(os.environ.get("CPT_BENCH_SEED", "42")),
    }

    # 4. Active configuration
    snapshot["config"] = _collect_config()

    # 5. Module hashes
    modules_path = PROJECT_ROOT / "backend" / "core_truth" / "modules.json"
    snapshot["modules"] = {
        "path": str(modules_path.relative_to(PROJECT_ROOT)),
        "hash": _file_hash(modules_path),
        "content_hash": _file_hash(modules_path),
    }

    # 6. Curriculum tree hash (all JSON under core_truth)
    core_truth_dir = PROJECT_ROOT / "backend" / "core_truth"
    snapshot["curriculum"] = {
        "directory_hash": _dir_content_hash(core_truth_dir, "**/*.json"),
    }

    # 7. Key file hashes
    key_files = [
        "backend/datasets/oracle_generator.py",
        "backend/benchmarks/cpt_bench/suite.py",
        "backend/traces/schema.py",
        "backend/validation/pipeline.py",
        "backend/validation/thresholds.py",
        "backend/tooling/permissions.py",
        "backend/core_truth/sandbox.py",
        "backend/main.py",
    ]
    snapshot["key_files"] = {}
    for rel in key_files:
        full = PROJECT_ROOT / rel
        if full.exists():
            snapshot["key_files"][rel] = _file_hash(full)

    # 8. Test results
    snapshot["tests"] = _run_pytest()

    # 9. Benchmark fingerprint
    try:
        from backend.benchmarks.cpt_bench.suite import CPTBenchSuite
        suite = CPTBenchSuite()
        result = suite.run()
        snapshot["benchmark"] = {
            "version": result.version,
            "fingerprint": result.fingerprint,
            "metrics": result.metrics,
        }
    except Exception as exc:
        snapshot["benchmark"] = {"error": str(exc)}

    # 10. Module coverage summary
    try:
        data = json.loads(modules_path.read_text(encoding="utf-8"))
        modules = data.get("modules", {})
        by_layer: dict[int, list[str]] = {}
        for key, mod in modules.items():
            layer = mod.get("level", -1)
            by_layer.setdefault(layer, []).append(key)
        snapshot["curriculum_coverage"] = {
            "total_modules": len(modules),
            "layers": {str(k): v for k, v in sorted(by_layer.items())},
        }
    except Exception:
        snapshot["curriculum_coverage"] = {"error": "could not read modules"}

    # 11. Fingerprint of the entire snapshot (for tamper detection)
    snapshot_str = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    snapshot["fingerprint"] = hashlib.sha256(snapshot_str.encode()).hexdigest()[:16]

    # 12. Version
    snapshot["schema_version"] = "1.0.0"
    snapshot["cpt_version"] = "2.6.0"

    # Write
    out = output_path or PROJECT_ROOT / "snapshot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    snapshot["_output_path"] = str(out)

    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible CPT v2.6 snapshot")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: PROJECT_ROOT/snapshot.json)")
    parser.add_argument("--dry-run", action="store_true", help="Print snapshot to stdout without writing file")
    args = parser.parse_args()

    snap = generate_snapshot(args.output)
    output_text = json.dumps(snap, indent=2, sort_keys=True, ensure_ascii=False)

    if args.dry_run:
        print(output_text)
    else:
        print(f"Snapshot written to: {snap['_output_path']}")
        print(f"Fingerprint: {snap['fingerprint']}")
        print(f"Tests: {snap['tests']['summary']}")
        print(f"Benchmark fingerprint: {snap.get('benchmark', {}).get('fingerprint', 'N/A')}")
        print(f"Curriculum modules: {snap.get('curriculum_coverage', {}).get('total_modules', 'N/A')}")


if __name__ == "__main__":
    main()
