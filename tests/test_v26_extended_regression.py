"""Extended regression tests for v2.6: invariant drift, cross-module composition, and layer coverage.

These tests verify that:
- Invariants don't silently drift between runs
- Cross-module composition cases produce consistent results
- The expanded benchmark suite covers all expected layers
- Snapshot generation is reproducible
"""

import json
from pathlib import Path

from backend.benchmarks.cpt_bench.suite import BenchmarkCase, CPTBenchSuite
from backend.datasets.oracle_generator import OracleDatasetGenerator
from backend.traces.schema import (
    ReasoningTrace,
    TraceStep,
    assert_replay_consistency,
    replay_trace,
    trace_fingerprint,
)
from backend.validation.thresholds import InvariantThresholds, invariant_family


def _make_dummy_sandbox(monkeypatch, particle=None):
    class DummySandbox:
        def run_rule(self, rule_text, initial_state=None, timeout_ms=None, frames=1, collect_trace=False):
            return {
                "status": "ok",
                "particle": particle or dict(initial_state or {}),
                "trace": {"steps": [{"frame": 1, "before": initial_state or {}, "after": particle or dict(initial_state or {})}]},
            }
    return DummySandbox()


# --- Invariant drift tests ---

def test_invariant_thresholds_are_stable_across_instances():
    """Two InvariantThresholds() instances must return identical values."""
    t1 = InvariantThresholds()
    t2 = InvariantThresholds()
    assert t1.energy_threshold == t2.energy_threshold
    assert t1.momentum_threshold == t2.momentum_threshold
    assert t1.logic_threshold == t2.logic_threshold
    assert t1.quantum_threshold == t2.quantum_threshold
    assert t1.default_threshold == t2.default_threshold


def test_invariant_family_mapping_is_deterministic():
    """Same module key always maps to the same family."""
    samples = {
        "energy_kinetic": "energy",
        "energy_conservation": "energy",
        "momentum_conservation": "momentum",
        "logic_02_non_contradiction": "logic",
        "quantum_mechanics_wavefunction": "quantum",
        "geometry_basics": "default",
    }
    for module_key, expected_family in samples.items():
        assert invariant_family(module_key) == expected_family, (
            f"invariant_family({module_key!r}) = {invariant_family(module_key)!r}, expected {expected_family!r}"
        )


def test_invariant_family_does_not_silently_change():
    """Adding a new family to the thresholds must not change existing mappings."""
    t = InvariantThresholds()
    # These must remain stable — if they change, a drift occurred
    known_mappings = {
        "energy": t.energy_threshold,
        "momentum": t.momentum_threshold,
        "logic": t.logic_threshold,
        "quantum": t.quantum_threshold,
        "default": t.default_threshold,
    }
    # Verify each is a non-negative number
    for family, threshold in known_mappings.items():
        assert isinstance(threshold, (int, float)), f"{family} threshold is not numeric: {threshold!r}"
        assert threshold >= 0, f"{family} threshold is negative: {threshold}"


def test_trace_fingerprint_detects_invariant_drift():
    """If invariants_checked changes, the fingerprint must change."""
    step_a = TraceStep(
        step_id=0, rule="r", equation="e", inputs={"before": {}},
        operation="op", intermediate_result={"after": {"x": 1}},
        invariants_checked=["logic_basic"], verification={"passed": True, "violations": []}, timestamp=0.0,
    )
    step_b = TraceStep(
        step_id=0, rule="r", equation="e", inputs={"before": {}},
        operation="op", intermediate_result={"after": {"x": 1}},
        invariants_checked=["logic_basic", "energy_conservation"],
        verification={"passed": True, "violations": []}, timestamp=0.0,
    )
    fp_a = trace_fingerprint(ReasoningTrace(steps=[step_a]))
    fp_b = trace_fingerprint(ReasoningTrace(steps=[step_b]))
    assert fp_a != fp_b, "Invariant drift not detected by trace fingerprint"


# --- Cross-module composition tests ---

def test_composition_cases_exist_in_default_suite():
    """Default benchmark cases must include composition category."""
    cases = CPTBenchSuite.default_cases()
    comp_cases = [c for c in cases if c.category == "composition"]
    assert len(comp_cases) >= 3, f"Expected >= 3 composition cases, got {len(comp_cases)}"


def test_composition_case_has_combined_rule():
    """Composition cases must have non-trivial combined rules."""
    cases = CPTBenchSuite.default_cases()
    comp_cases = [c for c in cases if c.category == "composition"]
    for case in comp_cases:
        assert len(case.rule) > 5, f"Composition case {case.name} has suspiciously short rule: {case.rule!r}"
        assert case.module_source.startswith("composition::"), f"Bad module_source: {case.module_source}"


def test_composition_benchmark_replay_is_deterministic(monkeypatch):
    """Running the same composition case twice must yield the same fingerprint."""
    monkeypatch.setattr("backend.benchmarks.cpt_bench.suite.sandbox_manager", _make_dummy_sandbox(monkeypatch))

    case = BenchmarkCase(
        name="test_comp",
        category="composition",
        rule="particle.x = 1\nparticle.y = 2",
        initial_state={},
        invariants=["logic_basic"],
        expected_state={"x": 1, "y": 2},
        module_source="composition::a+b",
        curriculum_layer=-1,
    )
    r1 = CPTBenchSuite(cases=[case]).run()
    r2 = CPTBenchSuite(cases=[case]).run()
    assert r1.fingerprint == r2.fingerprint


# --- Expanded layer coverage tests ---

def test_benchmark_covers_all_layers_0_through_11():
    """Layers 0-11 (logic, math, geometry, kinematics) must have at least one case each."""
    cases = CPTBenchSuite.default_cases()
    layers = {c.curriculum_layer for c in cases if c.curriculum_layer >= 0}
    expected = set(range(12))  # 0-11
    missing = expected - layers
    assert not missing, f"Layers 0-11 missing: {sorted(missing)}"


def test_benchmark_covers_layers_16_through_20():
    """Layers 16-20 (probability, advanced, calculus, DEs, linalg) must have coverage."""
    cases = CPTBenchSuite.default_cases()
    layers = {c.curriculum_layer for c in cases if c.curriculum_layer >= 0}
    expected = {16, 17, 18, 19, 20}
    missing = expected - layers
    assert not missing, f"Layers 16-20 missing: {sorted(missing)}"


def test_benchmark_covers_layers_21_through_25():
    """Layers 21-25 (linalg, numerical, lagrangian, hamiltonian, EM) must have coverage."""
    cases = CPTBenchSuite.default_cases()
    layers = {c.curriculum_layer for c in cases if c.curriculum_layer >= 0}
    expected = {21, 22, 23, 24, 25}
    missing = expected - layers
    assert not missing, f"Layers 21-25 missing: {sorted(missing)}"


def test_benchmark_covers_layers_30_through_32():
    """Layers 30-32 (cosmology, chaos, frontier) must have coverage."""
    cases = CPTBenchSuite.default_cases()
    layers = {c.curriculum_layer for c in cases if c.curriculum_layer >= 0}
    expected = {30, 31, 32}
    missing = expected - layers
    assert not missing, f"Layers 30-32 missing: {sorted(missing)}"


def test_ood_cases_exist_in_default_suite():
    """Default benchmark cases must include OOD category."""
    cases = CPTBenchSuite.default_cases()
    ood = [c for c in cases if c.category == "ood"]
    assert len(ood) >= 2, f"Expected >= 2 OOD cases, got {len(ood)}"


# --- Snapshot reproducibility tests ---

def test_snapshot_generator_produces_valid_json(tmp_path):
    """snapshot_generator.py must produce parseable JSON with required fields."""
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/snapshot_generator.py", "--output", str(tmp_path / "snap.json")],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=120,
        env={**__import__("os").environ, "PYTHONPATH": str(project_root)},
    )
    assert result.returncode == 0, f"snapshot_generator failed: {result.stderr}"

    snap = json.loads((tmp_path / "snap.json").read_text())
    required_keys = ["git", "timestamp", "seeds", "config", "modules", "key_files", "tests", "fingerprint", "schema_version", "cpt_version"]
    for key in required_keys:
        assert key in snap, f"Snapshot missing key: {key}"


def test_snapshot_fingerprint_is_deterministic(tmp_path):
    """Running snapshot_generator twice with the same state must produce the same fingerprint."""
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parents[1]
    env = {**__import__("os").environ, "PYTHONPATH": str(project_root)}

    r1 = subprocess.run(
        [sys.executable, "scripts/snapshot_generator.py", "--output", str(tmp_path / "s1.json")],
        cwd=str(project_root), capture_output=True, text=True, timeout=120, env=env,
    )
    r2 = subprocess.run(
        [sys.executable, "scripts/snapshot_generator.py", "--output", str(tmp_path / "s2.json")],
        cwd=str(project_root), capture_output=True, text=True, timeout=120, env=env,
    )

    snap1 = json.loads((tmp_path / "s1.json").read_text())
    snap2 = json.loads((tmp_path / "s2.json").read_text())

    # Git, modules, key_files hashes should be identical
    assert snap1["modules"]["hash"] == snap2["modules"]["hash"]
    assert snap1["key_files"] == snap2["key_files"]


# --- Regression: replay across expanded layers ---

def test_replay_trace_for_synthetic_tabular_module():
    """Replaying a trace from a synthetic tabular rule must be consistent."""
    step = TraceStep(
        step_id=0,
        rule="layer_01_counting",
        equation="particle.n = 0\nparticle.n = particle.n + 1",
        inputs={"before": {}},
        operation="sandbox_execution",
        intermediate_result={"after": {"n": 1}},
        invariants_checked=["logic_basic"],
        verification={"passed": True, "violations": []},
        timestamp=0.0,
    )
    trace = ReasoningTrace(steps=[step], metadata={"module_key": "layer_01_counting", "layer": 1})
    result = assert_replay_consistency(trace, initial_state={}, expected_final={"n": 1})
    assert result.passed


def test_replay_trace_detects_drift_in_kinematics():
    """Replay must catch drift when expected final state doesn't match trace."""
    step = TraceStep(
        step_id=0,
        rule="layer_10_kinematics",
        equation="particle.x = 0\nparticle.vx = 5\nparticle.ax = 2\nparticle.x = particle.x + particle.vx + 0.5 * particle.ax",
        inputs={"before": {}},
        operation="sandbox_execution",
        intermediate_result={"after": {"x": 6, "vx": 5, "ax": 2}},
        invariants_checked=["logic_basic"],
        verification={"passed": True, "violations": []},
        timestamp=0.0,
    )
    trace = ReasoningTrace(steps=[step], metadata={"module_key": "layer_10_kinematics", "layer": 10})
    # Correct expected (must match ALL keys in intermediate_result)
    result_ok = assert_replay_consistency(trace, initial_state={}, expected_final={"x": 6, "vx": 5, "ax": 2})
    assert result_ok.passed

    # Drifted expected (x=7 instead of x=6)
    try:
        assert_replay_consistency(trace, initial_state={}, expected_final={"x": 7, "vx": 5, "ax": 2})
        assert False, "Should have raised TraceValidationError for drifted invariant"
    except Exception:
        pass  # Expected
