"""Tests for dataset fingerprint and benchmark replay regression."""

import json

from backend.datasets.oracle_generator import OracleDatasetGenerator
from backend.benchmarks.cpt_bench.suite import BenchmarkCase, CPTBenchSuite
from backend.traces.schema import (
    ReasoningTrace,
    TraceStep,
    TraceValidationError,
    assert_replay_consistency,
    replay_trace,
    trace_fingerprint,
)


def _make_dummy_sandbox(monkeypatch, particle=None):
    class DummySandbox:
        def run_rule(self, rule_text, initial_state=None, timeout_ms=None, frames=1, collect_trace=False):
            return {
                "status": "ok",
                "particle": particle or dict(initial_state or {}),
                "trace": {"steps": [{"frame": 1, "before": initial_state or {}, "after": particle or dict(initial_state or {})}]},
            }
    return DummySandbox()


# --- Dataset fingerprint tests ---

def test_dataset_fingerprint_is_deterministic(tmp_path, monkeypatch):
    modules = {
        "modules": {
            "fp_unit": {
                "level": 3,
                "engine_type": "lua",
                "description": "Set y to 42.",
                "lua_code": "particle.y = 42",
                "simulation_frames": 1,
                "invariants": ["logic_basic"],
            }
        }
    }
    modules_path = tmp_path / "modules.json"
    modules_path.write_text(json.dumps(modules), encoding="utf-8")
    monkeypatch.setattr("backend.datasets.oracle_generator.sandbox_manager", _make_dummy_sandbox(monkeypatch, {"y": 42}))

    gen1 = OracleDatasetGenerator(tmp_path / "o1.jsonl", modules_path=modules_path, seed=7)
    gen2 = OracleDatasetGenerator(tmp_path / "o2.jsonl", modules_path=modules_path, seed=7)
    r1 = gen1.generate_batch()
    r2 = gen2.generate_batch()

    assert r1.dataset_fingerprint == r2.dataset_fingerprint
    assert r1.dataset_fingerprint != ""


def test_dataset_fingerprint_changes_with_different_seed(tmp_path, monkeypatch):
    modules = {
        "modules": {
            "fp_unit": {
                "level": 3,
                "engine_type": "lua",
                "description": "Set y to 42.",
                "lua_code": "particle.y = 42",
                "simulation_frames": 1,
                "invariants": ["logic_basic"],
            }
        }
    }
    modules_path = tmp_path / "modules.json"
    modules_path.write_text(json.dumps(modules), encoding="utf-8")
    monkeypatch.setattr("backend.datasets.oracle_generator.sandbox_manager", _make_dummy_sandbox(monkeypatch, {"y": 42}))

    gen1 = OracleDatasetGenerator(tmp_path / "o1.jsonl", modules_path=modules_path, seed=7)
    gen2 = OracleDatasetGenerator(tmp_path / "o2.jsonl", modules_path=modules_path, seed=99)
    r1 = gen1.generate_batch()
    r2 = gen2.generate_batch()

    assert r1.dataset_fingerprint != r2.dataset_fingerprint


def test_manifest_includes_module_versions(tmp_path, monkeypatch):
    modules = {
        "modules": {
            "ver_unit": {
                "level": 1,
                "engine_type": "lua",
                "description": "Set a to 1.",
                "lua_code": "particle.a = 1",
                "simulation_frames": 1,
                "invariants": ["logic_basic"],
            }
        }
    }
    modules_path = tmp_path / "modules.json"
    modules_path.write_text(json.dumps(modules), encoding="utf-8")
    monkeypatch.setattr("backend.datasets.oracle_generator.sandbox_manager", _make_dummy_sandbox(monkeypatch, {"a": 1}))

    gen = OracleDatasetGenerator(tmp_path / "o.jsonl", modules_path=modules_path, seed=1)
    result = gen.generate_batch()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert "module_versions" in manifest
    assert "ver_unit" in manifest["module_versions"]
    assert manifest["module_versions"]["ver_unit"] != ""


# --- Benchmark replay tests ---

def test_benchmark_replay_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.benchmarks.cpt_bench.suite.sandbox_manager", _make_dummy_sandbox(monkeypatch))

    case = BenchmarkCase(
        name="replay_unit",
        category="test",
        rule="particle.x = 1",
        initial_state={"x": 1},
        invariants=["logic_basic"],
        expected_state={"x": 1},
    )
    suite1 = CPTBenchSuite(cases=[case])
    suite2 = CPTBenchSuite(cases=[case])
    r1 = suite1.run()
    r2 = suite2.run()

    assert r1.fingerprint == r2.fingerprint


def test_benchmark_case_fingerprint_changes_on_rule_change(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.benchmarks.cpt_bench.suite.sandbox_manager", _make_dummy_sandbox(monkeypatch))

    case_a = BenchmarkCase(name="a", category="t", rule="particle.x = 1", initial_state={"x": 1}, expected_state={"x": 1})
    case_b = BenchmarkCase(name="b", category="t", rule="particle.x = 2", initial_state={"x": 2}, expected_state={"x": 2})

    r_a = CPTBenchSuite(cases=[case_a]).run()
    r_b = CPTBenchSuite(cases=[case_b]).run()

    assert r_a.cases[0]["case_fingerprint"] != r_b.cases[0]["case_fingerprint"]


# --- Trace fingerprint tests ---

def test_trace_fingerprint_is_deterministic():
    step = TraceStep(
        step_id=0,
        rule="test",
        equation="x = 1",
        inputs={"before": {}},
        operation="test",
        intermediate_result={"after": {"x": 1}},
        invariants_checked=["logic_basic"],
        verification={"passed": True, "violations": []},
        timestamp=0.0,
    )
    trace = ReasoningTrace(steps=[step], metadata={"module_key": "test"})
    fp1 = trace_fingerprint(trace)
    fp2 = trace_fingerprint(trace)
    assert fp1 == fp2
    assert fp1 != ""


def test_trace_fingerprint_changes_with_different_content():
    step_a = TraceStep(
        step_id=0, rule="a", equation="x = 1", inputs={"before": {}},
        operation="test", intermediate_result={"after": {"x": 1}},
        invariants_checked=[], verification={"passed": True, "violations": []}, timestamp=0.0,
    )
    step_b = TraceStep(
        step_id=0, rule="b", equation="x = 2", inputs={"before": {}},
        operation="test", intermediate_result={"after": {"x": 2}},
        invariants_checked=[], verification={"passed": True, "violations": []}, timestamp=0.0,
    )
    fp_a = trace_fingerprint(ReasoningTrace(steps=[step_a]))
    fp_b = trace_fingerprint(ReasoningTrace(steps=[step_b]))
    assert fp_a != fp_b


# --- Replay consistency tests ---

def test_assert_replay_consistency_passes_for_valid_trace():
    step = TraceStep(
        step_id=0,
        rule="test",
        equation="x = 1",
        inputs={"before": {}},
        operation="test",
        intermediate_result={"after": {"x": 1}},
        invariants_checked=["logic_basic"],
        verification={"passed": True, "violations": []},
        timestamp=0.0,
    )
    trace = ReasoningTrace(steps=[step])
    result = assert_replay_consistency(trace, initial_state={}, expected_final={"x": 1})
    assert result.passed


def test_assert_replay_consistency_raises_on_mismatch():
    step = TraceStep(
        step_id=0,
        rule="test",
        equation="x = 1",
        inputs={"before": {}},
        operation="test",
        intermediate_result={"after": {"x": 1}},
        invariants_checked=["logic_basic"],
        verification={"passed": True, "violations": []},
        timestamp=0.0,
    )
    trace = ReasoningTrace(steps=[step])
    try:
        assert_replay_consistency(trace, initial_state={}, expected_final={"x": 999})
        assert False, "Should have raised TraceValidationError"
    except TraceValidationError:
        pass


def test_assert_replay_consistency_raises_on_invalid_step():
    step = TraceStep(
        step_id=-1,
        rule="test",
        equation="x = 1",
        inputs={"before": {}},
        operation="test",
        intermediate_result={"after": {"x": 1}},
        invariants_checked=["logic_basic"],
        verification={"passed": True, "violations": []},
        timestamp=0.0,
    )
    trace = ReasoningTrace(steps=[step])
    try:
        assert_replay_consistency(trace)
        assert False, "Should have raised TraceValidationError"
    except TraceValidationError:
        pass


# --- Module filter tests for more layers ---

def test_oracle_filter_by_multiple_layers(tmp_path, monkeypatch):
    modules = {
        "modules": {
            "l0_mod": {"level": 0, "engine_type": "lua", "description": "L0", "lua_code": "particle.x = 0", "invariants": ["logic_basic"]},
            "l5_mod": {"level": 5, "engine_type": "lua", "description": "L5", "lua_code": "particle.x = 5", "invariants": ["logic_basic"]},
            "l12_mod": {"level": 12, "engine_type": "lua", "description": "L12", "lua_code": "particle.x = 12", "invariants": ["logic_basic"]},
            "l20_mod": {"level": 20, "engine_type": "lua", "description": "L20", "lua_code": "particle.x = 20", "invariants": ["logic_basic"]},
        }
    }
    modules_path = tmp_path / "modules.json"
    modules_path.write_text(json.dumps(modules), encoding="utf-8")
    monkeypatch.setattr("backend.datasets.oracle_generator.sandbox_manager", _make_dummy_sandbox(monkeypatch))

    gen = OracleDatasetGenerator(tmp_path / "o.jsonl", modules_path=modules_path, seed=1)
    result = gen.generate_batch(curriculum_layers=[0, 12])

    assert "l0_mod" in result.modules_used
    assert "l12_mod" in result.modules_used
    assert "l5_mod" not in result.modules_used
    assert "l20_mod" not in result.modules_used


def test_bench_suite_layer_coverage_from_real_modules(monkeypatch):
    """Verify that default_cases picks up real modules across many layers."""
    monkeypatch.setattr("backend.benchmarks.cpt_bench.suite.sandbox_manager", _make_dummy_sandbox(monkeypatch))
    cases = CPTBenchSuite.default_cases()
    layers = {case.curriculum_layer for case in cases}

    # Should cover at least 4 distinct layers from real modules
    real_layers = {l for l in layers if l >= 0}
    assert len(real_layers) >= 4, f"Expected >= 4 real layers, got {real_layers}"
