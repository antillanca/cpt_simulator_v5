import json

from backend.datasets.oracle_generator import OracleDatasetGenerator


def test_oracle_generator_is_reproducible(tmp_path, monkeypatch):
    modules = {
        "modules": {
            "oracle_unit": {
                "level": 1,
                "engine_type": "lua",
                "description": "Set x to 1.",
                "lua_code": "particle.x = 1",
                "simulation_frames": 1,
                "invariants": ["logic_basic"],
            }
        }
    }
    modules_path = tmp_path / "modules.json"
    modules_path.write_text(json.dumps(modules), encoding="utf-8")

    class DummySandbox:
        def run_rule(self, rule_text, initial_state=None, timeout_ms=None, frames=1, collect_trace=False):
            return {
                "status": "ok",
                "particle": {"x": 1},
                "trace": {"steps": [{"frame": 1, "before": initial_state or {}, "after": {"x": 1}}]},
            }

    monkeypatch.setattr("backend.datasets.oracle_generator.sandbox_manager", DummySandbox())

    out1 = tmp_path / "oracle_1.jsonl"
    out2 = tmp_path / "oracle_2.jsonl"

    gen1 = OracleDatasetGenerator(out1, modules_path=modules_path, seed=42)
    gen2 = OracleDatasetGenerator(out2, modules_path=modules_path, seed=42)

    res1 = gen1.generate_batch(limit=1)
    res2 = gen2.generate_batch(limit=1)

    assert res1.samples_generated == 1
    assert res2.samples_generated == 1
    assert res1.dataset_fingerprint == res2.dataset_fingerprint
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
    row = json.loads(out1.read_text(encoding="utf-8").strip())
    manifest = json.loads((out1.with_suffix(".manifest.json")).read_text(encoding="utf-8"))
    assert row["row_fingerprint"]
    assert manifest["dataset_fingerprint"] == res1.dataset_fingerprint
    assert {"question", "structured_state", "reasoning_trace", "equations_used", "final_answer", "verification_status", "module_source", "curriculum_layer", "seed", "timestamp"} <= set(row.keys())

