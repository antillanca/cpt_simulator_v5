import json

from backend.datasets.oracle_generator import OracleDatasetGenerator


def test_oracle_generator_filters_by_layer(tmp_path, monkeypatch):
    modules = {
        "modules": {
            "layer_00_existence": {
                "level": 0,
                "engine_type": "tabular",
                "description": "Identity",
                "target_state": {"essence": 1, "is_self": 1},
                "simulation_frames": 1,
                "invariants": ["logic_basic"],
            },
            "energy_kinetic": {
                "level": 12,
                "engine_type": "lua",
                "description": "KE",
                "lua_code": "particle.x = 1",
                "simulation_frames": 1,
                "invariants": ["logic_basic"],
            },
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

    out = tmp_path / "oracle.jsonl"
    gen = OracleDatasetGenerator(out, modules_path=modules_path, seed=1)
    result = gen.generate_batch(curriculum_layers=[12], limit=10)

    assert result.samples_generated == 1
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["curriculum_layer"] == 12
    assert row["module_key"] == "energy_kinetic"

