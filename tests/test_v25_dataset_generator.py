import json

from backend.datasets.systematic_generator import SystematicDatasetGenerator


def test_dataset_generator_builds_jsonl(tmp_path, monkeypatch):
    out = tmp_path / "dataset.jsonl"
    gen = SystematicDatasetGenerator(out, spaces={"force": [0], "mass": [1], "dt": [0.1]})

    class DummySandbox:
        def run_rule(self, rule_text, initial_state=None, timeout_ms=None, frames=1, collect_trace=False):
            return {
                "status": "ok",
                "particle": dict(initial_state or {}),
                "trace": {"steps": [{"before": initial_state or {}, "after": initial_state or {}}]},
            }

    monkeypatch.setattr("backend.datasets.systematic_generator.sandbox_manager", DummySandbox())

    dsl = """
law:
  name: noop
inputs:
  - force
equations:
  - force = force
invariants:
  - logic_basic
"""
    path = gen.generate(dsl)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["structured_state"]["metadata"]["law"]["name"] == "noop"

