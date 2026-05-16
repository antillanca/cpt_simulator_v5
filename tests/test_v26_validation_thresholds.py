from backend.validation.pipeline import ValidationPipeline
from backend.validation.thresholds import InvariantThresholds


def test_family_threshold_rejects_when_over_limit(monkeypatch):
    class DummySandbox:
        def run_rule(self, rule_text, initial_state=None, timeout_ms=None, frames=1, collect_trace=False):
            return {
                "status": "ok",
                "particle": {"x": 1},
                "trace": {"steps": [{"frame": 1, "before": initial_state or {}, "after": {"x": 1}}]},
            }

    monkeypatch.setattr("backend.validation.pipeline.sandbox_manager", DummySandbox())
    monkeypatch.setattr(
        "backend.validation.pipeline.verify_simulation",
        lambda trace, invariant_set: {"passed": False, "violations": [{"reason": "forced"}], "metrics": {}},
    )

    pipeline = ValidationPipeline(
        violation_threshold=1.0,
        thresholds=InvariantThresholds(energy_threshold=0.0, momentum_threshold=0.0, logic_threshold=0.0, quantum_threshold=0.0),
    )
    report = pipeline.evaluate(
        [{"rule": "particle.x = 1", "initial_state": {"x": 0}, "frames": 1, "invariants": ["energy_conservation"]}]
    )

    assert report.passed is False
    assert report.rejected is True
    assert report.metrics["family_violation_rates"]["energy"] == 1.0
    assert "threshold_profile" in report.metrics
