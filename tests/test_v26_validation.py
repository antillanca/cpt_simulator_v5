from backend.validation.pipeline import ValidationPipeline


def test_validation_pipeline_passes_on_exact_match(monkeypatch):
    class DummySandbox:
        def run_rule(self, rule_text, initial_state=None, timeout_ms=None, frames=1, collect_trace=False):
            return {
                "status": "ok",
                "particle": {"x": 2, "vx": 1},
                "trace": {"steps": [{"frame": 1, "before": initial_state or {}, "after": {"x": 2, "vx": 1}}]},
            }

    monkeypatch.setattr("backend.validation.pipeline.sandbox_manager", DummySandbox())

    pipeline = ValidationPipeline(violation_threshold=0.0)
    report = pipeline.evaluate(
        [{"rule": "particle.x = 2", "initial_state": {"x": 0}, "frames": 1}],
        model_predictor=lambda case: {"x": 2, "vx": 1},
    )

    assert report.passed is True
    assert report.rejected is False
    assert report.metrics["exact_match_rate"] == 1.0


def test_validation_pipeline_rejects_on_invariant_failure(monkeypatch):
    class DummySandbox:
        def run_rule(self, rule_text, initial_state=None, timeout_ms=None, frames=1, collect_trace=False):
            return {
                "status": "ok",
                "particle": {"x": 2},
                "trace": {"steps": [{"frame": 1, "before": initial_state or {}, "after": {"x": 2}}]},
            }

    monkeypatch.setattr("backend.validation.pipeline.sandbox_manager", DummySandbox())
    monkeypatch.setattr(
        "backend.validation.pipeline.verify_simulation",
        lambda trace, invariant_set: {"passed": False, "violations": [{"reason": "forced failure"}], "metrics": {}},
    )

    pipeline = ValidationPipeline(violation_threshold=0.0)
    report = pipeline.evaluate([{"rule": "particle.x = 2", "initial_state": {"x": 0}, "frames": 1}])

    assert report.passed is False
    assert report.rejected is True
    assert report.metrics["invariant_violation_rate"] == 1.0

