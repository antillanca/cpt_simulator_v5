from pathlib import Path

from backend.benchmarks.cpt_bench.suite import BenchmarkCase, CPTBenchSuite


def test_cpt_bench_writes_versioned_report(tmp_path, monkeypatch):
    class DummySandbox:
        def run_rule(self, rule_text, initial_state=None, timeout_ms=None, frames=1, collect_trace=False):
            return {
                "status": "ok",
                "particle": dict(initial_state or {}),
                "trace": {"steps": [{"frame": 1, "before": initial_state or {}, "after": dict(initial_state or {})}]},
            }

    monkeypatch.setattr("backend.benchmarks.cpt_bench.suite.sandbox_manager", DummySandbox())

    suite = CPTBenchSuite(
        cases=[
            BenchmarkCase(
                name="logic",
                category="logical consistency",
                rule="particle.x = 1",
                initial_state={"x": 1},
                invariants=["logic_basic"],
                expected_state={"x": 1},
            )
        ]
    )
    report = suite.write_report(tmp_path / "report.json")

    assert report.report_path == tmp_path / "report.json"
    assert report.version == "2.6.0"
    assert report.fingerprint
    assert Path(report.report_path).exists()
    assert all(case["case_fingerprint"] for case in report.cases)
    payload = Path(report.report_path).read_text(encoding="utf-8")
    assert '"fingerprint"' in payload
