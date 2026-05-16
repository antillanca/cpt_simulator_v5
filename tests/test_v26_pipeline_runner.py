import sys
from pathlib import Path


def test_oracle_pipeline_runner_emits_summary(tmp_path, monkeypatch):
    import scripts.oracle_pipeline_runner as runner

    class DummyDatasetResult:
        output_path = tmp_path / "oracle.jsonl"
        manifest_path = tmp_path / "oracle.manifest.json"
        samples_generated = 3
        modules_used = ["m1", "m2"]
        seed = 7
        dataset_fingerprint = "dataset-fingerprint"

    class DummyBenchmarkResult:
        report_path = tmp_path / "bench.json"
        version = "2.6.0"
        fingerprint = "bench-fingerprint"
        metrics = {"cases": 2, "pass_rate": 1.0}

    class DummyGenerator:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def generate_batch(self, **kwargs):
            return DummyDatasetResult()

    class DummySuite:
        def __init__(self, *args, **kwargs):
            pass

        def write_report(self, destination):
            return DummyBenchmarkResult()

    monkeypatch.setattr(runner, "OracleDatasetGenerator", DummyGenerator)
    monkeypatch.setattr(runner, "CPTBenchSuite", DummySuite)
    monkeypatch.setattr(runner, "validate_dataset_file", lambda *args, **kwargs: {"report": {"passed": True}, "output_path": str(tmp_path / "dataset.validation.json")})
    monkeypatch.setattr(runner, "validate_benchmark_file", lambda *args, **kwargs: {"report": {"passed": True}, "output_path": str(tmp_path / "bench.validation.json")})
    monkeypatch.setattr(sys, "argv", [
        "oracle_pipeline_runner.py",
        "--dataset-output",
        str(tmp_path / "oracle.jsonl"),
        "--benchmark-output",
        str(tmp_path / "bench.json"),
        "--seed",
        "7",
    ])

    exit_code = runner.main()
    assert exit_code == 0
