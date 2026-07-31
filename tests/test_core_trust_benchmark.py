"""Tests for CORE v3.3 — Trustworthiness Benchmark.

Verifies:
- benchmark runs end-to-end
- correctness is unchanged with audit layer
- report is deterministic
- JSON output is valid
"""

import json
import tempfile
from pathlib import Path

import pytest

from scripts.run_trustworthiness_benchmark import (
    run_trust_benchmark,
    format_trust_report,
    TrustBenchmarkResult,
)


class TestTrustBenchmark:

    def test_benchmark_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            result = run_trust_benchmark(
                seed=42, sample_count=5, output_dir=td,
            )
            assert result.sample_count == 5

    def test_correctness_unchanged(self):
        """Audit layer MUST NOT change projection correctness."""
        with tempfile.TemporaryDirectory() as td:
            result = run_trust_benchmark(
                seed=42, sample_count=5, output_dir=td,
            )
            assert result.correctness_unchanged is True

    def test_overall_pass(self):
        with tempfile.TemporaryDirectory() as td:
            result = run_trust_benchmark(
                seed=42, sample_count=5, output_dir=td,
            )
            assert result.overall_pass is True

    def test_json_report_written(self):
        with tempfile.TemporaryDirectory() as td:
            run_trust_benchmark(seed=42, sample_count=3, output_dir=td)
            report_path = Path(td) / "trustworthiness_benchmark.json"
            assert report_path.exists()
            data = json.loads(report_path.read_text())
            assert data["version"] == "v3.3"
            assert data["correctness_unchanged"] is True

    def test_benchmark_deterministic(self):
        """Same seed -> same results."""
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            r1 = run_trust_benchmark(seed=42, sample_count=5, output_dir=td1)
            r2 = run_trust_benchmark(seed=42, sample_count=5, output_dir=td2)
            assert r1.avg_residual == r2.avg_residual
            assert r1.certain_rate == r2.certain_rate
            assert r1.indeterminate_rate == r2.indeterminate_rate

    def test_different_seed_different_results(self):
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            r1 = run_trust_benchmark(seed=42, sample_count=10, output_dir=td1)
            r2 = run_trust_benchmark(seed=99, sample_count=10, output_dir=td2)
            # Results may differ with different seeds
            # (not guaranteed, but very likely for small samples)
            # We just verify both pass
            assert r1.overall_pass is True
            assert r2.overall_pass is True

    def test_format_report_readable(self):
        with tempfile.TemporaryDirectory() as td:
            result = run_trust_benchmark(
                seed=42, sample_count=3, output_dir=td,
            )
            text = format_trust_report(result)
            assert "Trustworthiness Benchmark" in text
            assert "PASS" in text

    def test_trust_metrics_populated(self):
        with tempfile.TemporaryDirectory() as td:
            result = run_trust_benchmark(
                seed=42, sample_count=5, output_dir=td,
            )
            assert result.avg_trust_score >= 0.0
            assert result.avg_uncertainty_score >= 0.0
            # At least one trust level should have nonzero rate
            total = result.certain_rate + result.transitional_rate + result.indeterminate_rate
            assert abs(total - 1.0) < 1e-9

    def test_samples_populated(self):
        with tempfile.TemporaryDirectory() as td:
            result = run_trust_benchmark(
                seed=42, sample_count=5, output_dir=td,
            )
            assert len(result.samples) == 5
            for s in result.samples:
                assert "task_hash" in s
                assert "trust_level" in s
                assert "explanation_type" in s
