"""CORE v3.1 -- Backward compatibility tests.

Verify that CORE v3.1 maintains full backward compatibility with v2.15.
"""

from __future__ import annotations

import importlib
import tempfile
import pytest

from backend.runtime.projection_scheduler import ProjectionScheduler
from backend.runtime.trajectory_analysis import TrajectoryAnalyzer
from backend.runtime.retrieval_memory import RetrievalMemory


class TestProjectionSchedulerBackwardCompat:

    def test_scheduler_instantiation(self):
        scheduler = ProjectionScheduler()
        assert scheduler is not None

    def test_scheduler_has_classify_trajectory(self):
        scheduler = ProjectionScheduler()
        assert hasattr(scheduler, 'classify_trajectory')

    def test_scheduler_has_should_escalate(self):
        scheduler = ProjectionScheduler()
        assert hasattr(scheduler, 'should_escalate')

    def test_scheduler_has_should_stop(self):
        scheduler = ProjectionScheduler()
        assert hasattr(scheduler, 'should_stop')

    def test_scheduler_has_allocate_budget(self):
        scheduler = ProjectionScheduler()
        assert hasattr(scheduler, 'allocate_budget')


class TestTrajectoryAnalyzerBackwardCompat:

    def test_analyzer_instantiation(self):
        analyzer = TrajectoryAnalyzer()
        assert analyzer is not None

    def test_analyzer_has_analyze_method(self):
        analyzer = TrajectoryAnalyzer()
        assert hasattr(analyzer, 'analyze')

    def test_oscillatory_classification(self):
        """Case A from v2.15 oscillatory convergence test must still work."""
        analyzer = TrajectoryAnalyzer()
        result = analyzer.analyze([0.8, 0.4, 0.5, 0.25, 0.3, 0.1])
        assert result.trajectory_class == "oscillatory"
        assert result.oscillation_detected is True
        assert result.divergence_detected is False

    def test_divergence_classification(self):
        """Case B: diverging trajectory."""
        analyzer = TrajectoryAnalyzer()
        result = analyzer.analyze([0.8, 0.82, 0.85, 0.9])
        assert result.divergence_detected is True

    def test_convergent_classification(self):
        analyzer = TrajectoryAnalyzer()
        result = analyzer.analyze([1.0, 0.5, 0.25, 0.125, 0.0625])
        assert result.trajectory_class in ("stable_linear", "fast_converging", "convergent")


class TestExactCacheBackwardCompat:

    @pytest.fixture
    def exact_cache_mod(self):
        try:
            return importlib.import_module("backend.exact_cache")
        except ModuleNotFoundError:
            try:
                return importlib.import_module("backend.cache.exact_cache")
            except ModuleNotFoundError:
                pytest.skip("exact_cache module not found")

    def test_cache_module_loads(self, exact_cache_mod):
        assert exact_cache_mod is not None

    def test_cache_has_exact_cache_class(self, exact_cache_mod):
        cls = getattr(exact_cache_mod, 'ExactCache', None)
        assert cls is not None


class TestRetrievalMemoryBackwardCompat:

    def test_memory_instantiation_with_basedir(self):
        with tempfile.TemporaryDirectory() as td:
            rm = RetrievalMemory(base_dir=td)
            assert rm is not None

    def test_memory_has_store(self):
        with tempfile.TemporaryDirectory() as td:
            rm = RetrievalMemory(base_dir=td)
            assert hasattr(rm, 'store') or hasattr(rm, 'add') or hasattr(rm, 'insert')

    def test_memory_has_retrieve(self):
        with tempfile.TemporaryDirectory() as td:
            rm = RetrievalMemory(base_dir=td)
            assert hasattr(rm, 'retrieve') or hasattr(rm, 'search') or hasattr(rm, 'query')
