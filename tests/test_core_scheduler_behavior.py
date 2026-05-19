"""CORE v3.1 -- Scheduler behavior preservation tests.

Verify that CORE scheduler behavior is consistent after DomainTask abstraction.
"""

from __future__ import annotations

import json
import os
import pytest


STATS_PATH = "core_runtime/data/operational_experience/scheduler_statistics.json"


@pytest.fixture
def scheduler_stats():
    if not os.path.exists(STATS_PATH):
        pytest.skip("scheduler_statistics.json not found")
    with open(STATS_PATH) as f:
        return json.load(f)


class TestSchedulerMetricsPresent:

    def test_has_avg_scheduler_overhead(self, scheduler_stats):
        assert "avg_scheduler_overhead_ms" in scheduler_stats
        overhead = scheduler_stats["avg_scheduler_overhead_ms"]
        assert overhead >= 0

    def test_has_avg_projection_runtime(self, scheduler_stats):
        assert "avg_projection_runtime_ms" in scheduler_stats
        rt = scheduler_stats["avg_projection_runtime_ms"]
        assert rt > 0

    def test_has_escalation_rate(self, scheduler_stats):
        assert "escalation_rate" in scheduler_stats
        rate = scheduler_stats["escalation_rate"]
        assert 0.0 <= rate <= 1.0

    def test_has_warmstart_count(self, scheduler_stats):
        assert "warmstart_count" in scheduler_stats
        assert scheduler_stats["warmstart_count"] >= 0

    def test_has_total_escalations(self, scheduler_stats):
        assert "total_escalations" in scheduler_stats
        assert scheduler_stats["total_escalations"] >= 0


class TestSchedulerEfficiencyRatio:

    def test_efficiency_ratio_exists(self, scheduler_stats):
        assert "scheduler_efficiency_ratio" in scheduler_stats

    def test_efficiency_ratio_meets_target(self, scheduler_stats):
        """scheduler_efficiency_ratio > 5.0"""
        ratio = scheduler_stats["scheduler_efficiency_ratio"]
        assert ratio > 5.0, f"Efficiency ratio {ratio:.1f}x < 5.0x target"


class TestSchedulerRoutingDistribution:

    def test_routing_distribution_exists(self, scheduler_stats):
        assert "route_distribution" in scheduler_stats
        dist = scheduler_stats["route_distribution"]
        assert isinstance(dist, dict)
        assert len(dist) > 0

    def test_all_routes_non_negative(self, scheduler_stats):
        dist = scheduler_stats["route_distribution"]
        for route, count in dist.items():
            assert count >= 0, f"Route {route} has negative count {count}"

    def test_outcome_distribution_exists(self, scheduler_stats):
        assert "outcome_distribution" in scheduler_stats
        dist = scheduler_stats["outcome_distribution"]
        assert isinstance(dist, dict)


class TestSchedulerOverheadReasonable:

    def test_overhead_under_1ms(self, scheduler_stats):
        """Scheduler overhead should be under 1ms on average."""
        overhead = scheduler_stats["avg_scheduler_overhead_ms"]
        assert overhead < 1.0, f"Scheduler overhead {overhead:.3f}ms exceeds 1ms"

    def test_overhead_under_projection_time(self, scheduler_stats):
        """Scheduler overhead should be a small fraction of projection time."""
        overhead = scheduler_stats["avg_scheduler_overhead_ms"]
        avg_proj = scheduler_stats["avg_projection_runtime_ms"]
        if avg_proj > 0:
            assert overhead / avg_proj < 0.1, "Scheduler overhead > 10% of projection time"
