"""CORE Domain SDK validation tests."""

from __future__ import annotations

import pytest

from core_runtime.core.domain_sdk import (
    DomainTaskBase,
    DomainOracle,
    DomainSurrogate,
    DomainProjection,
    DomainEvaluator,
    DomainConfidence,
    register_domain,
    get_domain_components,
    list_domains,
)


class TestDomainTaskBase:
    def test_construction(self):
        task = DomainTaskBase(
            task_id="t1", domain_name="test", input_artifact="art1",
            metadata={"key": "val"},
        )
        assert task.task_id == "t1"
        assert task.domain_name == "test"
        assert task.metadata["key"] == "val"

    def test_frozen(self):
        task = DomainTaskBase(task_id="t1", domain_name="test", input_artifact="a")
        with pytest.raises(Exception):
            task.task_id = "changed"  # type: ignore


class TestDomainRegistry:
    def test_register_domain(self):
        register_domain("test_sdk_domain", oracle=type("FakeOracle", (), {}))
        components = get_domain_components("test_sdk_domain")
        assert "oracle" in components

    def test_list_domains(self):
        # Import circuits domain to trigger registration
        try:
            import core_runtime.domains.circuits  # noqa: F401
        except ImportError:
            pass
        import core_runtime.domains.linear_system  # noqa: F401
        domains = list_domains()
        assert isinstance(domains, list)
        assert "linear_system" in domains

    def test_unknown_domain_returns_empty(self):
        assert get_domain_components("nonexistent_domain_xyz") == {}


class TestDomainProtocols:
    def test_oracle_protocol(self):
        class MyOracle:
            def solve(self, task): return {"solution": None, "residual": 0, "runtime_ms": 0}
        assert isinstance(MyOracle(), DomainOracle)

    def test_surrogate_protocol(self):
        class MySurrogate:
            def predict(self, task): return {"prediction": None, "residual": 0, "runtime_ms": 0}
        assert isinstance(MySurrogate(), DomainSurrogate)

    def test_projection_protocol(self):
        class MyProjection:
            def project(self, task, prediction, budget): return {"solution": None, "residual": 0, "iterations": 0, "converged": False, "runtime_ms": 0}
        assert isinstance(MyProjection(), DomainProjection)

    def test_evaluator_protocol(self):
        class MyEvaluator:
            def evaluate(self, task, solution): return {"residual": 0, "correct": True, "metrics": {}}
        assert isinstance(MyEvaluator(), DomainEvaluator)

    def test_confidence_protocol(self):
        class MyConfidence:
            def score(self, task, prediction): return 0.5
        assert isinstance(MyConfidence(), DomainConfidence)
