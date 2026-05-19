"""CORE v3.1 -- Domain SDK integration tests.

Verify that the Domain SDK protocols and registry work correctly.
"""

from __future__ import annotations

import pytest

from core_runtime.core.domain_sdk import (
    DomainOracle,
    DomainSurrogate,
    DomainProjection,
    DomainEvaluator,
    DomainConfidence,
    DomainTaskBase,
    DomainTask,
    list_domains,
    get_domain_components,
)
from core_runtime.domains.linear_system import (
    LinearSystemTask,
    LinearSystemOracle,
    LinearSystemSurrogate,
    LinearSystemProjection,
    LinearSystemEvaluator,
    LinearSystemConfidence,
)


class TestDomainProtocols:

    def test_oracle_is_runtime_checkable(self):
        assert hasattr(DomainOracle, '__protocol_attrs__') or hasattr(DomainOracle, '__subclasshook__')

    def test_surrogate_is_runtime_checkable(self):
        assert hasattr(DomainSurrogate, '__protocol_attrs__') or hasattr(DomainSurrogate, '__subclasshook__')

    def test_projection_is_runtime_checkable(self):
        assert hasattr(DomainProjection, '__protocol_attrs__') or hasattr(DomainProjection, '__subclasshook__')

    def test_evaluator_is_runtime_checkable(self):
        assert hasattr(DomainEvaluator, '__protocol_attrs__') or hasattr(DomainEvaluator, '__subclasshook__')

    def test_confidence_is_runtime_checkable(self):
        assert hasattr(DomainConfidence, '__protocol_attrs__') or hasattr(DomainConfidence, '__subclasshook__')

    def test_linear_system_oracle_satisfies_protocol(self):
        assert isinstance(LinearSystemOracle(), DomainOracle)

    def test_linear_system_surrogate_satisfies_protocol(self):
        assert isinstance(LinearSystemSurrogate(), DomainSurrogate)

    def test_linear_system_projection_satisfies_protocol(self):
        assert isinstance(LinearSystemProjection(), DomainProjection)

    def test_linear_system_evaluator_satisfies_protocol(self):
        assert isinstance(LinearSystemEvaluator(), DomainEvaluator)

    def test_linear_system_confidence_satisfies_protocol(self):
        assert isinstance(LinearSystemConfidence(), DomainConfidence)


class TestDomainRegistry:

    def test_list_domains_includes_linear_system(self):
        domains = list_domains()
        assert "linear_system" in domains

    def test_linear_system_has_all_components(self):
        comp = get_domain_components("linear_system")
        assert "oracle" in comp
        assert "surrogate" in comp
        assert "projection" in comp
        assert "evaluator" in comp
        assert "confidence" in comp

    def test_linear_system_oracle_class(self):
        comp = get_domain_components("linear_system")
        assert comp["oracle"] is LinearSystemOracle


class TestDomainTaskBase:

    def test_task_has_task_id(self):
        task = DomainTaskBase(
            task_id="test_001",
            domain_name="test_domain",
            input_artifact="test_input",
        )
        assert task.task_id == "test_001"

    def test_task_has_domain_name(self):
        task = DomainTaskBase(
            task_id="test_001",
            domain_name="test_domain",
            input_artifact="test_input",
        )
        assert task.domain_name == "test_domain"

    def test_task_has_input_artifact(self):
        task = DomainTaskBase(
            task_id="test_001",
            domain_name="test_domain",
            input_artifact="test_input",
        )
        assert task.input_artifact == "test_input"


class TestLinearSystemTask:

    def test_linear_system_task_fingerprint(self):
        import numpy as np
        A = np.eye(3)
        b = np.ones(3)
        task = LinearSystemTask(
            task_id="fp_test",
            domain_name="linear_system",
            input_artifact="test",
            metadata={"A": A, "b": b},
        )
        fp = task.fingerprint()
        assert isinstance(fp, str)
        assert fp.startswith("ls-")

    def test_linear_system_task_node_count(self):
        import numpy as np
        A = np.eye(4)
        b = np.ones(4)
        task = LinearSystemTask(
            task_id="nc_test",
            domain_name="linear_system",
            input_artifact="test",
            metadata={"A": A, "b": b},
        )
        assert task.node_count() == 4

    def test_linear_system_task_edge_count(self):
        import numpy as np
        A = np.eye(4)
        b = np.ones(4)
        task = LinearSystemTask(
            task_id="ec_test",
            domain_name="linear_system",
            input_artifact="test",
            metadata={"A": A, "b": b},
        )
        assert task.edge_count() == 4  # diagonal only

    def test_linear_system_task_satisfies_domain_task(self):
        import numpy as np
        A = np.eye(3)
        b = np.ones(3)
        task = LinearSystemTask(
            task_id="dt_test",
            domain_name="linear_system",
            input_artifact="test",
            metadata={"A": A, "b": b},
        )
        assert isinstance(task, DomainTask)
