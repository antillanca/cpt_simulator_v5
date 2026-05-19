"""CORE Runtime Task -- domain-agnostic execution unit.

Replaces the old circuit-specific RuntimeTask with a generic version
that only depends on the Domain SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core_runtime.core.domain_sdk import DomainTaskBase


@dataclass(frozen=True)
class RuntimeTask:
    """Domain-agnostic runtime task.

    The core runtime operates ONLY on this type. Domain-specific
    task details are carried in the `task` field (DomainTaskBase
    or a domain-specific subclass).

    Compatibility: the old backend.core_runtime.task_runtime.RuntimeTask
    is re-exported as a compatibility shim that extends this class.
    """
    task_id: str
    domain_name: str
    task: DomainTaskBase
    oracle_name: str = ""
    surrogate_name: str = ""
    projection_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        return self.task.fingerprint()

    def node_count(self) -> int:
        return self.task.node_count()

    def edge_count(self) -> int:
        return self.task.edge_count()


# Backward compatibility: allow construction with string input_artifact
def make_runtime_task(
    task_id: str,
    domain_name: str,
    input_artifact: str = "",
    oracle_name: str = "",
    surrogate_name: str = "",
    projection_enabled: bool = True,
    metadata: dict[str, Any] | None = None,
    task: DomainTaskBase | None = None,
) -> RuntimeTask:
    """Factory that supports both new (DomainTaskBase) and legacy (string) construction."""
    if task is None:
        task = DomainTaskBase(
            task_id=task_id,
            domain_name=domain_name,
            input_artifact=input_artifact,
            metadata=metadata or {},
        )
    return RuntimeTask(
        task_id=task_id,
        domain_name=domain_name,
        task=task,
        oracle_name=oracle_name,
        surrogate_name=surrogate_name,
        projection_enabled=projection_enabled,
        metadata=metadata or {},
    )
