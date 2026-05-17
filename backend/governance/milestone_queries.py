# backend/governance/milestone_queries.py
from __future__ import annotations
from typing import Tuple
from backend.governance.milestones import MilestoneRecord


def query_milestones(
    milestones: Tuple[MilestoneRecord, ...],
    *,
    version: str | None = None,
    status: str | None = None,
    tag: str | None = None,
    dependency: str | None = None,
    artifact: str | None = None,
) -> Tuple[MilestoneRecord, ...]:
    results = milestones
    if version:
        results = [m for m in results if m.version == version]
    if status:
        results = [m for m in results if m.status == status]
    if tag:
        results = [m for m in results if tag in m.tags]
    if dependency:
        results = [m for m in results if dependency in m.dependencies]
    if artifact:
        results = [m for m in results if any(artifact in a for a in m.artifacts)]
    return tuple(results)
