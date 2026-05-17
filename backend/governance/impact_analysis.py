"""Impact analysis for artifact removal or change."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from backend.governance.artifact_inventory import InventoryIndex
from backend.governance.lineage_graph import LineageGraph
from backend.governance.reverse_dependencies import build_reverse_dependency_index, find_reverse_dependencies


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ImpactAnalysisResult:
    artifact_id: str
    impacted_artifacts: tuple[str, ...]
    dependency_depth_affected: int
    archive_bundles_affected: int
    report_invalidations: int
    checkpoint_invalidations: int
    impacted_types: dict[str, int]
    impact_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "artifact_id": self.artifact_id,
            "impacted_artifacts": list(self.impacted_artifacts),
            "dependency_depth_affected": self.dependency_depth_affected,
            "archive_bundles_affected": self.archive_bundles_affected,
            "report_invalidations": self.report_invalidations,
            "checkpoint_invalidations": self.checkpoint_invalidations,
            "impacted_types": dict(sorted(self.impacted_types.items())),
            "impact_fingerprint": self.impact_fingerprint,
        }
        return payload


def analyze_artifact_impact(
    artifact_id: str,
    inventory: InventoryIndex,
    lineage_graph: LineageGraph,
) -> ImpactAnalysisResult:
    reverse_index, _edges = build_reverse_dependency_index(inventory.entries)
    reverse = find_reverse_dependencies(artifact_id, reverse_index)
    impacted = list(reverse.impacted_artifacts)
    entry_by_id = {entry.artifact_id: entry for entry in inventory.entries}
    impacted_types: dict[str, int] = {}
    archive_bundles = 0
    report_invalidations = 0
    checkpoint_invalidations = 0
    depth_affected = 0
    for dep_id in impacted:
        entry = entry_by_id.get(dep_id)
        if entry is None:
            continue
        impacted_types[entry.artifact_type] = impacted_types.get(entry.artifact_type, 0) + 1
        depth_affected = max(depth_affected, len(entry.lineage_parents))
        if entry.artifact_type == "archive_bundle":
            archive_bundles += 1
        if entry.artifact_type == "evaluation_report":
            report_invalidations += 1
        if entry.artifact_type == "checkpoint":
            checkpoint_invalidations += 1
    payload = {
        "artifact_id": artifact_id,
        "reverse_fingerprint": reverse.dependent_count,
        "graph_fingerprint": lineage_graph.graph_fingerprint,
        "impacted": impacted,
    }
    return ImpactAnalysisResult(
        artifact_id=artifact_id,
        impacted_artifacts=tuple(impacted),
        dependency_depth_affected=depth_affected,
        archive_bundles_affected=archive_bundles,
        report_invalidations=report_invalidations,
        checkpoint_invalidations=checkpoint_invalidations,
        impacted_types=impacted_types,
        impact_fingerprint=_stable_hash(payload),
    )

