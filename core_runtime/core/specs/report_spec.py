"""CPT Core Specification — Evaluation Report Standard.

Canonical, immutable evaluation report. Every arena run, ablation study,
or benchmark produces an EvaluationReport with deterministic fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationReport:
    """Canonical evaluation report for CPT models.

    Covers IID/OOD accuracy, physics invariants, projection metrics,
    topology-specific breakdowns, and failure summary.
    """
    report_id: str
    model_fingerprint: str
    dataset_fingerprint: str
    iid_mae: float
    ood_mae: float
    iid_kcl_max: float
    ood_kcl_max: float
    iid_kvl_max: float
    ood_kvl_max: float
    projection_iterations_mean: float
    speedup_factor: float
    topology_metrics: dict[str, Any] = field(default_factory=dict)
    failure_summary: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "2.11"

    # -- Fingerprint ---------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_json_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    # -- Serialization -------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "model_fingerprint": self.model_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "iid_mae": round(self.iid_mae, 9),
            "ood_mae": round(self.ood_mae, 9),
            "iid_kcl_max": round(self.iid_kcl_max, 12),
            "ood_kcl_max": round(self.ood_kcl_max, 12),
            "iid_kvl_max": round(self.iid_kvl_max, 12),
            "ood_kvl_max": round(self.ood_kvl_max, 12),
            "projection_iterations_mean": round(self.projection_iterations_mean, 3),
            "speedup_factor": round(self.speedup_factor, 3),
            "topology_metrics": _sort_dict(self.topology_metrics),
            "failure_summary": {k: self.failure_summary[k] for k in sorted(self.failure_summary)},
            "metadata": _sort_dict(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "EvaluationReport":
        return cls(
            report_id=data["report_id"],
            model_fingerprint=data["model_fingerprint"],
            dataset_fingerprint=data["dataset_fingerprint"],
            iid_mae=data["iid_mae"],
            ood_mae=data["ood_mae"],
            iid_kcl_max=data["iid_kcl_max"],
            ood_kcl_max=data["ood_kcl_max"],
            iid_kvl_max=data["iid_kvl_max"],
            ood_kvl_max=data["ood_kvl_max"],
            projection_iterations_mean=data["projection_iterations_mean"],
            speedup_factor=data["speedup_factor"],
            topology_metrics=data.get("topology_metrics", {}),
            failure_summary=data.get("failure_summary", {}),
            metadata=data.get("metadata", {}),
            schema_version=data.get("schema_version", "2.11"),
        )

    # -- Validation ----------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.report_id:
            errors.append("report_id must not be empty")
        if not self.model_fingerprint:
            errors.append("model_fingerprint must not be empty")
        if self.iid_mae < 0 or self.ood_mae < 0:
            errors.append("MAE values must be non-negative")
        if self.iid_kcl_max < 0 or self.ood_kcl_max < 0:
            errors.append("KCL max values must be non-negative")
        if self.speedup_factor < 0:
            errors.append("speedup_factor must be non-negative")
        return errors


def _sort_dict(d: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k in sorted(d):
        v = d[k]
        result[k] = _sort_dict(v) if isinstance(v, dict) else v
    return result
