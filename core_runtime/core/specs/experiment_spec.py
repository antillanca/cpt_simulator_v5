"""CPT Core Specification — Experiment Specification Format.

Canonical, reproducible experiment configuration. Every training run,
evaluation run, or ablation study must produce an ExperimentSpec that
fully determines its behavior given the same inputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentSpec:
    """Canonical experiment specification.

    Contains ALL parameters needed to reproduce an experiment.
    Deterministic fingerprint over all fields.
    """
    experiment_id: str
    seed: int
    dataset_fingerprint: str
    checkpoint_fingerprint: str | None
    target_mode: str
    topology_curriculum: bool
    projection_enabled: bool
    projection_config: dict[str, Any]
    training_config: dict[str, Any]
    evaluation_config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- Fingerprint ---------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_json_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    # -- Serialization -------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "dataset_fingerprint": self.dataset_fingerprint,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "target_mode": self.target_mode,
            "topology_curriculum": self.topology_curriculum,
            "projection_enabled": self.projection_enabled,
            "projection_config": _sort_dict(self.projection_config),
            "training_config": _sort_dict(self.training_config),
            "evaluation_config": _sort_dict(self.evaluation_config),
            "metadata": _sort_dict(self.metadata),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "ExperimentSpec":
        return cls(
            experiment_id=data["experiment_id"],
            seed=data["seed"],
            dataset_fingerprint=data["dataset_fingerprint"],
            checkpoint_fingerprint=data.get("checkpoint_fingerprint"),
            target_mode=data["target_mode"],
            topology_curriculum=data["topology_curriculum"],
            projection_enabled=data["projection_enabled"],
            projection_config=data.get("projection_config", {}),
            training_config=data.get("training_config", {}),
            evaluation_config=data.get("evaluation_config", {}),
            metadata=data.get("metadata", {}),
        )

    def to_yaml_lines(self) -> list[str]:
        """Simple YAML-like output (no pyyaml dependency)."""
        lines: list[str] = []
        d = self.to_json_dict()
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{k}:")
                for sk, sv in sorted(v.items()):
                    lines.append(f"  {sk}: {sv}")
            elif isinstance(v, list):
                lines.append(f"{k}: {json.dumps(v)}")
            else:
                lines.append(f"{k}: {v}")
        return lines

    # -- Validation ----------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.experiment_id:
            errors.append("experiment_id must not be empty")
        if self.seed < 0:
            errors.append("seed must be non-negative")
        if not self.dataset_fingerprint:
            errors.append("dataset_fingerprint must not be empty")
        if self.target_mode not in ("oracle", "blended_projection"):
            errors.append(f"unknown target_mode: {self.target_mode}")
        return errors


def _sort_dict(d: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k in sorted(d):
        v = d[k]
        result[k] = _sort_dict(v) if isinstance(v, dict) else v
    return result
