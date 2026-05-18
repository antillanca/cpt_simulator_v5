"""CPT Core Specification — Model Interface Contract.

Defines the Protocol that ALL CPT models must satisfy, plus canonical
ModelMetadata. The existing CircuitGNN is wrapped via an adapter.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch


# ---------------------------------------------------------------------------
# Model metadata — frozen, deterministic
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelMetadata:
    """Canonical metadata for any CPT model."""
    model_name: str
    version: str
    parameter_count: int
    topology_specialization: str | None
    training_dataset_fingerprint: str
    projection_aware: bool
    schema_version: str = "2.11"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps({
            "model_name": self.model_name,
            "version": self.version,
            "parameter_count": self.parameter_count,
            "topology_specialization": self.topology_specialization,
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "projection_aware": self.projection_aware,
            "schema_version": self.schema_version,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "parameter_count": self.parameter_count,
            "topology_specialization": self.topology_specialization,
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "projection_aware": self.projection_aware,
            "schema_version": self.schema_version,
            "metadata": _sort_dict(self.metadata),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        return cls(
            model_name=data["model_name"],
            version=data["version"],
            parameter_count=data["parameter_count"],
            topology_specialization=data.get("topology_specialization"),
            training_dataset_fingerprint=data["training_dataset_fingerprint"],
            projection_aware=data["projection_aware"],
            schema_version=data.get("schema_version", "2.11"),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Model Protocol — the contract every model must satisfy
# ---------------------------------------------------------------------------

@runtime_checkable
class CPTModel(Protocol):
    """Stable interface that all CPT models must implement.

    Methods:
        predict: Take a canonical graph, return raw voltage predictions.
        fingerprint: Deterministic identity hash for this model+weights.
        export: Serialize model weights + metadata to disk.
        load: Deserialize from disk.
        metadata: Return ModelMetadata.
    """

    def predict(self, graph: Any) -> torch.Tensor:
        """Predict normalized voltages given a canonical graph."""
        ...

    def fingerprint(self) -> str:
        """Deterministic fingerprint of model architecture + weights."""
        ...

    def export(self, path: str | Path) -> None:
        """Save model weights + metadata to path."""
        ...

    def load(self, path: str | Path) -> None:
        """Load model weights + metadata from path."""
        ...

    def metadata(self) -> ModelMetadata:
        """Return canonical model metadata."""
        ...


# ---------------------------------------------------------------------------
# Adapter — wraps existing CircuitGNN to satisfy CPTModel
# ---------------------------------------------------------------------------

class CircuitGNNAdapter:
    """Adapter that makes existing CircuitGNN conform to CPTModel protocol."""

    def __init__(self, model: Any, meta: ModelMetadata | None = None):
        self._model = model
        self._meta = meta

    def predict(self, graph: Any) -> torch.Tensor:
        with torch.no_grad():
            edge_feat = getattr(graph, "edge_features", None) or getattr(graph, "edge_attr", None)
            return self._model(
                getattr(graph, "node_features", getattr(graph, "x", None)),
                graph.edge_index,
                edge_feat,
            )

    def fingerprint(self) -> str:
        """Deterministic fingerprint from model state_dict."""
        state = {k: v.tolist() for k, v in sorted(self._model.state_dict().items())}
        payload = json.dumps(state, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def export(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self._model.state_dict(), "metadata": self.metadata().to_json_dict()}, p)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        self._model.load_state_dict(checkpoint["model_state"])
        if "metadata" in checkpoint:
            self._meta = ModelMetadata.from_json_dict(checkpoint["metadata"])

    def metadata(self) -> ModelMetadata:
        if self._meta is not None:
            return self._meta
        count = sum(p.numel() for p in self._model.parameters())
        return ModelMetadata(
            model_name=type(self._model).__name__,
            version="unknown",
            parameter_count=count,
            topology_specialization=None,
            training_dataset_fingerprint="unknown",
            projection_aware=False,
        )


def _sort_dict(d: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k in sorted(d):
        v = d[k]
        result[k] = _sort_dict(v) if isinstance(v, dict) else v
    return result
