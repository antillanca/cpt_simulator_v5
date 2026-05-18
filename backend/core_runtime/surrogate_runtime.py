"""CPT Core Runtime — Surrogate Execution Layer.

Defines SurrogatePrediction and SurrogateRuntime. Wraps existing GNN
models and is extensible to future LoRA experts, distilled models, etc.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


# ---------------------------------------------------------------------------
# SurrogatePrediction — canonical prediction result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurrogatePrediction:
    """Immutable surrogate prediction with timing metadata."""

    prediction: torch.Tensor
    latency_ms: float
    surrogate_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Deterministic SHA-256 of prediction values + metadata."""
        import hashlib, json
        values = self.prediction.detach().cpu().numpy().tobytes()
        meta = json.dumps(_sorted(self.metadata), sort_keys=True)
        h = hashlib.sha256(values)
        h.update(meta.encode())
        return h.hexdigest()


# ---------------------------------------------------------------------------
# SurrogateRuntime — manages model inference
# ---------------------------------------------------------------------------

class SurrogateRuntime:
    """Surrogate execution layer supporting baseline GNN + future LoRA/distilled.

    Usage:
        runtime = SurrogateRuntime(model)
        pred = runtime.predict(graph)
    """

    def __init__(
        self,
        model: Any = None,
        device: str = "cpu",
        name: str = "circuit_gnn",
    ) -> None:
        self._model = model
        self._device = device
        self._name = name

    @property
    def model(self) -> Any:
        return self._model

    @property
    def name(self) -> str:
        return self._name

    def predict(self, graph: Any) -> SurrogatePrediction:
        """Run surrogate inference on graph, return SurrogatePrediction."""
        if self._model is None:
            # Zero baseline if no model loaded
            num_nodes = self._infer_num_nodes(graph)
            t0 = time.perf_counter()
            prediction = torch.zeros(num_nodes, dtype=torch.float32)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return SurrogatePrediction(
                prediction=prediction,
                latency_ms=latency_ms,
                surrogate_name=self._name,
                metadata={"mode": "zero_baseline"},
            )

        self._model.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = self._forward(graph)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return SurrogatePrediction(
            prediction=pred,
            latency_ms=latency_ms,
            surrogate_name=self._name,
        )

    def predict_raw(self, graph: Any) -> Any:
        """Return raw tensor from model (for runtime compatibility)."""
        result = self.predict(graph)
        return result.prediction

    # -- internal helpers --

    def _forward(self, graph: Any) -> torch.Tensor:
        """Forward pass through model, handling CircuitGraph or raw tensors."""
        model = self._model
        if hasattr(graph, "node_features") and hasattr(graph, "edge_index"):
            x = graph.node_features
            ei = graph.edge_index
            edge_attr = getattr(graph, "edge_features", None)

            if edge_attr is not None and hasattr(model, "conv1") and hasattr(model, "edge_encoder"):
                # EdgeAwareCircuitGNN
                return model(x, ei, edge_attr)
            elif hasattr(model, "forward"):
                return model(x, ei)
        # Fallback: try calling model directly
        return model(graph)

    @staticmethod
    def _infer_num_nodes(graph: Any) -> int:
        if hasattr(graph, "num_nodes"):
            return graph.num_nodes
        if hasattr(graph, "node_features"):
            return graph.node_features.shape[0]
        if hasattr(graph, "x"):
            return graph.x.shape[0]
        return 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted(d: dict) -> dict:
    if not isinstance(d, dict):
        return d
    return {k: _sorted(v) if isinstance(v, dict) else v for k, v in sorted(d.items())}
