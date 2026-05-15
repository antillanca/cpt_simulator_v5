"""Abstract neural model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


class NeuralModel(ABC):
    @abstractmethod
    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def validate_against_invariants(self, prediction: dict[str, Any], invariant_result: dict[str, Any]) -> bool:
        return bool(invariant_result.get("passed", False))


@dataclass
class TinyTransformerConfig:
    vocab_size: int = 256
    hidden_size: int = 64
    n_heads: int = 2
    n_layers: int = 2


class TinyTransformerModel(NeuralModel):
    def __init__(self, config: TinyTransformerConfig | None = None):
        self.config = config or TinyTransformerConfig()

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"prediction": inputs, "model": "tiny_transformer", "config": self.config.__dict__}


@dataclass
class GNNConfig:
    node_features: int = 16
    edge_features: int = 8
    hidden_size: int = 64


class GraphNeuralNetworkModel(NeuralModel):
    def __init__(self, config: GNNConfig | None = None):
        self.config = config or GNNConfig()

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"prediction": inputs, "model": "gnn", "config": self.config.__dict__}


@dataclass
class PINNConfig:
    equation_count: int = 1
    hidden_size: int = 64


class PhysicsInformedModel(NeuralModel):
    def __init__(self, config: PINNConfig | None = None):
        self.config = config or PINNConfig()

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"prediction": inputs, "model": "pinn", "config": self.config.__dict__}

