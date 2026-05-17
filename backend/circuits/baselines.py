"""Deterministic baseline predictors for circuit surrogate evaluation."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from backend.circuits.graph_dataset import CircuitGraph
from backend.circuits.models import Circuit
def _stable_float(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return (value / float(2**64 - 1)) * 2.0 - 1.0


class BaselinePredictor(ABC):
    """Abstract deterministic baseline predictor."""

    def fit(self, graphs: Sequence[CircuitGraph]) -> "BaselinePredictor":
        return self

    @abstractmethod
    def predict(self, graph: CircuitGraph, circuit: Circuit | None = None) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class MeanBaselinePredictor(BaselinePredictor):
    mean_voltage: float = 0.0

    def fit(self, graphs: Sequence[CircuitGraph]) -> "MeanBaselinePredictor":
        values: list[float] = []
        for graph in graphs:
            values.extend(float(v) for v in graph.target_voltages.tolist())
        self.mean_voltage = float(np.mean(values)) if values else 0.0
        return self

    def predict(self, graph: CircuitGraph, circuit: Circuit | None = None) -> torch.Tensor:
        return torch.full_like(graph.target_voltages, float(self.mean_voltage))


@dataclass
class LinearRegressionBaselinePredictor(BaselinePredictor):
    weights: torch.Tensor | None = None
    bias: torch.Tensor | None = None

    def fit(self, graphs: Sequence[CircuitGraph]) -> "LinearRegressionBaselinePredictor":
        rows: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for graph in graphs:
            if graph.node_features.numel() == 0:
                continue
            x = graph.node_features.to(dtype=torch.float64)
            y = graph.target_voltages.to(dtype=torch.float64).unsqueeze(-1)
            rows.append(torch.cat([x, torch.ones(x.size(0), 1, dtype=torch.float64)], dim=1))
            targets.append(y)

        if not rows:
            self.weights = torch.zeros(0, dtype=torch.float64)
            self.bias = torch.zeros((), dtype=torch.float64)
            return self

        design = torch.cat(rows, dim=0)
        target = torch.cat(targets, dim=0)
        solution = torch.linalg.lstsq(design, target).solution.squeeze(-1)
        self.weights = solution[:-1].contiguous()
        self.bias = solution[-1:].contiguous().squeeze(0)
        return self

    def predict(self, graph: CircuitGraph, circuit: Circuit | None = None) -> torch.Tensor:
        if self.weights is None or self.bias is None:
            return torch.zeros_like(graph.target_voltages)
        x = graph.node_features.to(dtype=torch.float64)
        weights = self.weights.to(dtype=torch.float64)
        bias = self.bias.to(dtype=torch.float64)
        if x.numel() == 0:
            return torch.zeros_like(graph.target_voltages)
        pred = x @ weights + bias
        return pred.to(dtype=graph.target_voltages.dtype)


@dataclass
class RandomStableBaselinePredictor(BaselinePredictor):
    scale: float = 1.0
    seed: int = 42

    def fit(self, graphs: Sequence[CircuitGraph]) -> "RandomStableBaselinePredictor":
        values: list[float] = []
        for graph in graphs:
            values.extend(abs(float(v)) for v in graph.target_voltages.tolist())
        self.scale = float(np.mean(values)) if values else 1.0
        return self

    def predict(self, graph: CircuitGraph, circuit: Circuit | None = None) -> torch.Tensor:
        if graph.target_voltages.numel() == 0:
            return torch.zeros_like(graph.target_voltages)
        base = graph.fingerprint or "|".join(graph.node_names)
        values = [
            _stable_float(f"{self.seed}:{base}:{idx}:{node}") * self.scale
            for idx, node in enumerate(graph.node_names)
        ]
        return torch.tensor(values, dtype=graph.target_voltages.dtype)


def evaluate_baseline_predictor(
    predictor: BaselinePredictor,
    graphs: Sequence[CircuitGraph],
    circuits: Sequence[Circuit],
) -> dict[str, float]:
    from backend.circuits.surrogate_eval import _compute_kcl_violation, _compute_kvl_violation

    maes: list[float] = []
    rmses: list[float] = []
    max_errs: list[float] = []
    kcl_values: list[float] = []
    kvl_values: list[float] = []
    replay_values: list[float] = []

    for graph, circuit in zip(graphs, circuits):
        if graph.node_features.numel() == 0:
            continue
        pred_1 = predictor.predict(graph, circuit).detach().cpu()
        pred_2 = predictor.predict(graph, circuit).detach().cpu()
        replay_values.append((pred_1 - pred_2).abs().max().item())

        pred_v = pred_1
        diff = pred_v - graph.target_voltages
        maes.append(diff.abs().mean().item())
        rmses.append(diff.pow(2).mean().sqrt().item())
        max_errs.append(diff.abs().max().item())

        pred_voltage_dict = dict(zip(graph.node_names, pred_v.tolist()))
        pred_voltage_dict[circuit.ground_node] = 0.0
        kcl_values.append(_compute_kcl_violation(circuit, pred_voltage_dict))
        kvl_values.append(_compute_kvl_violation(circuit, pred_voltage_dict))

    count = len(maes)
    return {
        "mae": sum(maes) / max(count, 1),
        "rmse": sum(rmses) / max(count, 1),
        "max_error": max(max_errs) if max_errs else 0.0,
        "kcl_max_violation": max(kcl_values) if kcl_values else 0.0,
        "kvl_max_violation": max(kvl_values) if kvl_values else 0.0,
        "replay_consistency": max(replay_values) if replay_values else 0.0,
        "count": count,
    }
