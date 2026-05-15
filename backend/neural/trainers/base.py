"""Trainer abstractions for restricted neural models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingResult:
    epochs: int = 0
    loss_history: list[float] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class BaseTrainer(ABC):
    @abstractmethod
    def train(self, model, dataset) -> TrainingResult:
        raise NotImplementedError

