"""Abstract evaluation interfaces for restricted neural experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    samples: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, model, dataset) -> EvaluationResult:
        raise NotImplementedError
