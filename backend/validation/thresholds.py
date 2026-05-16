"""Invariant family thresholds for deterministic validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def invariant_family(name: str) -> str:
    lowered = (name or "").lower()
    if "energy" in lowered or "thermo" in lowered:
        return "energy"
    if "momentum" in lowered:
        return "momentum"
    if "logic" in lowered:
        return "logic"
    if "quantum" in lowered:
        return "quantum"
    return "default"


@dataclass(frozen=True)
class InvariantThresholds:
    energy_threshold: float = 0.0
    momentum_threshold: float = 0.0
    logic_threshold: float = 0.0
    quantum_threshold: float = 0.0
    default_threshold: float = 0.0
    neural_tolerance: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "InvariantThresholds":
        return cls(
            energy_threshold=_env_float("ENERGY_THRESHOLD", 0.0),
            momentum_threshold=_env_float("MOMENTUM_THRESHOLD", 0.0),
            logic_threshold=_env_float("LOGIC_THRESHOLD", 0.0),
            quantum_threshold=_env_float("QUANTUM_THRESHOLD", 0.0),
            default_threshold=_env_float("DEFAULT_THRESHOLD", 0.0),
            neural_tolerance=_env_float("NEURAL_APPROX_TOLERANCE", 0.0),
        )

    def threshold_for_family(self, family: str) -> float:
        if family == "energy":
            return self.energy_threshold
        if family == "momentum":
            return self.momentum_threshold
        if family == "logic":
            return self.logic_threshold
        if family == "quantum":
            return self.quantum_threshold
        return self.default_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "energy_threshold": self.energy_threshold,
            "momentum_threshold": self.momentum_threshold,
            "logic_threshold": self.logic_threshold,
            "quantum_threshold": self.quantum_threshold,
            "default_threshold": self.default_threshold,
            "neural_tolerance": self.neural_tolerance,
            "extra": dict(self.extra),
        }

