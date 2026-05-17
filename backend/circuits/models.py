"""Circuit domain models for CPT v2.8 Circuit Oracle Core.

Frozen dataclasses with deterministic ordering everywhere.
Ground node is normalized to "0".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class Resistor:
    name: str
    node_a: str
    node_b: str
    resistance_ohm: float

    def __post_init__(self) -> None:
        if self.resistance_ohm <= 0:
            raise ValueError(f"Resistor {self.name}: resistance must be > 0, got {self.resistance_ohm}")


@dataclass(frozen=True)
class VoltageSource:
    name: str
    positive: str
    negative: str
    voltage: float


@dataclass(frozen=True)
class CurrentSource:
    name: str
    positive: str
    negative: str
    current: float

    def __post_init__(self) -> None:
        if self.current == 0:
            raise ValueError(f"CurrentSource {self.name}: current must be non-zero")


def _normalize_ground(node: str, ground_node: str = "0") -> str:
    """Normalize 'GND' or '0' to canonical '0'."""
    if node.upper() == "GND":
        return "0"
    return node


@dataclass(frozen=True)
class Circuit:
    name: str = "unnamed"
    resistors: Tuple[Resistor, ...] = ()
    voltage_sources: Tuple[VoltageSource, ...] = ()
    current_sources: Tuple[CurrentSource, ...] = ()
    ground_node: str = "0"

    def __post_init__(self) -> None:
        # Normalize ground in all components
        normalized_r = tuple(
            Resistor(
                name=r.name,
                node_a=_normalize_ground(r.node_a),
                node_b=_normalize_ground(r.node_b),
                resistance_ohm=r.resistance_ohm,
            )
            for r in sorted(self.resistors, key=lambda x: x.name)
        )
        normalized_v = tuple(
            VoltageSource(
                name=v.name,
                positive=_normalize_ground(v.positive),
                negative=_normalize_ground(v.negative),
                voltage=v.voltage,
            )
            for v in sorted(self.voltage_sources, key=lambda x: x.name)
        )
        normalized_i = tuple(
            CurrentSource(
                name=i.name,
                positive=_normalize_ground(i.positive),
                negative=_normalize_ground(i.negative),
                current=i.current,
            )
            for i in sorted(self.current_sources, key=lambda x: x.name)
        )
        # Use object.__setattr__ because frozen=True
        object.__setattr__(self, "resistors", normalized_r)
        object.__setattr__(self, "voltage_sources", normalized_v)
        object.__setattr__(self, "current_sources", normalized_i)
        object.__setattr__(self, "ground_node", "0")

    @property
    def all_nodes(self) -> Tuple[str, ...]:
        nodes: set[str] = set()
        for r in self.resistors:
            nodes.update([r.node_a, r.node_b])
        for v in self.voltage_sources:
            nodes.update([v.positive, v.negative])
        for i in self.current_sources:
            nodes.update([i.positive, i.negative])
        nodes.discard(self.ground_node)
        return tuple(sorted(nodes))

    @property
    def all_component_names(self) -> Tuple[str, ...]:
        names: list[str] = []
        for r in self.resistors:
            names.append(r.name)
        for v in self.voltage_sources:
            names.append(v.name)
        for i in self.current_sources:
            names.append(i.name)
        return tuple(sorted(names))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ground_node": self.ground_node,
            "resistors": [
                {"name": r.name, "node_a": r.node_a, "node_b": r.node_b, "resistance_ohm": r.resistance_ohm}
                for r in self.resistors
            ],
            "voltage_sources": [
                {"name": v.name, "positive": v.positive, "negative": v.negative, "voltage": v.voltage}
                for v in self.voltage_sources
            ],
            "current_sources": [
                {"name": i.name, "positive": i.positive, "negative": i.negative, "current": i.current}
                for i in self.current_sources
            ],
        }


@dataclass(frozen=True)
class CircuitSolution:
    node_voltages: Dict[str, float]  # node name -> voltage (ground is 0.0)
    branch_currents: Dict[str, float]  # component name -> current
    power_dissipation: Dict[str, float]  # resistor name -> power

    def to_dict(self) -> dict:
        return {
            "node_voltages": dict(sorted(self.node_voltages.items())),
            "branch_currents": dict(sorted(self.branch_currents.items())),
            "power_dissipation": dict(sorted(self.power_dissipation.items())),
        }
