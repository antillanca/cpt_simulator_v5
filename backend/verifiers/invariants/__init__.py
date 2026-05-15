"""Invariant registry."""

from __future__ import annotations

from .energy import energy_conservation
from .logic import logic_basic
from .momentum import momentum_conservation


_REGISTRY = {
    "energy_conservation": energy_conservation,
    "momentum_conservation": momentum_conservation,
    "logic_basic": logic_basic,
}


def get_invariant(name):
    return _REGISTRY.get(name)

