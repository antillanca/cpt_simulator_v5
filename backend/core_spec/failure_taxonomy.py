"""CPT Core Specification — Failure Taxonomy Freeze.

Official, frozen failure type names and category groupings.
This is the SINGLE SOURCE OF TRUTH — all other modules reference these
constants. No ad-hoc failure strings anywhere else.
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence


# ---------------------------------------------------------------------------
# Official failure types — FROZEN, ordered alphabetically
# ---------------------------------------------------------------------------

FAILURE_TYPES: list[str] = [
    "bridge_node_instability",
    "conservation_drift",
    "cycle_drift_failure",
    "dense_mesh_leakage",
    "disconnected_graph_confusion",
    "extreme_resistance_instability",
    "node_aliasing",
    "ood_extreme_resistance",
    "ood_generalization_failure",  # legacy fallback — retained for classify_failure compatibility
    "ood_voltage_explosion",
    "projection_overshoot",
    "symmetry_failure",
    "topology_collapse",
]


# Legacy aliases — old names that map to new canonical names
FAILURE_ALIASES: dict[str, str] = {
    "ood_generalization_failure": "ood_extreme_resistance",
}


# ---------------------------------------------------------------------------
# Category groupings — for reporting and analysis
# ---------------------------------------------------------------------------

class FailureCategory(str, Enum):
    TOPOLOGY = "topology"
    PHYSICS = "physics"
    PROJECTION = "projection"
    OOD = "ood"


FAILURE_CATEGORIES: dict[FailureCategory, list[str]] = {
    FailureCategory.TOPOLOGY: [
        "topology_collapse",
        "disconnected_graph_confusion",
        "symmetry_failure",
        "node_aliasing",
    ],
    FailureCategory.PHYSICS: [
        "conservation_drift",
        "cycle_drift_failure",
        "dense_mesh_leakage",
        "bridge_node_instability",
        "extreme_resistance_instability",
    ],
    FailureCategory.PROJECTION: [
        "projection_overshoot",
    ],
    FailureCategory.OOD: [
        "ood_extreme_resistance",
        "ood_generalization_failure",  # legacy
        "ood_voltage_explosion",
    ],
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def is_valid_failure_type(name: str) -> bool:
    """Check if a failure type name is in the official taxonomy."""
    return name in FAILURE_TYPES


def validate_failure_type(name: str) -> str:
    """Return name if valid, else raise ValueError."""
    if name not in FAILURE_TYPES:
        valid = ", ".join(FAILURE_TYPES)
        raise ValueError(f"Invalid failure type '{name}'. Valid: {valid}")
    return name


def category_of(failure_type: str) -> FailureCategory:
    """Return the category for a given failure type."""
    for cat, types in FAILURE_CATEGORIES.items():
        if failure_type in types:
            return cat
    raise ValueError(f"Failure type '{failure_type}' not in any category")


def all_types_in_category(category: FailureCategory) -> list[str]:
    """Return all failure types in a category."""
    return list(FAILURE_CATEGORIES.get(category, []))


def validate_taxonomy_consistency() -> list[str]:
    """Verify all FAILURE_TYPES appear in exactly one category."""
    errors: list[str] = []
    seen: set[str] = set()
    for cat, types in FAILURE_CATEGORIES.items():
        for t in types:
            if t in seen:
                errors.append(f"Duplicate failure type in categories: {t}")
            seen.add(t)
            if t not in FAILURE_TYPES:
                errors.append(f"Category {cat.value} references unknown type: {t}")
    for t in FAILURE_TYPES:
        if t not in seen:
            errors.append(f"FAILURE_TYPE '{t}' not in any category")
    return errors
