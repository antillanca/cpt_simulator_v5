"""CPT Core Runtime — Canonical Task Hashing.

Deterministic task hashing that normalizes topology, component values,
boundary conditions, and oracle/projection configuration. Equivalent
circuits produce identical hashes.

Hash schema versioned: HASH_SCHEMA_VERSION = "v1"
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from backend.core_runtime.task_runtime import RuntimeTask


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

HASH_SCHEMA_VERSION: str = "v1"

# Float normalization: round to this many significant digits
_FLOAT_SIG_DIGITS = 12


# ---------------------------------------------------------------------------
# Float normalization
# ---------------------------------------------------------------------------

def _normalize_float(v: float, sig_digits: int = _FLOAT_SIG_DIGITS) -> float:
    """Round float to sig_digits significant decimal digits.

    This ensures that 1000.00000000001 and 1000.0 hash identically.
    """
    if v == 0.0:
        return 0.0
    return round(v, sig_digits - 1 - int(math.floor(math.log10(abs(v)))))


def _normalize_value(v: Any) -> Any:
    """Recursively normalize values for hashing."""
    if isinstance(v, float):
        return _normalize_float(v)
    if isinstance(v, dict):
        return {k: _normalize_value(vv) for k, vv in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [_normalize_value(i) for i in v]
    return v


# ---------------------------------------------------------------------------
# Canonicalize a RuntimeTask
# ---------------------------------------------------------------------------

def canonicalize_task(task: RuntimeTask) -> dict[str, Any]:
    """Produce a canonical dict from a RuntimeTask for hashing.

    Includes: task identity, domain, oracle/surrogate/projection config,
    and normalized metadata. The key order is always sorted.
    """
    canon = {
        "hash_schema_version": HASH_SCHEMA_VERSION,
        "task_id": task.task_id,
        "domain": task.domain,
        "input_artifact": task.input_artifact,
        "oracle_name": task.oracle_name,
        "surrogate_name": task.surrogate_name,
        "projection_enabled": task.projection_enabled,
        "metadata": _normalize_value(task.metadata),
    }
    return canon


# ---------------------------------------------------------------------------
# Compute canonical task hash
# ---------------------------------------------------------------------------

def compute_task_hash(task: RuntimeTask) -> str:
    """Deterministic SHA-256 hash of canonicalized task.

    Equivalent tasks (same topology, values, config) produce identical
    hashes regardless of field ordering.
    """
    canon = canonicalize_task(task)
    blob = json.dumps(canon, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Hash a circuit directly (for cache key when circuit is the input)
# ---------------------------------------------------------------------------

def compute_circuit_hash(circuit: Any) -> str:
    """Deterministic SHA-256 hash of a Circuit object.

    Normalizes: component ordering, node names, float values.
    """
    if hasattr(circuit, "to_dict"):
        raw = circuit.to_dict()
    elif hasattr(circuit, "__dict__"):
        raw = {k: v for k, v in circuit.__dict__.items() if not k.startswith("_")}
    else:
        raw = {"str": str(circuit)}

    canon = _normalize_value(raw)
    blob = json.dumps(canon, sort_keys=True, ensure_ascii=False)
    prefix = f"circuit:{HASH_SCHEMA_VERSION}:"
    return hashlib.sha256((prefix + blob).encode()).hexdigest()
