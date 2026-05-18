'''Dataset export contract module for CPT v2.7.

Provides schema version, required fields, validation, normalization, and fingerprinting
for rows produced by the oracle generator.
'''

import json
import hashlib
from typing import List, Dict, Any

# Export schema version identifier
EXPORT_SCHEMA_VERSION = "2.7.0"

# Strict list of fields required in the exported contract, in exact order
STRICT_EXPORT_FIELDS = [
    "sample_id",
    "question",
    "structured_state",
    "reasoning_trace",
    "equations_used",
    "invariants_checked",
    "final_answer",
    "verification_status",
    "module_source",
    "curriculum_layer",
    "seed",
    "timestamp",
    "dataset_version",
    "snapshot_hash",
    "module_hash",
]

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_export_row(row: Dict[str, Any]) -> List[str]:
    """Validate that *row* conforms to the strict export contract.

    Returns a list of error messages. An empty list indicates the row is valid.
    """
    errors: List[str] = []

    # Presence check
    for field in STRICT_EXPORT_FIELDS:
        if field not in row:
            errors.append(f"Missing required field: {field}")

    # Type checks – only for fields that are required to exist
    # (skip checks if the field is missing; presence errors already reported)
    if "sample_id" in row and not isinstance(row["sample_id"], str):
        errors.append("sample_id must be a string")
    if "curriculum_layer" in row and not isinstance(row["curriculum_layer"], int):
        errors.append("curriculum_layer must be an integer")
    if "seed" in row and not isinstance(row["seed"], int):
        errors.append("seed must be an integer")
    if "timestamp" in row and not isinstance(row["timestamp"], (int, float)):
        errors.append("timestamp must be a number (float)")
    if "dataset_version" in row and not isinstance(row["dataset_version"], str):
        errors.append("dataset_version must be a string")
    if "snapshot_hash" in row and not isinstance(row["snapshot_hash"], str):
        errors.append("snapshot_hash must be a string")
    if "module_hash" in row and not isinstance(row["module_hash"], str):
        errors.append("module_hash must be a string")

    return errors

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_export_row(
    row: Dict[str, Any],
    dataset_version: str,
    snapshot_hash: str,
    module_hash: str,
) -> Dict[str, Any]:
    """Upgrade a raw oracle‑generator row to the v2.7 contract.

    The function adds the three contract‑specific fields and returns a new dict
    whose key order follows ``STRICT_EXPORT_FIELDS``.
    """
    # Create a shallow copy to avoid mutating the caller's dict
    base = dict(row)
    base["dataset_version"] = dataset_version
    base["snapshot_hash"] = snapshot_hash
    base["module_hash"] = module_hash

    # Build an ordered dict respecting STRICT_EXPORT_FIELDS order
    ordered: Dict[str, Any] = {}
    for field in STRICT_EXPORT_FIELDS:
        if field in base:
            ordered[field] = base[field]
    return ordered

# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

def export_fingerprint(row: Dict[str, Any]) -> str:
    """Compute a deterministic SHA‑256 fingerprint of the canonical fields.

    Only the fields listed in ``STRICT_EXPORT_FIELDS`` are considered. The JSON
    representation is compact (no whitespace) and keys are sorted to guarantee
    reproducibility.
    """
    canonical = {field: row[field] for field in STRICT_EXPORT_FIELDS if field in row}
    # Compact JSON encoding with sorted keys
    serialized = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest

# ---------------------------------------------------------------------------
# Contract assembly
# ---------------------------------------------------------------------------

def row_to_contract(
    row: Dict[str, Any],
    dataset_version: str,
    snapshot_hash: str,
    module_hash: str,
) -> Dict[str, Any]:
    """Return a contract‑compliant representation of *row*.

    The function performs normalization, validates the result and, if valid,
    adds a ``row_fingerprint`` field. A ``ValueError`` is raised when validation
    fails, containing the list of error messages.
    """
    normalized = normalize_export_row(row, dataset_version, snapshot_hash, module_hash)
    errors = validate_export_row(normalized)
    if errors:
        raise ValueError("Export row validation failed: " + "; ".join(errors))

    # Compute and attach fingerprint
    fingerprint = export_fingerprint(normalized)
    normalized["row_fingerprint"] = fingerprint
    return normalized

# End of export_contract.py
