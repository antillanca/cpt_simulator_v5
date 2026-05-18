"""Dataset loader for CPT v2.7 distillation readiness.

Loads and validates JSONL datasets (full or sharded) against the
export contract schema. Supports backward-compatible reading of
v2.6 datasets by auto-upgrading missing fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from backend.datasets.export_contract import (
    EXPORT_SCHEMA_VERSION,
    STRICT_EXPORT_FIELDS,
    normalize_export_row,
    validate_export_row,
)
from backend.datasets.manifest import DatasetManifest, validate_manifest


class DatasetLoadError(Exception):
    """Raised when a dataset cannot be loaded or validated."""


def load_jsonl(path: str | Path, validate: bool = True) -> list[dict[str, Any]]:
    """Load a JSONL dataset file into a list of dicts.

    Args:
        path: Path to the .jsonl file.
        validate: If True, validate each row against the export contract.

    Returns:
        List of row dicts.

    Raises:
        DatasetLoadError: If the file cannot be read or rows fail validation.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetLoadError(f"Dataset file not found: {path}")

    rows: list[dict[str, Any]] = []
    errors_all: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors_all.append(f"Line {line_num}: JSON decode error: {e}")
                continue
            if validate:
                errors = validate_export_row(row)
                if errors:
                    errors_all.append(f"Line {line_num}: {', '.join(errors)}")
            rows.append(row)

    if validate and errors_all:
        raise DatasetLoadError(f"Dataset validation errors ({len(errors_all)}):\n" + "\n".join(errors_all[:10]))

    return rows


def iter_jsonl(path: str | Path, validate: bool = False) -> Iterator[dict[str, Any]]:
    """Stream JSONL records one at a time.

    Args:
        path: Path to the .jsonl file.
        validate: If True, validate each row (yields None for invalid rows).

    Yields:
        Row dicts, or None for invalid rows when validate=True.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetLoadError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if validate:
                errors = validate_export_row(row)
                if errors:
                    yield None
                    continue
            yield row


def load_with_manifest(
    dataset_path: str | Path,
    manifest_path: str | Path | None = None,
    validate: bool = True,
) -> tuple[list[dict[str, Any]], DatasetManifest]:
    """Load a dataset along with its manifest.

    Args:
        dataset_path: Path to the .jsonl file.
        manifest_path: Path to the manifest (defaults to dataset_path with .manifest.json suffix).
        validate: If True, validate rows and manifest.

    Returns:
        Tuple of (rows, manifest).
    """
    dataset_path = Path(dataset_path)
    if manifest_path is None:
        manifest_path = dataset_path.with_suffix(".manifest.json")
    manifest_path = Path(manifest_path)

    rows = load_jsonl(dataset_path, validate=False)  # Load raw first

    manifest = DatasetManifest.from_file(manifest_path) if manifest_path.exists() else DatasetManifest()

    if validate:
        # Validate manifest
        manifest_errors = validate_manifest(manifest.to_dict())
        if manifest_errors:
            raise DatasetLoadError(f"Manifest errors: {', '.join(manifest_errors)}")

        # Validate record count
        if manifest.record_count > 0 and len(rows) != manifest.record_count:
            raise DatasetLoadError(
                f"Record count mismatch: manifest says {manifest.record_count}, file has {len(rows)}"
            )

        # Validate rows
        row_errors: list[str] = []
        for i, row in enumerate(rows):
            errs = validate_export_row(row)
            if errs:
                row_errors.append(f"Row {i}: {', '.join(errs)}")
        if row_errors:
            raise DatasetLoadError(f"Row validation errors ({len(row_errors)}):\n" + "\n".join(row_errors[:10]))

    return rows, manifest


def upgrade_v26_row(
    row: dict[str, Any],
    dataset_version: str = EXPORT_SCHEMA_VERSION,
    snapshot_hash: str = "",
    module_hash: str = "",
) -> dict[str, Any]:
    """Upgrade a v2.6 oracle row to v2.7 export contract.

    v2.6 rows may be missing: dataset_version, snapshot_hash, module_hash.
    This function adds them with provided or default values.
    """
    return normalize_export_row(row, dataset_version=dataset_version, snapshot_hash=snapshot_hash, module_hash=module_hash)


def load_sharded_dataset(
    shard_dir: str | Path,
    manifest_path: str | Path,
    validate: bool = True,
) -> tuple[list[dict[str, Any]], DatasetManifest]:
    """Load a sharded dataset from a directory and its shard manifest.

    Returns:
        Tuple of (all rows, dataset manifest if exists).
    """
    from backend.datasets.sharding import iter_dataset_from_shards, load_shard_manifest

    shard_manifest = load_shard_manifest(manifest_path)
    rows = list(iter_dataset_from_shards(shard_dir, shard_manifest))

    # Look for dataset manifest in the same directory
    ds_manifest_path = Path(shard_dir) / "dataset.manifest.json"
    ds_manifest = DatasetManifest.from_file(ds_manifest_path) if ds_manifest_path.exists() else DatasetManifest(
        record_count=len(rows),
    )

    if validate:
        row_errors: list[str] = []
        for i, row in enumerate(rows):
            errs = validate_export_row(row)
            if errs:
                row_errors.append(f"Row {i}: {', '.join(errs)}")
        if row_errors:
            raise DatasetLoadError(f"Row validation errors ({len(row_errors)}):\n" + "\n".join(row_errors[:10]))

        if ds_manifest.record_count > 0 and len(rows) != ds_manifest.record_count:
            raise DatasetLoadError(
                f"Record count mismatch: manifest={ds_manifest.record_count}, actual={len(rows)}"
            )

    return rows, ds_manifest
