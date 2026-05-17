"""Archive manifest helpers for deterministic artifact bundles."""

from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ArchiveManifestError(ValueError):
    pass


_REQUIRED_KEYS = {
    "bundle_version",
    "created_at",
    "policy_fingerprint",
    "artifact_count",
    "artifacts",
    "source_snapshot_hash",
    "bundle_fingerprint",
}


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def validate_archive_manifest(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ArchiveManifestError("Archive manifest must be a mapping.")
    missing = sorted(_REQUIRED_KEYS - set(data))
    if missing:
        raise ArchiveManifestError("Missing archive manifest field(s): " + ", ".join(missing))
    if not isinstance(data["bundle_version"], str) or not data["bundle_version"].strip():
        raise ArchiveManifestError("bundle_version must be a non-empty string")
    if not isinstance(data["artifacts"], list):
        raise ArchiveManifestError("artifacts must be a list")
    if not isinstance(data["artifact_count"], int) or data["artifact_count"] < 0:
        raise ArchiveManifestError("artifact_count must be a non-negative integer")
    if not isinstance(data["policy_fingerprint"], str):
        raise ArchiveManifestError("policy_fingerprint must be a string")
    if not isinstance(data["source_snapshot_hash"], str):
        raise ArchiveManifestError("source_snapshot_hash must be a string")
    if not isinstance(data["bundle_fingerprint"], str):
        raise ArchiveManifestError("bundle_fingerprint must be a string")


def build_archive_manifest(bundle_path: Path) -> dict[str, Any]:
    bundle_path = Path(bundle_path)
    manifest_path = bundle_path.with_suffix(bundle_path.suffix + ".manifest.json") if bundle_path.suffix else bundle_path.with_name(bundle_path.name + ".manifest.json")
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_archive_manifest(payload)
        return payload
    if bundle_path.exists() and tarfile.is_tarfile(bundle_path):
        with tarfile.open(bundle_path, mode="r:*") as archive:
            try:
                member = archive.getmember("bundle_manifest.json")
            except KeyError as exc:
                raise ArchiveManifestError("bundle_manifest.json not found in archive") from exc
            fileobj = archive.extractfile(member)
            if fileobj is None:
                raise ArchiveManifestError("bundle_manifest.json could not be read")
            payload = json.loads(fileobj.read().decode("utf-8"))
            validate_archive_manifest(payload)
            return payload
    raise ArchiveManifestError(f"No archive manifest found for {bundle_path}")


def fingerprint_archive_manifest(data: dict[str, Any]) -> str:
    normalized = dict(data)
    normalized.pop("bundle_fingerprint", None)
    return _stable_hash(_normalize(normalized))
