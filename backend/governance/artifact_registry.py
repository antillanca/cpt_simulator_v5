"""Global artifact governance for reproducible CPT artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    artifact_type: str
    schema_version: str
    fingerprint: str
    parent_fingerprints: list[str] = field(default_factory=list)
    created_at: float = 0.0
    last_accessed_at: float | None = None
    archived_at: float | None = None
    pinned: bool = False
    retention_status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)
    compatibility_status: str = "compatible"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "parent_fingerprints": sorted(self.parent_fingerprints),
            "created_at": float(self.created_at),
            "last_accessed_at": None if self.last_accessed_at is None else float(self.last_accessed_at),
            "archived_at": None if self.archived_at is None else float(self.archived_at),
            "pinned": bool(self.pinned),
            "retention_status": self.retention_status,
            "metadata": _normalize(self.metadata),
            "compatibility_status": self.compatibility_status,
        }


class ArtifactRegistry:
    """Deterministic in-memory and on-disk artifact registry."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else Path("artifacts") / "artifact_registry.json"
        self._records: dict[str, ArtifactRecord] = {}

    @staticmethod
    def build_artifact_id(artifact_type: str, schema_version: str, fingerprint: str) -> str:
        return _stable_hash({"artifact_type": artifact_type, "schema_version": schema_version, "fingerprint": fingerprint})[:16]

    def register(
        self,
        *,
        artifact_type: str,
        schema_version: str,
        fingerprint: str,
        parent_fingerprints: list[str] | None = None,
        created_at: float = 0.0,
        last_accessed_at: float | None = None,
        archived_at: float | None = None,
        pinned: bool = False,
        retention_status: str = "active",
        metadata: dict[str, Any] | None = None,
        compatibility_status: str = "compatible",
    ) -> ArtifactRecord:
        artifact_id = self.build_artifact_id(artifact_type, schema_version, fingerprint)
        record = ArtifactRecord(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            schema_version=schema_version,
            fingerprint=fingerprint,
            parent_fingerprints=sorted(parent_fingerprints or []),
            created_at=float(created_at),
            last_accessed_at=None if last_accessed_at is None else float(last_accessed_at),
            archived_at=None if archived_at is None else float(archived_at),
            pinned=bool(pinned),
            retention_status=retention_status,
            metadata=dict(sorted((metadata or {}).items())),
            compatibility_status=compatibility_status,
        )
        self._records[artifact_id] = record
        return record

    def records(self) -> list[ArtifactRecord]:
        return [self._records[key] for key in sorted(self._records)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "2.7.8",
            "records": [record.to_dict() for record in self.records()],
            "registry_fingerprint": _stable_hash([record.to_dict() for record in self.records()]),
        }

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path is not None else self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def from_file(cls, path: str | Path) -> "ArtifactRegistry":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        registry = cls(path=path)
        for record in payload.get("records", []):
            registry._records[record["artifact_id"]] = ArtifactRecord(
                artifact_id=record["artifact_id"],
                artifact_type=record["artifact_type"],
                schema_version=record["schema_version"],
                fingerprint=record["fingerprint"],
                parent_fingerprints=list(record.get("parent_fingerprints", [])),
                created_at=float(record.get("created_at", 0.0)),
                last_accessed_at=record.get("last_accessed_at"),
                archived_at=record.get("archived_at"),
                pinned=bool(record.get("pinned", False)),
                retention_status=record.get("retention_status", "active"),
                metadata=dict(record.get("metadata", {})),
                compatibility_status=record.get("compatibility_status", "compatible"),
            )
        return registry


def register_artifact(
    artifact_type: str,
    *,
    schema_version: str,
    fingerprint: str,
    parent_fingerprints: list[str] | None = None,
    created_at: float = 0.0,
    last_accessed_at: float | None = None,
    archived_at: float | None = None,
    pinned: bool = False,
    retention_status: str = "active",
    metadata: dict[str, Any] | None = None,
    compatibility_status: str = "compatible",
    registry_path: str | Path | None = None,
) -> ArtifactRecord:
    registry = ArtifactRegistry(path=registry_path)
    record = registry.register(
        artifact_type=artifact_type,
        schema_version=schema_version,
        fingerprint=fingerprint,
        parent_fingerprints=parent_fingerprints,
        created_at=created_at,
        last_accessed_at=last_accessed_at,
        archived_at=archived_at,
        pinned=pinned,
        retention_status=retention_status,
        metadata=metadata,
        compatibility_status=compatibility_status,
    )
    registry.save()
    return record
