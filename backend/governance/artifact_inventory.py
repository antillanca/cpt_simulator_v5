"""Deterministic artifact inventory indices for CPT workspaces."""

from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.governance.artifact_policy import ArtifactPolicy, artifact_policy_fingerprint, load_artifact_policy
from backend.governance.artifact_registry import ArtifactRegistry
from backend.governance.archive_manifest import build_archive_manifest

INVENTORY_SCHEMA_VERSION = "2.7.9"
_DEFAULT_POLICY_PATH = Path("configs") / "artifact_policy.yaml"
_IGNORED_NAMES = {
    "inventory_index.json",
    "inventory_index.md",
    "workspace_summary.json",
    "workspace_summary.md",
    "retention_report.json",
    "retention_report.md",
    "lineage_graph.json",
    "lineage_graph.md",
}


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    return value


def _human_bytes(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} B"


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_text_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_policy(policy: ArtifactPolicy | None) -> ArtifactPolicy | None:
    if policy is not None:
        return policy
    if _DEFAULT_POLICY_PATH.exists():
        return load_artifact_policy(_DEFAULT_POLICY_PATH)
    return None


def _is_archive_bundle(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".tar.gz") or name.endswith(".tar.zst") or name.endswith(".tgz"):
        return True
    if path.suffix == ".tar":
        return True
    if path.suffix in {".gz", ".zst"} and "bundle" in name:
        return True
    return False


def _artifact_type_for_path(path: Path, payload: dict[str, Any]) -> str | None:
    name = path.name.lower()
    if name == "artifact_registry.json":
        return "artifact_registry"
    if name.endswith("inventory_index.json"):
        return "inventory_index"
    if name.endswith("workspace_summary.json"):
        return "workspace_summary"
    if name.endswith("retention_report.json"):
        return "retention_report"
    if name.endswith("lineage_graph.json"):
        return "lineage_graph"
    if "training_snapshot" in name and name.endswith(".json"):
        return "training_snapshot"
    if name.endswith(".manifest.json") or name == "bundle_manifest.json":
        return "manifest"
    if _is_archive_bundle(path):
        return "archive_bundle"
    if path.suffix in {".pt", ".ckpt"} or name.endswith(".pt") or name.endswith(".ckpt"):
        return "checkpoint"
    if name.endswith(".jsonl"):
        return "dataset"
    if payload.get("artifact_type") == "evaluation_report" or "evaluation_report" in name or ("report" in name and name.endswith(".json")):
        return "evaluation_report"
    if "snapshot" in name and name.endswith(".json"):
        return "benchmark_snapshot"
    if payload.get("dataset_version") and payload.get("snapshot_hash") and payload.get("module_hash"):
        return "manifest"
    return None


def _artifact_payload(path: Path) -> dict[str, Any]:
    name = path.name.lower()
    if path.suffix in {".pt", ".ckpt"} or name.endswith(".pt") or name.endswith(".ckpt"):
        try:
            import torch

            payload = torch.load(path, map_location="cpu")
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    if path.suffix in {".json", ".jsonl"} or name.endswith(".json"):
        return _read_text_json(path)
    if _is_archive_bundle(path):
        try:
            manifest = build_archive_manifest(path)
            return manifest if isinstance(manifest, dict) else {}
        except Exception:
            return {}
    return {}


def _fingerprint_for_payload(path: Path, payload: dict[str, Any]) -> str:
    for key in (
        "artifact_fingerprint",
        "fingerprint",
        "report_fingerprint",
        "bundle_fingerprint",
        "registry_fingerprint",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if path.suffix in {".jsonl"}:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return _stable_hash(lines)
    return _file_hash(path)


def _schema_version_for_payload(payload: dict[str, Any]) -> str:
    for key in ("schema_version", "dataset_version", "bundle_version"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if "state_dict" in payload and "tokenizer" in payload and ("config" in payload or "training_config" in payload):
        return "2.7.5"
    return "unknown"


def _retention_status_for_payload(path: Path, payload: dict[str, Any]) -> str:
    if bool(payload.get("pinned", False)):
        return "pinned"
    status = payload.get("retention_status")
    if isinstance(status, str) and status.strip():
        return status
    if _is_archive_bundle(path):
        return "archived"
    return "active"


def _compatibility_status_for_payload(payload: dict[str, Any]) -> str:
    status = payload.get("compatibility_status")
    if isinstance(status, str) and status.strip():
        return status
    return "compatible"


def _entry_tags(artifact_type: str, schema_version: str, retention_status: str, compatibility_status: str, payload: dict[str, Any]) -> tuple[str, ...]:
    tags = {
        artifact_type,
        f"schema:{schema_version or 'unknown'}",
        f"retention:{retention_status}",
        f"compat:{compatibility_status}",
    }
    if bool(payload.get("pinned", False)):
        tags.add("pinned")
    if artifact_type in {"archive_bundle", "artifact_registry", "inventory_index", "workspace_summary", "retention_report", "lineage_graph", "training_snapshot"}:
        tags.add("workspace_meta")
    if artifact_type in {"evaluation_report", "checkpoint", "dataset", "manifest", "archive_bundle", "benchmark_snapshot"}:
        tags.add("artifact")
    if schema_version.startswith("2.7.5"):
        tags.add("legacy")
    return tuple(sorted(tags))


def _parent_refs(artifact_type: str, payload: dict[str, Any]) -> tuple[str, ...]:
    refs: list[str] = []
    extra = payload.get("extra", {}) if isinstance(payload.get("extra", {}), dict) else {}
    if artifact_type == "checkpoint":
        for key in ("dataset_manifest_hash", "dataset_fingerprint", "config_fingerprint", "snapshot_hash", "eval_fingerprint", "parent_oracle_version"):
            value = payload.get(key, extra.get(key))
            if isinstance(value, str) and value.strip():
                refs.append(value)
    elif artifact_type == "evaluation_report":
        summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
        checkpoint_fp = summary.get("checkpoint_artifact_fingerprint")
        if isinstance(checkpoint_fp, str) and checkpoint_fp.strip():
            refs.append(checkpoint_fp)
        dataset_fp = summary.get("dataset_manifest_hash")
        if isinstance(dataset_fp, str) and dataset_fp.strip():
            refs.append(dataset_fp)
    elif artifact_type == "manifest":
        for key in ("benchmark_fingerprint", "source_snapshot_hash", "snapshot_hash"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                refs.append(value)
    elif artifact_type == "archive_bundle":
        for item in payload.get("artifacts", []):
            if isinstance(item, dict):
                value = item.get("fingerprint")
                if isinstance(value, str) and value.strip():
                    refs.append(value)
    elif artifact_type == "workspace_summary":
        value = payload.get("inventory_fingerprint")
        if isinstance(value, str) and value.strip():
            refs.append(value)
    elif artifact_type == "training_snapshot":
        for key in ("dataset_fingerprint", "config_fingerprint", "model_fingerprint", "eval_fingerprint", "evaluation_fingerprint", "parent_oracle_version"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                refs.append(value)
    return tuple(dict.fromkeys(refs))


def _relative_path(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


@dataclass(frozen=True)
class InventoryEntry:
    artifact_id: str
    artifact_type: str
    fingerprint: str
    schema_version: str
    workspace_root: str
    relative_path: str
    created_at: float
    size_bytes: int
    lineage_parents: tuple[str, ...]
    tags: tuple[str, ...]
    retention_status: str = "active"
    compatibility_status: str = "compatible"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "fingerprint": self.fingerprint,
            "schema_version": self.schema_version,
            "workspace_root": self.workspace_root,
            "relative_path": self.relative_path,
            "created_at": float(self.created_at),
            "size_bytes": int(self.size_bytes),
            "lineage_parents": list(self.lineage_parents),
            "tags": list(self.tags),
            "retention_status": self.retention_status,
            "compatibility_status": self.compatibility_status,
        }


@dataclass(frozen=True)
class InventoryIndex:
    generated_at: float
    workspace_root: str
    entry_count: int
    inventory_fingerprint: str
    entries: tuple[InventoryEntry, ...]
    schema_version: str = INVENTORY_SCHEMA_VERSION
    policy_fingerprint: str = ""
    source_workspace_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": float(self.generated_at),
            "workspace_root": self.workspace_root,
            "entry_count": self.entry_count,
            "inventory_fingerprint": self.inventory_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "source_workspace_hash": self.source_workspace_hash,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class _RawInventoryRecord:
    artifact_type: str
    schema_version: str
    fingerprint: str
    workspace_root: str
    relative_path: str
    created_at: float
    size_bytes: int
    lineage_refs: tuple[str, ...]
    tags: tuple[str, ...]
    retention_status: str
    compatibility_status: str

    def to_entry(self, artifact_id: str, resolved_parents: tuple[str, ...]) -> InventoryEntry:
        return InventoryEntry(
            artifact_id=artifact_id,
            artifact_type=self.artifact_type,
            fingerprint=self.fingerprint,
            schema_version=self.schema_version,
            workspace_root=self.workspace_root,
            relative_path=self.relative_path,
            created_at=self.created_at,
            size_bytes=self.size_bytes,
            lineage_parents=resolved_parents,
            tags=self.tags,
            retention_status=self.retention_status,
            compatibility_status=self.compatibility_status,
        )


def _source_workspace_hash(root: Path, records: list[_RawInventoryRecord]) -> str:
    payload = {
        "workspace_root": str(root),
        "records": [
            {
                "relative_path": record.relative_path,
                "artifact_type": record.artifact_type,
                "schema_version": record.schema_version,
                "fingerprint": record.fingerprint,
                "created_at": record.created_at,
                "size_bytes": record.size_bytes,
                "tags": list(record.tags),
                "retention_status": record.retention_status,
                "compatibility_status": record.compatibility_status,
                "lineage_refs": list(record.lineage_refs),
            }
            for record in sorted(records, key=lambda item: item.relative_path)
        ],
    }
    return _stable_hash(payload)


def _inventory_fingerprint(index_payload: dict[str, Any]) -> str:
    normalized = dict(index_payload)
    normalized.pop("generated_at", None)
    return _stable_hash(_normalize(normalized))


def build_inventory_index(root: Path, *, policy: ArtifactPolicy | None = None, previous_index: InventoryIndex | None = None) -> InventoryIndex:
    root = Path(root).resolve()
    policy = _load_policy(policy)
    policy_fp = artifact_policy_fingerprint(policy) if policy is not None else ""
    raw_records: list[_RawInventoryRecord] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        if not path.is_file():
            continue
        if path.name in _IGNORED_NAMES:
            continue
        payload = _artifact_payload(path)
        artifact_type = _artifact_type_for_path(path, payload)
        if artifact_type is None:
            continue
        schema_version = _schema_version_for_payload(payload)
        retention_status = _retention_status_for_payload(path, payload)
        compatibility_status = _compatibility_status_for_payload(payload)
        fingerprint = _fingerprint_for_payload(path, payload)
        raw_records.append(
            _RawInventoryRecord(
                artifact_type=artifact_type,
                schema_version=schema_version,
                fingerprint=fingerprint,
                workspace_root=str(root),
                relative_path=_relative_path(root, path),
                created_at=float(payload.get("created_at", payload.get("timestamp_unix", path.stat().st_mtime))),
                size_bytes=int(path.stat().st_size),
                lineage_refs=_parent_refs(artifact_type, payload),
                tags=_entry_tags(artifact_type, schema_version, retention_status, compatibility_status, payload),
                retention_status=retention_status,
                compatibility_status=compatibility_status,
            )
        )

    fingerprint_to_id: dict[str, str] = {}
    for record in raw_records:
        artifact_id = ArtifactRegistry.build_artifact_id(record.artifact_type, record.schema_version, record.fingerprint)
        fingerprint_to_id.setdefault(record.fingerprint, artifact_id)

    entries: list[InventoryEntry] = []
    for record in sorted(raw_records, key=lambda item: (item.artifact_type, item.relative_path, item.fingerprint)):
        artifact_id = fingerprint_to_id[record.fingerprint]
        resolved_parents = tuple(
            sorted(
                {
                    fingerprint_to_id.get(ref, f"ref:{ref}")
                    for ref in record.lineage_refs
                    if ref
                }
            )
        )
        entries.append(record.to_entry(artifact_id, resolved_parents))

    entries = tuple(sorted(entries, key=lambda item: (item.artifact_type, item.relative_path, item.fingerprint)))
    generated_at = max((entry.created_at for entry in entries), default=0.0)
    source_workspace_hash = _source_workspace_hash(root, raw_records)
    payload = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "workspace_root": str(root),
        "entry_count": len(entries),
        "policy_fingerprint": policy_fp,
        "source_workspace_hash": source_workspace_hash,
        "entries": [entry.to_dict() for entry in entries],
    }
    payload["inventory_fingerprint"] = _inventory_fingerprint(payload)
    index = InventoryIndex(
        generated_at=generated_at,
        workspace_root=str(root),
        entry_count=len(entries),
        inventory_fingerprint=payload["inventory_fingerprint"],
        entries=entries,
        schema_version=INVENTORY_SCHEMA_VERSION,
        policy_fingerprint=policy_fp,
        source_workspace_hash=source_workspace_hash,
    )
    if previous_index is not None and previous_index.inventory_fingerprint == index.inventory_fingerprint:
        return previous_index
    return index


def save_inventory_index(index: InventoryIndex, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index.to_dict(), indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def load_inventory_index(path: Path) -> InventoryIndex:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for item in payload.get("entries", []):
        entries.append(
            InventoryEntry(
                artifact_id=str(item["artifact_id"]),
                artifact_type=str(item["artifact_type"]),
                fingerprint=str(item["fingerprint"]),
                schema_version=str(item["schema_version"]),
                workspace_root=str(item["workspace_root"]),
                relative_path=str(item["relative_path"]),
                created_at=float(item.get("created_at", 0.0)),
                size_bytes=int(item.get("size_bytes", 0)),
                lineage_parents=tuple(item.get("lineage_parents", [])),
                tags=tuple(item.get("tags", [])),
                retention_status=str(item.get("retention_status", "active")),
                compatibility_status=str(item.get("compatibility_status", "compatible")),
            )
        )
    index = InventoryIndex(
        generated_at=float(payload.get("generated_at", 0.0)),
        workspace_root=str(payload.get("workspace_root", "")),
        entry_count=int(payload.get("entry_count", len(entries))),
        inventory_fingerprint=str(payload.get("inventory_fingerprint", "")),
        entries=tuple(sorted(entries, key=lambda item: (item.artifact_type, item.relative_path, item.fingerprint))),
        schema_version=str(payload.get("schema_version", INVENTORY_SCHEMA_VERSION)),
        policy_fingerprint=str(payload.get("policy_fingerprint", "")),
        source_workspace_hash=str(payload.get("source_workspace_hash", "")),
    )
    return index
