"""JSON-only saved query support for deterministic inventory searches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from backend.governance.artifact_inventory import InventoryIndex, load_inventory_index

SAVED_QUERY_SCHEMA_VERSION = "2.7.10"
_REQUIRED_KEYS = {"query_name", "created_at", "inventory_fingerprint", "filters", "query_fingerprint"}
_ALLOWED_FILTER_KEYS = {
    "artifact_type",
    "schema_version",
    "fingerprint",
    "tag",
    "parent_id",
    "retention_status",
    "workspace_root",
    "created_after",
    "created_before",
    "relative_prefix",
    "pinned",
    "lineage_depth",
    "archive_status",
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


def validate_saved_query_data(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("Saved query must be a mapping.")
    missing = sorted(_REQUIRED_KEYS - set(data))
    if missing:
        raise ValueError("Missing saved query field(s): " + ", ".join(missing))
    if not isinstance(data["query_name"], str) or not data["query_name"].strip():
        raise ValueError("query_name must be a non-empty string")
    if not isinstance(data["created_at"], (int, float)):
        raise ValueError("created_at must be numeric")
    if not isinstance(data["inventory_fingerprint"], str):
        raise ValueError("inventory_fingerprint must be a string")
    if not isinstance(data["filters"], dict):
        raise ValueError("filters must be a mapping")
    unknown = sorted(set(data["filters"]) - _ALLOWED_FILTER_KEYS)
    if unknown:
        raise ValueError("Unknown filter key(s): " + ", ".join(unknown))
    if not isinstance(data["query_fingerprint"], str):
        raise ValueError("query_fingerprint must be a string")
    if data.get("schema_version", SAVED_QUERY_SCHEMA_VERSION) not in {SAVED_QUERY_SCHEMA_VERSION, None}:
        raise ValueError("Unsupported saved query schema version")


def _query_fingerprint(data: dict[str, Any]) -> str:
    payload = {
        "schema_version": SAVED_QUERY_SCHEMA_VERSION,
        "query_name": data["query_name"],
        "created_at": float(data["created_at"]),
        "inventory_fingerprint": data["inventory_fingerprint"],
        "filters": _normalize(data["filters"]),
    }
    return _stable_hash(payload)


def save_query(path, query_data):
    path = Path(path)
    payload = dict(query_data)
    payload["schema_version"] = SAVED_QUERY_SCHEMA_VERSION
    payload["filters"] = dict(sorted(payload.get("filters", {}).items()))
    payload["query_fingerprint"] = _query_fingerprint(payload)
    validate_saved_query_data(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return path


def load_query(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_saved_query_data(payload)
    payload["filters"] = dict(sorted(payload["filters"].items()))
    return payload


def _entry_matches(entry, filters: dict[str, Any]) -> bool:
    for key, value in filters.items():
        if key == "artifact_type" and entry.artifact_type != value:
            return False
        if key == "schema_version" and entry.schema_version != value:
            return False
        if key == "fingerprint" and entry.fingerprint != value:
            return False
        if key == "tag" and value not in entry.tags:
            return False
        if key == "parent_id" and value not in entry.lineage_parents:
            return False
        if key == "retention_status" and entry.retention_status != value:
            return False
        if key == "workspace_root" and entry.workspace_root != str(value):
            return False
        if key == "created_after" and entry.created_at < float(value):
            return False
        if key == "created_before" and entry.created_at > float(value):
            return False
        if key == "relative_prefix" and not entry.relative_path.startswith(str(value)):
            return False
        if key == "pinned" and bool(value) != ("pinned" in entry.tags or entry.retention_status == "pinned"):
            return False
        if key == "archive_status":
            expected = str(value)
            actual = "archived" if entry.artifact_type == "archive_bundle" or "archive" in entry.retention_status else "active"
            if actual != expected:
                return False
    return True


def _depth_map(index: InventoryIndex) -> dict[str, int]:
    parents = {entry.artifact_id: tuple(entry.lineage_parents) for entry in index.entries}
    cache: dict[str, int] = {}

    def depth(artifact_id: str, seen: set[str] | None = None) -> int:
        if artifact_id in cache:
            return cache[artifact_id]
        seen = set(seen or ())
        if artifact_id in seen:
            return 0
        seen.add(artifact_id)
        parent_ids = parents.get(artifact_id, ())
        if not parent_ids:
            cache[artifact_id] = 0
            return 0
        result = 1 + max((depth(parent_id, seen) for parent_id in parent_ids if parent_id in parents), default=0)
        cache[artifact_id] = result
        return result

    return {entry.artifact_id: depth(entry.artifact_id) for entry in index.entries}


def execute_saved_query(query, inventory):
    if isinstance(inventory, (str, Path)):
        inventory_index = load_inventory_index(Path(inventory))
    else:
        inventory_index = inventory
    filters = dict(sorted(query.get("filters", {}).items()))
    depth_map = _depth_map(inventory_index) if "lineage_depth" in filters else {}
    entries = []
    for entry in inventory_index.entries:
        if "lineage_depth" in filters and depth_map.get(entry.artifact_id, 0) != int(filters["lineage_depth"]):
            continue
        if _entry_matches(entry, filters):
            entries.append(entry)
    return tuple(sorted(entries, key=lambda item: (item.artifact_type, item.relative_path, item.fingerprint)))
