"""Deterministic retention sweeps for governed CPT artifacts."""

from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.governance.artifact_policy import ArtifactPolicy, get_artifact_policy


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _guess_artifact_type(path: Path) -> str:
    name = path.name.lower()
    suffixes = "".join(path.suffixes).lower()
    if name == "artifact_registry.json":
        return "artifact_registry"
    if name.endswith(".manifest.json"):
        return "manifest"
    if suffixes.endswith(".pt") or suffixes.endswith(".ckpt") or suffixes.endswith(".checkpoint"):
        return "checkpoint"
    if name.endswith(".jsonl"):
        return "dataset"
    if (suffixes.endswith(".tar.gz") or suffixes.endswith(".tar.zst") or suffixes.endswith(".tgz") or path.suffix == ".tar") or ("bundle" in name and path.suffix in {".gz", ".zst"}):
        return "archive_bundle"
    if "report" in name and name.endswith(".json"):
        return "evaluation_report"
    if "snapshot" in name and name.endswith(".json"):
        return "benchmark_snapshot"
    return path.suffix.lstrip(".") or "artifact"


def _read_json_if_possible(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@dataclass(frozen=True)
class SweepCandidate:
    artifact_path: Path
    artifact_type: str
    fingerprint: str
    created_at: float
    size_bytes: int
    retention_reason: str | None = None
    schema_version: str = ""
    pinned: bool = False
    archived_at: float | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": str(self.artifact_path),
            "artifact_type": self.artifact_type,
            "fingerprint": self.fingerprint,
            "created_at": float(self.created_at),
            "size_bytes": int(self.size_bytes),
            "retention_reason": self.retention_reason,
            "schema_version": self.schema_version,
            "pinned": bool(self.pinned),
            "archived_at": None if self.archived_at is None else float(self.archived_at),
            "metadata": _normalize(self.metadata or {}),
        }


@dataclass(frozen=True)
class SweepResult:
    scanned: int
    retained: int
    flagged: int
    deleted: int
    reclaimed_bytes: int
    planned: int = 0
    dry_run: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "retained": self.retained,
            "flagged": self.flagged,
            "deleted": self.deleted,
            "reclaimed_bytes": self.reclaimed_bytes,
            "planned": self.planned,
            "dry_run": self.dry_run,
        }


def _candidate_from_path(path: Path, *, registry: dict[str, Any] | None = None) -> SweepCandidate:
    stat = path.stat()
    payload = _read_json_if_possible(path) if path.is_file() else {}
    metadata = dict(payload) if isinstance(payload, dict) else {}
    fingerprint = str(payload.get("fingerprint", payload.get("artifact_fingerprint", _stable_hash({"path": str(path), "size": stat.st_size, "mtime": stat.st_mtime}))))
    schema_version = str(payload.get("schema_version", payload.get("dataset_version", "")))
    artifact_type = str(payload.get("artifact_type", _guess_artifact_type(path)))
    pinned = bool(payload.get("pinned", False))
    archived_at = payload.get("archived_at")
    if registry:
        for record in registry.get("records", []):
            if record.get("fingerprint") == fingerprint:
                pinned = bool(record.get("pinned", pinned))
                archived_at = record.get("archived_at", archived_at)
                if not schema_version:
                    schema_version = str(record.get("schema_version", ""))
                break
    return SweepCandidate(
        artifact_path=path,
        artifact_type=artifact_type,
        fingerprint=fingerprint,
        created_at=float(payload.get("created_at", payload.get("timestamp_unix", stat.st_mtime))),
        size_bytes=int(stat.st_size),
        retention_reason=None,
        schema_version=schema_version,
        pinned=pinned,
        archived_at=None if archived_at in (None, "") else float(archived_at),
        metadata=metadata,
    )


def scan_retention_candidates(root: Path, policy: ArtifactPolicy) -> list[SweepCandidate]:
    root = Path(root)
    registry_path = root / "artifact_registry.json"
    registry = _read_json_if_possible(registry_path) if registry_path.exists() else {}
    candidates: list[SweepCandidate] = []
    for path in sorted(root.rglob("*"), key=lambda p: str(p)):
        if not path.is_file():
            continue
        if path.name.endswith(".manifest.json") and path.name != "bundle_manifest.json":
            candidates.append(_candidate_from_path(path, registry=registry))
            continue
        if path.suffix in {".json", ".jsonl", ".pt", ".ckpt", ".tar", ".gz", ".zst"} or "report" in path.name.lower() or "snapshot" in path.name.lower() or tarfile.is_tarfile(path):
            candidates.append(_candidate_from_path(path, registry=registry))
    return candidates


def _type_policy(policy: ArtifactPolicy, artifact_type: str):
    try:
        return get_artifact_policy(artifact_type, policy)
    except Exception:
        return None


def _policy_value(retention: dict[str, Any], key: str, default: Any) -> Any:
    value = retention.get(key, default)
    return value


def build_retention_plan(candidates, policy: ArtifactPolicy) -> list[SweepCandidate]:
    candidates = sorted(list(candidates), key=lambda c: (str(c.artifact_type), str(c.fingerprint), float(c.created_at), str(c.artifact_path)))
    plan: list[SweepCandidate] = []
    by_type: dict[str, list[SweepCandidate]] = {}
    for candidate in candidates:
        by_type.setdefault(candidate.artifact_type, []).append(candidate)

    for artifact_type in sorted(by_type):
        items = by_type[artifact_type]
        type_policy = _type_policy(policy, artifact_type)
        if type_policy is None:
            for candidate in items:
                plan.append(
                    SweepCandidate(**{**candidate.__dict__, "retention_reason": "unknown_artifact_type"})
                )
            continue
        retention = type_policy.retention
        keep: set[str] = set()
        if _policy_value(retention, "keep_pinned", True):
            keep.update(candidate.fingerprint for candidate in items if candidate.pinned)
        keep_latest = int(_policy_value(retention, "keep_latest", 0) or 0)
        if keep_latest > 0:
            for candidate in sorted(items, key=lambda c: (-float(c.created_at), str(c.artifact_path)))[:keep_latest]:
                keep.add(candidate.fingerprint)
        if _policy_value(retention, "keep_by_fingerprint", False):
            newest_by_fingerprint: dict[str, SweepCandidate] = {}
            for candidate in sorted(items, key=lambda c: (-float(c.created_at), str(c.artifact_path))):
                newest_by_fingerprint.setdefault(candidate.fingerprint, candidate)
            keep.update(newest_by_fingerprint.keys())
        if _policy_value(retention, "keep_by_schema_version", False):
            newest_by_schema: dict[str, SweepCandidate] = {}
            for candidate in sorted(items, key=lambda c: (-float(c.created_at), str(c.artifact_path))):
                key = candidate.schema_version or "unknown"
                newest_by_schema.setdefault(key, candidate)
            keep.update(candidate.fingerprint for candidate in newest_by_schema.values())
        max_age_days = retention.get("max_artifact_age_days")
        age_cutoff = None if max_age_days in (None, "") else max(0.0, float(max_age_days)) * 86400.0
        newest_created_at = max(candidate.created_at for candidate in items) if items else 0.0
        age_boundary = newest_created_at - age_cutoff if age_cutoff is not None else None
        for candidate in items:
            reason: str | None = None
            if candidate.fingerprint in keep:
                reason = "retained_by_policy"
            elif retention.get("keep_failed_runs", False) and not bool((candidate.metadata or {}).get("verification_status", {}).get("passed", True)):
                reason = "retained_failed_run"
            elif candidate.pinned and _policy_value(retention, "keep_pinned", True):
                reason = "retained_pinned"
            elif age_boundary is not None and candidate.created_at >= age_boundary:
                reason = "retained_age_window"
            elif age_boundary is not None and candidate.created_at < age_boundary:
                reason = "expired_by_age"
            if reason is None and candidate.fingerprint not in keep:
                if retention.get("archive_before_delete", False):
                    reason = "archive_before_delete"
                else:
                    reason = f"eligible_for_deletion:{artifact_type}"
            plan.append(SweepCandidate(**{**candidate.__dict__, "retention_reason": reason}))
    return sorted(plan, key=lambda c: (str(c.artifact_type), str(c.fingerprint), str(c.artifact_path)))


def execute_retention_plan(plan, dry_run: bool = True) -> SweepResult:
    plan = sorted(list(plan), key=lambda c: (str(c.artifact_type), str(c.fingerprint), str(c.artifact_path)))
    retained = 0
    flagged = 0
    deleted = 0
    reclaimed_bytes = 0
    for candidate in plan:
        if candidate.retention_reason in {None, "retained_by_policy", "retained_pinned", "retained_failed_run", "retained_age_window"}:
            retained += 1
            continue
        if candidate.retention_reason == "archive_before_delete":
            flagged += 1
            continue
        flagged += 1
        if dry_run:
            continue
        if candidate.artifact_path.exists():
            reclaimed_bytes += int(candidate.size_bytes)
            candidate.artifact_path.unlink()
            deleted += 1
    return SweepResult(
        scanned=len(plan),
        retained=retained,
        flagged=flagged,
        deleted=deleted,
        reclaimed_bytes=reclaimed_bytes,
        planned=len([item for item in plan if item.retention_reason not in {None, "retained_by_policy", "retained_pinned", "retained_failed_run", "retained_age_window"}]),
        dry_run=dry_run,
    )
