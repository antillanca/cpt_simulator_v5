"""Checkpoint migration framework."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from backend.governance.artifact_policy import (
    ArtifactCompatibilityError,
    ArtifactPolicy,
    ArtifactPolicyError,
    MissingRequiredArtifactFieldError,
    policy_allows_version,
)
from backend.neural.checkpoints.fingerprint import checkpoint_artifact_fingerprint
from backend.neural.checkpoints.migrations import MIGRATION_GRAPH
from backend.neural.checkpoints.schema import CHECKPOINT_SCHEMA_VERSION
from backend.neural.checkpoints.validator import (
    CheckpointValidationError,
    ensure_checkpoint_payload,
    enforce_checkpoint_policy,
    infer_checkpoint_version,
)


@dataclass(frozen=True)
class MigrationResult:
    source_version: str
    target_version: str
    fields_changed: list[str]
    fields_added: list[str]
    fields_removed: list[str]
    compatibility_warnings: list[str]
    migration_fingerprint: str
    checkpoint_path: str
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_version": self.source_version,
            "target_version": self.target_version,
            "fields_changed": list(self.fields_changed),
            "fields_added": list(self.fields_added),
            "fields_removed": list(self.fields_removed),
            "compatibility_warnings": list(self.compatibility_warnings),
            "migration_fingerprint": self.migration_fingerprint,
            "checkpoint_path": self.checkpoint_path,
            "dry_run": self.dry_run,
        }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return torch.load(path, map_location="cpu")
    except json.JSONDecodeError:
        return torch.load(path, map_location="cpu")


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _migrate_payload(
    payload: dict[str, Any],
    target_version: str,
    *,
    policy: ArtifactPolicy | None = None,
    strict_policy: bool = False,
    dry_run: bool = False,
) -> tuple[dict[str, Any], list[str], list[str], list[str], list[str], str, str]:
    current_version = infer_checkpoint_version(payload)
    if current_version == target_version:
        validated = ensure_checkpoint_payload(payload, allow_legacy=False)
        fingerprint = checkpoint_artifact_fingerprint(validated)
        return validated, [], [], [], [], fingerprint, current_version

    if current_version not in MIGRATION_GRAPH:
        raise CheckpointValidationError(f"No migration path from {current_version} to {target_version}")
    if target_version not in MIGRATION_GRAPH[current_version]:
        raise CheckpointValidationError(f"No migration path from {current_version} to {target_version}")

    if policy is not None and (strict_policy or bool(policy.enforcement.get("strict_mode", False))):
        if not policy_allows_version(policy, target_version, write=True):
            raise ArtifactCompatibilityError(f"Target checkpoint version {target_version} is not allowed by policy")
        if dry_run and not policy.artifacts["checkpoint"].migration.get("allow_dry_run", False):
            raise ArtifactCompatibilityError("Dry-run migration is not allowed by policy")

    migration_fn = MIGRATION_GRAPH[current_version][target_version]
    step = migration_fn(payload)
    if policy is not None:
        if policy.artifacts["checkpoint"].migration.get("require_explicit_target_version", False) and not target_version:
            raise ArtifactPolicyError("Migration target version must be explicit")
    return (
        step.payload,
        step.changed_fields,
        step.added_fields,
        step.removed_fields,
        step.compatibility_warnings,
        step.migration_fingerprint,
        step.source_version,
    )


def migrate_checkpoint(
    checkpoint_path: Path,
    target_version: str,
    dry_run: bool = False,
    *,
    policy: ArtifactPolicy | None = None,
    strict_policy: bool = False,
) -> MigrationResult:
    checkpoint_path = Path(checkpoint_path)
    payload = _load_checkpoint(checkpoint_path)
    migrated, changed, added, removed, warnings, fingerprint, source_version = _migrate_payload(
        payload,
        target_version,
        policy=policy,
        strict_policy=strict_policy,
        dry_run=dry_run,
    )
    if policy is not None:
        enforce_checkpoint_policy(migrated, policy, strict_policy=strict_policy, allow_legacy=False)
    if not dry_run:
        _save_checkpoint(checkpoint_path, migrated)
    return MigrationResult(
        source_version=source_version,
        target_version=target_version,
        fields_changed=sorted(changed),
        fields_added=sorted(added),
        fields_removed=sorted(removed),
        compatibility_warnings=sorted(warnings),
        migration_fingerprint=fingerprint,
        checkpoint_path=str(checkpoint_path),
        dry_run=dry_run,
    )
