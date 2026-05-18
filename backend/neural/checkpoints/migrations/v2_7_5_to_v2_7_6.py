"""Migration from v2.7.5 legacy checkpoints to v2.7.6 governed checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.neural.checkpoints.fingerprint import hash_optimizer_state, hash_state_dict
from backend.neural.checkpoints.schema import (
    CHECKPOINT_SCHEMA_VERSION,
    LEGACY_CHECKPOINT_SCHEMA_VERSION,
    build_checkpoint_payload,
    checkpoint_model_config_from_payload,
    checkpoint_training_config_from_payload,
)


@dataclass(frozen=True)
class MigrationStepResult:
    source_version: str
    target_version: str
    changed_fields: list[str]
    added_fields: list[str]
    removed_fields: list[str]
    compatibility_warnings: list[str]
    migration_fingerprint: str
    payload: dict[str, Any]


def migrate_payload_v275_to_v276(payload: dict[str, Any]) -> MigrationStepResult:
    model_type = str(payload.get("model_type", "transformer"))
    model_config = checkpoint_model_config_from_payload(payload)
    training_config = checkpoint_training_config_from_payload(payload)
    state_dict = payload.get("state_dict", {})
    optimizer_state = payload.get("optimizer_state")
    state_hash = hash_state_dict(state_dict) if isinstance(state_dict, dict) else ""
    optimizer_hash = hash_optimizer_state(optimizer_state) if isinstance(optimizer_state, dict) else None

    migrated = build_checkpoint_payload(
        model_type=model_type,
        model_config=model_config,
        training_config=training_config,
        dataset_manifest_hash=str(payload.get("dataset_manifest_hash", "")),
        snapshot_hash=str(payload.get("snapshot_hash", "")),
        weights_hash=state_hash,
        optimizer_state_hash=optimizer_hash,
        eval_fingerprint=payload.get("eval_fingerprint"),
        curriculum_coverage=dict(payload.get("curriculum_coverage", {})),
        seed=int(payload.get("seed", 0)),
        created_at=float(payload.get("created_at", payload.get("timestamp", 0.0))),
        state_dict=state_dict if isinstance(state_dict, dict) else {},
        optimizer_state=optimizer_state if isinstance(optimizer_state, dict) else None,
        tokenizer=payload.get("tokenizer") if isinstance(payload.get("tokenizer"), dict) else None,
        extra=payload.get("extra") if isinstance(payload.get("extra"), dict) else None,
    )

    changed_fields = [field for field in ("schema_version", "model_config", "training_config", "weights_hash", "optimizer_state_hash", "artifact_fingerprint") if payload.get(field) != migrated.get(field)]
    added_fields = [field for field in migrated if field not in payload]
    removed_fields = [field for field in payload if field not in migrated and field not in {"config"}]
    warnings: list[str] = []
    if "config" in payload:
        warnings.append("legacy config field was normalized into model_config/training_config")

    migration_fingerprint = migrated["artifact_fingerprint"]
    return MigrationStepResult(
        source_version=LEGACY_CHECKPOINT_SCHEMA_VERSION,
        target_version=CHECKPOINT_SCHEMA_VERSION,
        changed_fields=sorted(changed_fields),
        added_fields=sorted(added_fields),
        removed_fields=sorted(removed_fields),
        compatibility_warnings=sorted(warnings),
        migration_fingerprint=migration_fingerprint,
        payload=migrated,
    )
