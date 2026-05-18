"""Checkpoint validation for governed tiny-model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.governance.artifact_policy import (
    ArtifactCompatibilityError,
    ArtifactPolicy,
    ArtifactPolicyError,
    MissingRequiredArtifactFieldError,
    UnsupportedArtifactTypeError,
    get_artifact_policy,
    policy_allows_version,
)
from backend.neural.checkpoints.schema import (
    CHECKPOINT_REQUIRED_FIELDS,
    CHECKPOINT_SCHEMA_VERSION,
    LEGACY_CHECKPOINT_SCHEMA_VERSION,
    compute_checkpoint_fingerprint,
    checkpoint_model_config_from_payload,
    checkpoint_training_config_from_payload,
)


class CheckpointValidationError(ValueError):
    """Raised when a checkpoint fails schema validation."""


def infer_checkpoint_version(payload: dict[str, Any]) -> str:
    version = payload.get("schema_version")
    if isinstance(version, str) and version.strip():
        return version
    if "state_dict" in payload and "tokenizer" in payload and ("config" in payload or "training_config" in payload):
        return LEGACY_CHECKPOINT_SCHEMA_VERSION
    return "unknown"


def validate_checkpoint_payload(payload: dict[str, Any], *, allow_legacy: bool = False) -> list[str]:
    return validate_checkpoint_payload_with_policy(payload, allow_legacy=allow_legacy, policy=None, strict_policy=False)


def validate_checkpoint_payload_with_policy(
    payload: dict[str, Any],
    *,
    allow_legacy: bool = False,
    policy: ArtifactPolicy | None = None,
    strict_policy: bool = False,
    artifact_type: str = "checkpoint",
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["Checkpoint payload must be a mapping."]

    version = infer_checkpoint_version(payload)
    validation_payload = payload
    legacy_read = allow_legacy and version == LEGACY_CHECKPOINT_SCHEMA_VERSION
    if legacy_read:
        from backend.neural.checkpoints.migrations.v2_7_5_to_v2_7_6 import migrate_payload_v275_to_v276

        validation_payload = migrate_payload_v275_to_v276(payload).payload
    if version == "unknown":
        errors.append("Unable to infer checkpoint schema version.")
    elif version != CHECKPOINT_SCHEMA_VERSION and not allow_legacy:
        errors.append(f"Unsupported checkpoint schema version: {version}")

    if not legacy_read:
        for field_name in CHECKPOINT_REQUIRED_FIELDS:
            if field_name not in validation_payload:
                errors.append(f"Missing required field: {field_name}")

    if "schema_version" in validation_payload and validation_payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        if not allow_legacy:
            errors.append(f"schema_version must be {CHECKPOINT_SCHEMA_VERSION}")

    if "model_type" in validation_payload and not isinstance(validation_payload["model_type"], str):
        errors.append("model_type must be a string")
    if "model_config" in validation_payload and not isinstance(validation_payload["model_config"], dict):
        errors.append("model_config must be a mapping")
    if "training_config" in validation_payload and not isinstance(validation_payload["training_config"], dict):
        errors.append("training_config must be a mapping")
    if "dataset_manifest_hash" in validation_payload and not isinstance(validation_payload["dataset_manifest_hash"], str):
        errors.append("dataset_manifest_hash must be a string")
    if "snapshot_hash" in validation_payload and not isinstance(validation_payload["snapshot_hash"], str):
        errors.append("snapshot_hash must be a string")
    if "weights_hash" in validation_payload and not isinstance(validation_payload["weights_hash"], str):
        errors.append("weights_hash must be a string")
    if "optimizer_state_hash" in validation_payload and validation_payload["optimizer_state_hash"] is not None and not isinstance(validation_payload["optimizer_state_hash"], str):
        errors.append("optimizer_state_hash must be a string or null")
    if "eval_fingerprint" in validation_payload and validation_payload["eval_fingerprint"] is not None and not isinstance(validation_payload["eval_fingerprint"], str):
        errors.append("eval_fingerprint must be a string or null")
    if "curriculum_coverage" in validation_payload and not isinstance(validation_payload["curriculum_coverage"], dict):
        errors.append("curriculum_coverage must be a mapping")
    if "seed" in validation_payload and not isinstance(validation_payload["seed"], int):
        errors.append("seed must be an integer")
    if "created_at" in validation_payload and not isinstance(validation_payload["created_at"], (int, float)):
        errors.append("created_at must be numeric")
    if "artifact_fingerprint" in validation_payload and not isinstance(validation_payload["artifact_fingerprint"], str):
        errors.append("artifact_fingerprint must be a string")

    expected = compute_checkpoint_fingerprint(validation_payload)
    if validation_payload.get("artifact_fingerprint") and validation_payload["artifact_fingerprint"] != expected:
        errors.append("artifact_fingerprint mismatch")

    if policy is not None and not legacy_read:
        try:
            enforce_checkpoint_policy(
                payload,
                policy,
                strict_policy=strict_policy,
                allow_legacy=allow_legacy,
                artifact_type=artifact_type,
            )
        except ArtifactPolicyError as exc:
            errors.append(str(exc))

    return errors


def ensure_checkpoint_payload(payload: dict[str, Any], *, allow_legacy: bool = False) -> dict[str, Any]:
    errors = validate_checkpoint_payload(payload, allow_legacy=allow_legacy)
    if errors:
        raise CheckpointValidationError("; ".join(errors))
    return payload


def enforce_checkpoint_policy(
    payload: dict[str, Any],
    policy: ArtifactPolicy,
    *,
    strict_policy: bool = False,
    allow_legacy: bool = False,
    artifact_type: str = "checkpoint",
) -> None:
    artifact_policy = get_artifact_policy(artifact_type, policy)
    version = infer_checkpoint_version(payload)
    enforce_now = strict_policy or bool(policy.enforcement.get("strict_mode", False))
    if not enforce_now:
        return
    legacy_read = version == LEGACY_CHECKPOINT_SCHEMA_VERSION and allow_legacy
    if legacy_read:
        from backend.neural.checkpoints.migrations.v2_7_5_to_v2_7_6 import migrate_payload_v275_to_v276

        payload = migrate_payload_v275_to_v276(payload).payload
    if policy.enforcement.get("fail_on_missing_fingerprint", False) and not payload.get("artifact_fingerprint"):
        raise MissingRequiredArtifactFieldError("Missing required field: artifact_fingerprint")
    if policy.enforcement.get("fail_on_unknown_artifact", False) and version == "unknown":
        raise ArtifactCompatibilityError("Unknown checkpoint artifact version")

    if payload.get("schema_version") and payload["schema_version"] not in (CHECKPOINT_SCHEMA_VERSION, LEGACY_CHECKPOINT_SCHEMA_VERSION):
        raise ArtifactCompatibilityError(f"Unsupported checkpoint schema version: {payload['schema_version']}")
    if not allow_legacy and version != CHECKPOINT_SCHEMA_VERSION:
        raise ArtifactCompatibilityError(f"Unsupported checkpoint schema version: {version}")
    if not policy_allows_version(policy, payload.get("schema_version", version), write=False):
        raise ArtifactCompatibilityError(f"Checkpoint version {payload.get('schema_version', version)} is not allowed by policy")

    if not legacy_read:
        for field_name in artifact_policy.required_fields:
            if field_name not in payload or payload.get(field_name) in (None, ""):
                raise MissingRequiredArtifactFieldError(f"Missing required field: {field_name}")

        if policy.defaults.get("require_manifest", False) and not payload.get("dataset_manifest_hash"):
            raise MissingRequiredArtifactFieldError("Missing required field: dataset_manifest_hash")
        if policy.defaults.get("require_snapshot_hash", False) and not payload.get("snapshot_hash"):
            raise MissingRequiredArtifactFieldError("Missing required field: snapshot_hash")
        if policy.defaults.get("require_fingerprint", False) and not payload.get("artifact_fingerprint"):
            raise MissingRequiredArtifactFieldError("Missing required field: artifact_fingerprint")


def summarize_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload.get("schema_version", ""),
        "model_type": payload.get("model_type", ""),
        "model_config": checkpoint_model_config_from_payload(payload),
        "training_config": checkpoint_training_config_from_payload(payload),
        "dataset_manifest_hash": payload.get("dataset_manifest_hash", ""),
        "snapshot_hash": payload.get("snapshot_hash", ""),
        "weights_hash": payload.get("weights_hash", ""),
        "optimizer_state_hash": payload.get("optimizer_state_hash"),
        "eval_fingerprint": payload.get("eval_fingerprint"),
        "curriculum_coverage": payload.get("curriculum_coverage", {}),
        "seed": payload.get("seed", 0),
        "created_at": payload.get("created_at", 0.0),
        "artifact_fingerprint": payload.get("artifact_fingerprint", ""),
    }
