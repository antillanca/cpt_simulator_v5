"""Checkpoint governance for tiny-model artifacts."""

from backend.neural.checkpoints.fingerprint import checkpoint_artifact_fingerprint, hash_optimizer_state, hash_state_dict
from backend.neural.checkpoints.migrate import MigrationResult, migrate_checkpoint
from backend.neural.checkpoints.schema import (
    CHECKPOINT_REQUIRED_FIELDS,
    CHECKPOINT_SCHEMA_VERSION,
    LEGACY_CHECKPOINT_SCHEMA_VERSION,
    build_checkpoint_payload,
    checkpoint_model_config_from_payload,
    checkpoint_summary,
    checkpoint_training_config_from_payload,
    ordered_checkpoint_dict,
)
from backend.neural.checkpoints.validator import CheckpointValidationError, ensure_checkpoint_payload, infer_checkpoint_version, validate_checkpoint_payload

__all__ = [
    "CHECKPOINT_REQUIRED_FIELDS",
    "CHECKPOINT_SCHEMA_VERSION",
    "LEGACY_CHECKPOINT_SCHEMA_VERSION",
    "build_checkpoint_payload",
    "checkpoint_artifact_fingerprint",
    "checkpoint_model_config_from_payload",
    "checkpoint_summary",
    "checkpoint_training_config_from_payload",
    "ordered_checkpoint_dict",
    "hash_optimizer_state",
    "hash_state_dict",
    "CheckpointValidationError",
    "ensure_checkpoint_payload",
    "infer_checkpoint_version",
    "validate_checkpoint_payload",
    "MigrationResult",
    "migrate_checkpoint",
]
