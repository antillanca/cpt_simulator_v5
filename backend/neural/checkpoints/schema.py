"""Formal checkpoint schema for governed tiny-model artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

CHECKPOINT_SCHEMA_VERSION = "2.7.6"
LEGACY_CHECKPOINT_SCHEMA_VERSION = "2.7.5"

CHECKPOINT_REQUIRED_FIELDS = [
    "schema_version",
    "model_type",
    "model_config",
    "training_config",
    "dataset_manifest_hash",
    "snapshot_hash",
    "weights_hash",
    "optimizer_state_hash",
    "eval_fingerprint",
    "curriculum_coverage",
    "seed",
    "created_at",
    "artifact_fingerprint",
]


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def checkpoint_model_config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract a deterministic model config from a checkpoint payload."""
    config = payload.get("model_config")
    if isinstance(config, dict):
        return dict(sorted((_normalize(config)).items()))

    legacy_config = payload.get("config", {})
    if isinstance(legacy_config, dict):
        model_keys = ("vocab_size", "hidden_size", "n_heads", "n_layers", "model_type")
        model_config = {key: legacy_config[key] for key in model_keys if key in legacy_config}
        if model_config:
            return dict(sorted(_normalize(model_config).items()))
    return {}


def checkpoint_training_config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("training_config")
    if isinstance(config, dict):
        return dict(sorted(_normalize(config).items()))
    legacy_config = payload.get("config", {})
    if isinstance(legacy_config, dict):
        return dict(sorted(_normalize(legacy_config).items()))
    return {}


def ordered_checkpoint_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Return checkpoint payload with deterministic top-level ordering."""
    ordered: dict[str, Any] = {}
    for field_name in CHECKPOINT_REQUIRED_FIELDS:
        if field_name in payload:
            ordered[field_name] = payload[field_name]
    for field_name in ("state_dict", "optimizer_state", "tokenizer", "config", "extra"):
        if field_name in payload:
            ordered[field_name] = _normalize(payload[field_name])
    return ordered


def compute_checkpoint_fingerprint(payload: dict[str, Any]) -> str:
    """Compute deterministic fingerprint for canonical checkpoint metadata."""
    canonical = {
        "schema_version": payload.get("schema_version", CHECKPOINT_SCHEMA_VERSION),
        "model_type": payload.get("model_type", ""),
        "model_config": _normalize(payload.get("model_config", {})),
        "training_config": _normalize(payload.get("training_config", payload.get("config", {}))),
        "dataset_manifest_hash": payload.get("dataset_manifest_hash", ""),
        "snapshot_hash": payload.get("snapshot_hash", ""),
        "weights_hash": payload.get("weights_hash", ""),
        "optimizer_state_hash": payload.get("optimizer_state_hash"),
        "eval_fingerprint": payload.get("eval_fingerprint"),
        "curriculum_coverage": _normalize(payload.get("curriculum_coverage", {})),
        "seed": int(payload.get("seed", 0)),
        "created_at": float(payload.get("created_at", 0.0)),
    }
    return _stable_hash(canonical)


def build_checkpoint_payload(
    *,
    model_type: str,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    dataset_manifest_hash: str,
    snapshot_hash: str,
    weights_hash: str,
    optimizer_state_hash: str | None,
    eval_fingerprint: str | None,
    curriculum_coverage: dict[str, Any],
    seed: int,
    created_at: float,
    state_dict: dict[str, Any],
    optimizer_state: dict[str, Any] | None = None,
    tokenizer: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": model_type,
        "model_config": dict(sorted(_normalize(model_config).items())),
        "training_config": dict(sorted(_normalize(training_config).items())),
        "dataset_manifest_hash": dataset_manifest_hash,
        "snapshot_hash": snapshot_hash,
        "weights_hash": weights_hash,
        "optimizer_state_hash": optimizer_state_hash,
        "eval_fingerprint": eval_fingerprint,
        "curriculum_coverage": _normalize(curriculum_coverage),
        "seed": int(seed),
        "created_at": float(created_at),
        "state_dict": state_dict,
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if tokenizer is not None:
        payload["tokenizer"] = tokenizer
    if extra is not None:
        payload["extra"] = extra
    payload["artifact_fingerprint"] = compute_checkpoint_fingerprint(payload)
    return ordered_checkpoint_dict(payload)


def checkpoint_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a compact deterministic summary of checkpoint metadata."""
    return {
        "schema_version": payload.get("schema_version", ""),
        "model_type": payload.get("model_type", ""),
        "model_config": _normalize(payload.get("model_config", {})),
        "training_config": _normalize(payload.get("training_config", payload.get("config", {}))),
        "dataset_manifest_hash": payload.get("dataset_manifest_hash", ""),
        "snapshot_hash": payload.get("snapshot_hash", ""),
        "weights_hash": payload.get("weights_hash", ""),
        "optimizer_state_hash": payload.get("optimizer_state_hash"),
        "eval_fingerprint": payload.get("eval_fingerprint"),
        "curriculum_coverage": _normalize(payload.get("curriculum_coverage", {})),
        "seed": int(payload.get("seed", 0)),
        "created_at": float(payload.get("created_at", 0.0)),
        "artifact_fingerprint": payload.get("artifact_fingerprint", ""),
    }
