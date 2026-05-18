"""Checkpoint fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from backend.neural.checkpoints.schema import compute_checkpoint_fingerprint


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _normalize(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu().contiguous()
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "hash": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def hash_state_dict(state_dict: dict[str, Any]) -> str:
    payload = {key: _normalize(state_dict[key]) for key in sorted(state_dict)}
    return _stable_hash(payload)


def hash_optimizer_state(optimizer_state: dict[str, Any] | None) -> str | None:
    if optimizer_state is None:
        return None
    return _stable_hash(_normalize(optimizer_state))


def checkpoint_artifact_fingerprint(payload: dict[str, Any]) -> str:
    return compute_checkpoint_fingerprint(payload)
