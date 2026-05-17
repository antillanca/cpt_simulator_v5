"""Artifact policy loader and validator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ArtifactPolicyError(ValueError):
    pass


class UnsupportedArtifactTypeError(ArtifactPolicyError):
    pass


class MissingRequiredArtifactFieldError(ArtifactPolicyError):
    pass


class ArtifactCompatibilityError(ArtifactPolicyError):
    pass


@dataclass(frozen=True)
class ArtifactTypePolicy:
    required_fields: tuple[str, ...]
    retention: dict[str, Any] = field(default_factory=dict)
    migration: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactPolicy:
    schema_version: str
    defaults: dict[str, Any]
    artifacts: dict[str, ArtifactTypePolicy]
    compatibility: dict[str, dict[str, bool]]
    enforcement: dict[str, Any]


_TOP_LEVEL_KEYS = {"schema_version", "defaults", "artifacts", "compatibility", "enforcement"}
_DEFAULT_KEYS = {"allow_legacy_read", "allow_legacy_write", "require_fingerprint", "require_manifest", "require_snapshot_hash"}
_ARTIFACT_KEYS = {"required_fields", "retention", "migration"}
_COMPAT_KEYS = {"read", "write"}
_ENFORCEMENT_KEYS = {"strict_mode", "fail_on_unknown_artifact", "fail_on_missing_fingerprint"}


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


def _ensure_mapping(value: Any, *, message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactPolicyError(message)
    return value


def _validate_unknown_keys(data: dict[str, Any], allowed: set[str], *, context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ArtifactPolicyError(f"Unknown keys in {context}: {', '.join(unknown)}")


def _validate_bool(value: Any, *, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ArtifactPolicyError(f"{field_name} must be a boolean")


def _artifact_version_key(version: str) -> str:
    normalized = version.strip().lower().replace(".", "")
    if normalized.startswith("v"):
        return normalized
    return f"v{normalized}"


def validate_artifact_policy_data(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ArtifactPolicyError("Policy file must contain a mapping at top level.")
    _validate_unknown_keys(data, _TOP_LEVEL_KEYS, context="top-level policy")
    for key in ("schema_version", "artifacts", "compatibility", "enforcement"):
        if key not in data:
            raise ArtifactPolicyError(f"Missing required policy key: {key}")
    if not isinstance(data["schema_version"], str) or not data["schema_version"].strip():
        raise ArtifactPolicyError("schema_version must be a non-empty string")

    defaults = _ensure_mapping(data.get("defaults", {}), message="defaults must be a mapping")
    _validate_unknown_keys(defaults, _DEFAULT_KEYS, context="defaults")
    for key in _DEFAULT_KEYS:
        if key in defaults:
            _validate_bool(defaults[key], field_name=f"defaults.{key}")

    artifacts = _ensure_mapping(data["artifacts"], message="artifacts must be a mapping")
    if not artifacts:
        raise ArtifactPolicyError("Policy artifacts must be a non-empty mapping.")
    for artifact_name, artifact_data in artifacts.items():
        if not isinstance(artifact_name, str) or not artifact_name.strip():
            raise ArtifactPolicyError("Artifact names must be non-empty strings")
        artifact_mapping = _ensure_mapping(artifact_data, message=f"artifact policy for {artifact_name} must be a mapping")
        _validate_unknown_keys(artifact_mapping, _ARTIFACT_KEYS, context=f"artifact policy {artifact_name}")
        if "required_fields" not in artifact_mapping:
            raise MissingRequiredArtifactFieldError(f"Missing required field: artifacts.{artifact_name}.required_fields")
        required_fields = artifact_mapping["required_fields"]
        if not isinstance(required_fields, list) or not required_fields or not all(isinstance(item, str) and item.strip() for item in required_fields):
            raise MissingRequiredArtifactFieldError(
                f"artifacts.{artifact_name}.required_fields must be a non-empty list of strings"
            )
        for section_name in ("retention", "migration"):
            section = _ensure_mapping(artifact_mapping.get(section_name, {}), message=f"artifacts.{artifact_name}.{section_name} must be a mapping")
            artifact_mapping[section_name] = section

    compatibility = _ensure_mapping(data["compatibility"], message="compatibility must be a mapping")
    if not compatibility:
        raise ArtifactPolicyError("compatibility must be a non-empty mapping.")
    for version_name, version_rules in compatibility.items():
        if not isinstance(version_name, str) or not version_name.strip():
            raise ArtifactPolicyError("Compatibility keys must be non-empty strings")
        rules = _ensure_mapping(version_rules, message=f"compatibility rules for {version_name} must be a mapping")
        _validate_unknown_keys(rules, _COMPAT_KEYS, context=f"compatibility rules {version_name}")
        for rule_name in _COMPAT_KEYS:
            if rule_name in rules:
                _validate_bool(rules[rule_name], field_name=f"compatibility.{version_name}.{rule_name}")

    enforcement = _ensure_mapping(data["enforcement"], message="enforcement must be a mapping")
    _validate_unknown_keys(enforcement, _ENFORCEMENT_KEYS, context="enforcement")
    for key in _ENFORCEMENT_KEYS:
        if key in enforcement:
            _validate_bool(enforcement[key], field_name=f"enforcement.{key}")


def _sorted_policy_payload(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": str(data["schema_version"]),
        "defaults": dict(sorted(_normalize(data.get("defaults", {})).items())),
        "artifacts": {
            name: {
                "required_fields": tuple(item.get("required_fields", ())),
                "retention": dict(sorted(_normalize(item.get("retention", {})).items())),
                "migration": dict(sorted(_normalize(item.get("migration", {})).items())),
            }
            for name, item in sorted(data["artifacts"].items())
        },
        "compatibility": {
            version: dict(sorted(_normalize(rules).items())) for version, rules in sorted(data["compatibility"].items())
        },
        "enforcement": dict(sorted(_normalize(data.get("enforcement", {})).items())),
    }


def load_artifact_policy(path: Path) -> ArtifactPolicy:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactPolicyError("Policy file must contain a mapping at top level.")
    validate_artifact_policy_data(payload)
    normalized = _sorted_policy_payload(payload)
    artifacts = {
        name: ArtifactTypePolicy(
            required_fields=tuple(item["required_fields"]),
            retention=dict(item["retention"]),
            migration=dict(item["migration"]),
        )
        for name, item in normalized["artifacts"].items()
    }
    policy = ArtifactPolicy(
        schema_version=normalized["schema_version"],
        defaults=dict(normalized["defaults"]),
        artifacts=artifacts,
        compatibility={version: dict(rules) for version, rules in normalized["compatibility"].items()},
        enforcement=dict(normalized["enforcement"]),
    )
    validate_artifact_policy(policy)
    return policy


def validate_artifact_policy(policy: ArtifactPolicy) -> None:
    if not isinstance(policy, ArtifactPolicy):
        raise ArtifactPolicyError("policy must be an ArtifactPolicy")
    if not isinstance(policy.schema_version, str) or not policy.schema_version.strip():
        raise ArtifactPolicyError("policy.schema_version must be a non-empty string")
    if not isinstance(policy.defaults, dict):
        raise ArtifactPolicyError("policy.defaults must be a mapping")
    if not policy.artifacts:
        raise ArtifactPolicyError("policy.artifacts must be non-empty")
    _validate_unknown_keys(policy.defaults, _DEFAULT_KEYS, context="policy.defaults")
    for key in _DEFAULT_KEYS:
        if key in policy.defaults:
            _validate_bool(policy.defaults[key], field_name=f"policy.defaults.{key}")
    for artifact_name, artifact_policy in policy.artifacts.items():
        if not artifact_policy.required_fields:
            raise MissingRequiredArtifactFieldError(f"Missing required field: {artifact_name}.required_fields")
        if not all(isinstance(item, str) and item.strip() for item in artifact_policy.required_fields):
            raise ArtifactPolicyError(f"{artifact_name}.required_fields must contain only strings")
        if not isinstance(artifact_policy.retention, dict):
            raise ArtifactPolicyError(f"{artifact_name}.retention must be a mapping")
        if not isinstance(artifact_policy.migration, dict):
            raise ArtifactPolicyError(f"{artifact_name}.migration must be a mapping")
    if not policy.compatibility:
        raise ArtifactPolicyError("policy.compatibility must be non-empty")
    if not isinstance(policy.compatibility, dict):
        raise ArtifactPolicyError("policy.compatibility must be a mapping")
    if not isinstance(policy.enforcement, dict):
        raise ArtifactPolicyError("policy.enforcement must be a mapping")
    _validate_unknown_keys(policy.enforcement, _ENFORCEMENT_KEYS, context="policy.enforcement")
    for key in _ENFORCEMENT_KEYS:
        if key in policy.enforcement:
            _validate_bool(policy.enforcement[key], field_name=f"policy.enforcement.{key}")


def get_artifact_policy(artifact_type: str, policy: ArtifactPolicy) -> ArtifactTypePolicy:
    if artifact_type not in policy.artifacts:
        raise UnsupportedArtifactTypeError(f"Unknown artifact type: {artifact_type}")
    return policy.artifacts[artifact_type]


def artifact_policy_fingerprint(policy: ArtifactPolicy) -> str:
    validate_artifact_policy(policy)
    payload = {
        "schema_version": policy.schema_version,
        "defaults": dict(sorted(_normalize(policy.defaults).items())),
        "artifacts": {
            name: {
                "required_fields": list(item.required_fields),
                "retention": dict(sorted(_normalize(item.retention).items())),
                "migration": dict(sorted(_normalize(item.migration).items())),
            }
            for name, item in sorted(policy.artifacts.items())
        },
        "compatibility": {version: dict(sorted(_normalize(rules).items())) for version, rules in sorted(policy.compatibility.items())},
        "enforcement": dict(sorted(_normalize(policy.enforcement).items())),
    }
    return _stable_hash(payload)


def policy_allows_version(policy: ArtifactPolicy, version: str, *, write: bool = False) -> bool:
    validate_artifact_policy(policy)
    key = _artifact_version_key(version)
    rules = policy.compatibility.get(key)
    if rules is None:
        return False
    return bool(rules.get("write" if write else "read", False))
