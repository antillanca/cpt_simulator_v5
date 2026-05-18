"""Dataset manifest for CPT v2.7 distillation readiness.

Every export generates a manifest with deterministic field ordering,
full provenance, and a fingerprint that changes when any input changes.
"""

from __future__ import annotations

import hashlib
import json
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.governance.artifact_policy import ArtifactPolicy, ArtifactPolicyError, MissingRequiredArtifactFieldError, get_artifact_policy

MANIFEST_SCHEMA_VERSION = "2.7.0"


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass
class DatasetManifest:
    """Deterministic manifest for an exported dataset."""

    dataset_version: str = "2.7.0"
    schema_version: str = MANIFEST_SCHEMA_VERSION
    snapshot_hash: str = ""
    module_hash: str = ""
    curriculum_coverage: list[int] = field(default_factory=list)
    generation_seed: int = 0
    record_count: int = 0
    shard_list: list[str] = field(default_factory=list)
    benchmark_fingerprint: str = ""
    modules_used: list[str] = field(default_factory=list)
    module_versions: dict[str, str] = field(default_factory=dict)
    parameter_sweeps: dict[str, list[Any]] = field(default_factory=dict)
    timestamp: str = ""
    timestamp_unix: float = 0.0
    generator_version: str = "oracle-v2.7"
    fingerprint: str = ""

    def compute_fingerprint(self) -> str:
        """Compute deterministic fingerprint from all manifest fields except fingerprint itself."""
        payload = {
            "dataset_version": self.dataset_version,
            "schema_version": self.schema_version,
            "snapshot_hash": self.snapshot_hash,
            "module_hash": self.module_hash,
            "curriculum_coverage": sorted(self.curriculum_coverage),
            "generation_seed": self.generation_seed,
            "record_count": self.record_count,
            "shard_list": sorted(self.shard_list),
            "benchmark_fingerprint": self.benchmark_fingerprint,
            "modules_used": sorted(self.modules_used),
            "module_versions": dict(sorted(self.module_versions.items())),
            "parameter_sweeps": {k: sorted(str(v) for v in vs) for k, vs in sorted(self.parameter_sweeps.items())},
            "timestamp": self.timestamp,
            "timestamp_unix": self.timestamp_unix,
            "generator_version": self.generator_version,
        }
        return _stable_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict with deterministic field ordering."""
        self.fingerprint = self.compute_fingerprint()
        return {
            "dataset_version": self.dataset_version,
            "schema_version": self.schema_version,
            "snapshot_hash": self.snapshot_hash,
            "module_hash": self.module_hash,
            "curriculum_coverage": sorted(self.curriculum_coverage),
            "generation_seed": self.generation_seed,
            "record_count": self.record_count,
            "shard_list": sorted(self.shard_list),
            "benchmark_fingerprint": self.benchmark_fingerprint,
            "modules_used": sorted(self.modules_used),
            "module_versions": dict(sorted(self.module_versions.items())),
            "parameter_sweeps": {k: sorted(str(v) for v in vs) for k, vs in sorted(self.parameter_sweeps.items())},
            "timestamp": self.timestamp,
            "timestamp_unix": self.timestamp_unix,
            "generator_version": self.generator_version,
            "fingerprint": self.fingerprint,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def save(self, path: str | Path, *, policy: ArtifactPolicy | None = None, strict_policy: bool = False) -> Path:
        if policy is not None:
            validate_manifest(self.to_dict(), policy=policy, strict_policy=strict_policy)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        return cls(
            dataset_version=data.get("dataset_version", "2.7.0"),
            schema_version=data.get("schema_version", MANIFEST_SCHEMA_VERSION),
            snapshot_hash=data.get("snapshot_hash", ""),
            module_hash=data.get("module_hash", ""),
            curriculum_coverage=data.get("curriculum_coverage", []),
            generation_seed=data.get("generation_seed", 0),
            record_count=data.get("record_count", 0),
            shard_list=data.get("shard_list", []),
            benchmark_fingerprint=data.get("benchmark_fingerprint", ""),
            modules_used=data.get("modules_used", []),
            module_versions=data.get("module_versions", {}),
            parameter_sweeps=data.get("parameter_sweeps", {}),
            timestamp=data.get("timestamp", ""),
            timestamp_unix=data.get("timestamp_unix", 0.0),
            generator_version=data.get("generator_version", "oracle-v2.7"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> DatasetManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = cls.from_dict(data)
        manifest.fingerprint = data.get("fingerprint", "")
        return manifest

    @classmethod
    def from_oracle_result(
        cls,
        output_path: str | Path,
        modules_used: list[str],
        module_versions: dict[str, str],
        seed: int,
        snapshot_hash: str,
        module_hash: str,
        record_count: int,
        parameter_sweeps: dict[str, list[Any]] | None = None,
        curriculum_coverage: list[int] | None = None,
        benchmark_fingerprint: str = "",
    ) -> DatasetManifest:
        """Build manifest from oracle generation result."""
        now = _time.strftime("%Y-%m-%dT%H:%M:%S", _time.gmtime())
        now_unix = _time.time()
        manifest = cls(
            dataset_version="2.7.0",
            snapshot_hash=snapshot_hash,
            module_hash=module_hash,
            curriculum_coverage=sorted(curriculum_coverage or []),
            generation_seed=seed,
            record_count=record_count,
            shard_list=[],
            benchmark_fingerprint=benchmark_fingerprint,
            modules_used=sorted(modules_used),
            module_versions=dict(sorted(module_versions.items())),
            parameter_sweeps=parameter_sweeps or {},
            timestamp=now,
            timestamp_unix=now_unix,
        )
        manifest.fingerprint = manifest.compute_fingerprint()
        return manifest


def validate_manifest(
    data: dict[str, Any],
    *,
    policy: ArtifactPolicy | None = None,
    strict_policy: bool = False,
) -> list[str]:
    """Validate a manifest dict has all required fields. Returns list of errors."""
    required = [
        "dataset_version", "schema_version", "snapshot_hash", "module_hash",
        "generation_seed", "record_count", "fingerprint",
    ]
    errors = []
    for key in required:
        if key not in data:
            errors.append(f"Missing required field: {key}")
    if "fingerprint" in data:
        recomputed = DatasetManifest.from_dict(data).compute_fingerprint()
        if data["fingerprint"] != recomputed:
            errors.append(f"Fingerprint mismatch: stored={data['fingerprint'][:16]} computed={recomputed[:16]}")
    if policy is not None:
        try:
            artifact_policy = get_artifact_policy("dataset", policy)
            enforce_now = strict_policy or bool(policy.enforcement.get("strict_mode", False))
            if enforce_now:
                for field_name in artifact_policy.required_fields:
                    if field_name not in data or data.get(field_name) in (None, ""):
                        raise MissingRequiredArtifactFieldError(f"Missing required field: {field_name}")
        except ArtifactPolicyError as exc:
            errors.append(str(exc))
    return errors
