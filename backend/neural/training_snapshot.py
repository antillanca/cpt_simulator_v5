"""Deterministic training snapshot metadata for circuit surrogate runs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import torch
except Exception:  # pragma: no cover - torch is expected in normal runs
    torch = None  # type: ignore[assignment]


SNAPSHOT_SCHEMA_VERSION = "2.9b"


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def stable_fingerprint(payload: Any) -> str:
    data = json.dumps(_normalize(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def fingerprint_jsonl(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return stable_fingerprint({"path": str(path), "missing": True})
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return stable_fingerprint({"path": str(path), "lines": lines})


def fingerprint_mapping(mapping: Mapping[str, Any]) -> str:
    return stable_fingerprint(dict(mapping))


def git_commit_hash(repo_root: str | Path | None = None) -> str:
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


@dataclass(frozen=True)
class TrainingSnapshot:
    seed: int
    dataset_fingerprint: str
    config_fingerprint: str
    git_commit: str
    model_fingerprint: str
    torch_version: str
    cuda_enabled: bool
    device_name: str

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "seed": int(self.seed),
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "git_commit": self.git_commit,
            "model_fingerprint": self.model_fingerprint,
            "torch_version": self.torch_version,
            "cuda_enabled": bool(self.cuda_enabled),
            "device_name": self.device_name,
        }

    def fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.canonical_payload())
        payload["artifact_fingerprint"] = self.fingerprint()
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)

    def export(self, output_dir: str | Path, filename: str = "training_snapshot.json") -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / filename
        output.write_text(self.to_json(), encoding="utf-8")
        return output

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        dataset_fingerprint: str,
        config: Mapping[str, Any],
        model_fingerprint: str,
        repo_root: str | Path | None = None,
        torch_version: str | None = None,
        cuda_enabled: bool | None = None,
        device_name: str | None = None,
    ) -> "TrainingSnapshot":
        if torch_version is None:
            torch_version = getattr(torch, "__version__", "unknown") if torch is not None else "unknown"
        if cuda_enabled is None:
            cuda_enabled = bool(torch is not None and torch.cuda.is_available())
        if device_name is None:
            if torch is not None and torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(torch.cuda.current_device())
            else:
                device_name = "cpu"
        clean_config = {k: v for k, v in config.items() if k != "output"}
        return cls(
            seed=int(seed),
            dataset_fingerprint=dataset_fingerprint,
            config_fingerprint=fingerprint_mapping(clean_config),
            git_commit=git_commit_hash(repo_root),
            model_fingerprint=model_fingerprint,
            torch_version=str(torch_version),
            cuda_enabled=bool(cuda_enabled),
            device_name=str(device_name),
        )


def load_training_snapshot(path: str | Path) -> TrainingSnapshot:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrainingSnapshot(
        seed=int(payload["seed"]),
        dataset_fingerprint=str(payload["dataset_fingerprint"]),
        config_fingerprint=str(payload["config_fingerprint"]),
        git_commit=str(payload.get("git_commit", "")),
        model_fingerprint=str(payload["model_fingerprint"]),
        torch_version=str(payload["torch_version"]),
        cuda_enabled=bool(payload["cuda_enabled"]),
        device_name=str(payload["device_name"]),
    )
