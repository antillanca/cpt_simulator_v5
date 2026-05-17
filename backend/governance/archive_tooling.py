"""Deterministic archive/export tooling for CPT artifacts."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from backend.governance.artifact_policy import ArtifactPolicy, artifact_policy_fingerprint
from backend.governance.archive_manifest import (
    ArchiveManifestError,
    build_archive_manifest,
    fingerprint_archive_manifest,
    validate_archive_manifest,
)


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
class ArchiveEntry:
    source_path: Path
    bundle_path: Path
    artifact_type: str
    schema_version: str
    fingerprint: str
    size_bytes: int
    created_at: float
    compatibility_status: str = "compatible"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "bundle_path": str(self.bundle_path),
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "size_bytes": int(self.size_bytes),
            "created_at": float(self.created_at),
            "compatibility_status": self.compatibility_status,
        }


def _archive_entries(paths: Iterable[Path]) -> list[ArchiveEntry]:
    entries: list[ArchiveEntry] = []
    for path in sorted({Path(item) for item in paths}, key=lambda p: str(p)):
        stat = path.stat()
        payload = _read_json_if_possible(path) if path.is_file() else {}
        entries.append(
            ArchiveEntry(
                source_path=path,
                bundle_path=Path("artifacts") / f"{len(entries):04d}_{path.name}",
                artifact_type=str(payload.get("artifact_type", _guess_artifact_type(path))),
                schema_version=str(payload.get("schema_version", payload.get("dataset_version", ""))),
                fingerprint=str(
                    payload.get(
                        "fingerprint",
                        payload.get(
                            "artifact_fingerprint",
                            payload.get("report_fingerprint", _stable_hash({"path": str(path), "size": stat.st_size, "mtime": stat.st_mtime})),
                        ),
                    )
                ),
                size_bytes=int(stat.st_size),
                created_at=float(payload.get("created_at", payload.get("timestamp_unix", stat.st_mtime))),
                compatibility_status=str(payload.get("compatibility_status", "compatible")),
            )
        )
    return entries


def _bundle_manifest(entries: list[ArchiveEntry], *, policy: ArtifactPolicy, source_snapshot_hash: str, bundle_version: str = "1.0", created_at: float = 0.0) -> dict[str, Any]:
    payload = {
        "bundle_version": bundle_version,
        "created_at": float(created_at),
        "policy_fingerprint": artifact_policy_fingerprint(policy),
        "artifact_count": len(entries),
        "artifacts": [entry.to_dict() for entry in entries],
        "source_snapshot_hash": source_snapshot_hash,
        "compatibility_matrix": _normalize(policy.compatibility),
    }
    payload["bundle_fingerprint"] = fingerprint_archive_manifest(payload)
    validate_archive_manifest(payload)
    return payload


def _write_tar(bundle_path: Path, entries: list[ArchiveEntry], manifest: dict[str, Any], *, compression: str) -> Path:
    bundle_path = Path(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "bundle.tar"
        with tarfile.open(tar_path, mode="w") as tar:
            manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
            info = tarfile.TarInfo("bundle_manifest.json")
            info.size = len(manifest_bytes)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            tar.addfile(info, io.BytesIO(manifest_bytes))
            policy_bytes = json.dumps(
                {
                    "policy_fingerprint": manifest["policy_fingerprint"],
                    "compatibility_matrix": manifest.get("compatibility_matrix", {}),
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
            info = tarfile.TarInfo("policy_snapshot.json")
            info.size = len(policy_bytes)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            tar.addfile(info, io.BytesIO(policy_bytes))
            for entry in entries:
                arcname = f"artifacts/{entry.bundle_path.name}"
                info = tar.gettarinfo(str(entry.source_path), arcname=arcname)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with entry.source_path.open("rb") as fp:
                    tar.addfile(info, fp)
        if compression == "zst":
            try:
                import zstandard as zstd
            except Exception:
                if bundle_path.name.endswith(".tar.zst"):
                    bundle_path = bundle_path.with_name(bundle_path.name[:-4] + ".gz")
                elif bundle_path.suffix == ".zst":
                    bundle_path = bundle_path.with_name(bundle_path.stem + ".tar.gz")
                else:
                    bundle_path = bundle_path.with_suffix(bundle_path.suffix + ".gz")
            else:
                cctx = zstd.ZstdCompressor(level=3)
                with tar_path.open("rb") as src, bundle_path.open("wb") as dst:
                    with cctx.stream_writer(dst) as compressor:
                        shutil.copyfileobj(src, compressor)
                return bundle_path
        with bundle_path.open("wb") as raw_dst:
            with tar_path.open("rb") as src, gzip.GzipFile(filename="", mode="wb", fileobj=raw_dst, mtime=0) as dst:
                shutil.copyfileobj(src, dst)
    return bundle_path


def create_artifact_bundle(
    paths: Iterable[Path],
    output_path: Path,
    *,
    policy: ArtifactPolicy,
    source_snapshot_hash: str = "",
    created_at: float = 0.0,
) -> tuple[Path, dict[str, Any]]:
    entries = _archive_entries(paths)
    if not entries:
        raise ArchiveManifestError("No artifacts provided for bundle creation.")
    output_path = Path(output_path)
    if output_path.suffix == ".zst":
        compression = "zst"
    else:
        compression = "gz"
        if output_path.suffix not in {".gz", ".tar", ".tgz", ".zst"}:
            output_path = output_path.with_suffix(".tar.gz")
    manifest = _bundle_manifest(entries, policy=policy, source_snapshot_hash=source_snapshot_hash or _stable_hash([entry.fingerprint for entry in entries]), created_at=created_at)
    bundle_path = _write_tar(output_path, entries, manifest, compression=compression)
    manifest_path = bundle_path.with_suffix(bundle_path.suffix + ".manifest.json") if bundle_path.suffix else bundle_path.with_name(bundle_path.name + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return bundle_path, manifest


def export_artifact_bundle(root: Path, output_path: Path, *, policy: ArtifactPolicy) -> tuple[Path, dict[str, Any]]:
    root = Path(root)
    candidates = [
        path for path in sorted(root.rglob("*"), key=lambda p: str(p))
        if path.is_file() and path.suffix not in {".pyc"}
    ]
    return create_artifact_bundle(candidates, output_path, policy=policy, source_snapshot_hash=_stable_hash([str(path) for path in candidates]))
