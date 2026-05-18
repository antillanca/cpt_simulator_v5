"""Dataset sharding for CPT v2.7 distillation readiness.

Supports deterministic sharding of large JSONL exports into
fixed-size shards with a shard manifest for reconstruction.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def shard_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    shard_size: int = 1000,
    prefix: str = "shard",
) -> dict[str, Any]:
    """Split a JSONL dataset into fixed-size shards.

    Returns a shard manifest dict with:
    - source: original file path
    - shard_size: records per shard
    - shards: list of {name, path, records, hash}
    - total_records: total lines
    - fingerprint: manifest fingerprint
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards: list[dict[str, Any]] = []
    current_records: list[str] = []
    shard_index = 0
    total_records = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            current_records.append(line)
            total_records += 1

            if len(current_records) >= shard_size:
                shard_name = f"{prefix}_{shard_index:04d}.jsonl"
                shard_path = output_dir / shard_name
                shard_path.write_text("\n".join(current_records) + "\n", encoding="utf-8")
                shard_hash = _stable_hash(current_records)
                shards.append({
                    "name": shard_name,
                    "path": str(shard_path),
                    "records": len(current_records),
                    "hash": shard_hash,
                })
                current_records = []
                shard_index += 1

    # Flush remaining records
    if current_records:
        shard_name = f"{prefix}_{shard_index:04d}.jsonl"
        shard_path = output_dir / shard_name
        shard_path.write_text("\n".join(current_records) + "\n", encoding="utf-8")
        shard_hash = _stable_hash(current_records)
        shards.append({
            "name": shard_name,
            "path": str(shard_path),
            "records": len(current_records),
            "hash": shard_hash,
        })

    manifest = {
        "source": str(input_path),
        "shard_size": shard_size,
        "shards": shards,
        "total_records": total_records,
    }
    manifest["fingerprint"] = _stable_hash(manifest)
    return manifest


def save_shard_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    """Save shard manifest to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return path


def load_shard_manifest(path: str | Path) -> dict[str, Any]:
    """Load shard manifest from JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_shard_manifest(manifest: dict[str, Any]) -> list[str]:
    """Validate a shard manifest. Returns list of errors."""
    errors = []
    required = ["source", "shard_size", "shards", "total_records", "fingerprint"]
    for key in required:
        if key not in manifest:
            errors.append(f"Missing field: {key}")
    if "fingerprint" in manifest:
        check = {k: v for k, v in manifest.items() if k != "fingerprint"}
        recomputed = _stable_hash(check)
        if manifest["fingerprint"] != recomputed:
            errors.append(f"Fingerprint mismatch: stored={manifest['fingerprint'][:16]} computed={recomputed[:16]}")
    return errors


def iter_shard_records(shard_path: str | Path) -> list[dict[str, Any]]:
    """Load all records from a single shard file."""
    records = []
    path = Path(shard_path)
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_dataset_from_shards(shard_dir: str | Path, manifest: dict[str, Any]):
    """Iterate all records across shards in manifest order."""
    shard_dir = Path(shard_dir)
    for shard_info in manifest.get("shards", []):
        shard_path = shard_dir / shard_info["name"]
        for record in iter_shard_records(shard_path):
            yield record


def reassemble_dataset(shard_dir: str | Path, manifest: dict[str, Any], output_path: str | Path) -> Path:
    """Reassemble a full JSONL dataset from its shards."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for record in iter_dataset_from_shards(shard_dir, manifest):
            out.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    return output_path
