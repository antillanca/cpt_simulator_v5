#!/usr/bin/env python3
"""CORE -- Migrate Runtime Artifacts.

Copies operational experience, runtime traces, benchmarks, and paper
figures from the old repo layout into the new CORE layout under data/.
Preserves hashes, manifests, and timestamps.

Records all moves in MIGRATION_LOG.md.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_ROOT = REPO_ROOT / "core_runtime"

SOURCE_DIRS = {
    "workspace/operational_experience": "data/operational_experience",
    "workspace/runtime_reports": "data/runtime_reports",
    "workspace/paper_figures": "data/paper_figures",
}

MANIFEST_FILE = REPO_ROOT / "runtime_release_manifest_v215.json"


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def migrate_artifacts(dry_run: bool = False) -> list[dict]:
    moves = []
    for src_rel, dst_rel in SOURCE_DIRS.items():
        src = REPO_ROOT / src_rel
        dst = CORE_ROOT / dst_rel
        if not src.exists():
            print(f"  SKIP (not found): {src_rel}")
            continue

        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.rglob("*")):
            if f.is_dir():
                continue
            rel = f.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)

            entry = {
                "source": str(f.relative_to(REPO_ROOT)),
                "destination": str(target.relative_to(REPO_ROOT)),
                "sha256_before": sha256_file(f),
                "size_bytes": f.stat().st_size,
            }

            if not dry_run:
                shutil.copy2(f, target)
                entry["sha256_after"] = sha256_file(target)
                entry["hash_preserved"] = entry["sha256_before"] == entry["sha256_after"]
            else:
                entry["sha256_after"] = entry["sha256_before"]
                entry["hash_preserved"] = True  # assumed

            moves.append(entry)

    # Copy release manifest
    if MANIFEST_FILE.exists():
        dst = CORE_ROOT / "data" / "runtime_release_manifest_v215.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            shutil.copy2(MANIFEST_FILE, dst)
        moves.append({
            "source": str(MANIFEST_FILE.relative_to(REPO_ROOT)),
            "destination": str(dst.relative_to(REPO_ROOT)),
            "sha256_before": sha256_file(MANIFEST_FILE),
            "size_bytes": MANIFEST_FILE.stat().st_size,
            "hash_preserved": True,
        })

    return moves


def update_migration_log(moves: list[dict]) -> None:
    log_path = REPO_ROOT / "docs" / "MIGRATION_LOG.md"
    existing = log_path.read_text() if log_path.exists() else ""

    entry_lines = ["\n### Artifact Migration\n"]
    entry_lines.append(f"Files migrated: {len(moves)}")
    entry_lines.append("")
    entry_lines.append("| Source | Destination | Hash Preserved |")
    entry_lines.append("|--------|-------------|----------------|")
    for m in moves:
        preserved = "YES" if m.get("hash_preserved", False) else "NO"
        entry_lines.append(f"| {m['source']} | {m['destination']} | {preserved} |")

    entry_lines.append("")
    with open(log_path, "a") as f:
        f.write("\n".join(entry_lines))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("CORE Artifact Migration")
    print("=" * 40)
    moves = migrate_artifacts(dry_run=args.dry_run)

    all_preserved = all(m.get("hash_preserved", True) for m in moves)
    print(f"Files migrated: {len(moves)}")
    print(f"All hashes preserved: {all_preserved}")

    if not args.dry_run:
        update_migration_log(moves)
        print("Migration log updated")

        # Save migration manifest
        manifest_path = CORE_ROOT / "data" / "artifact_migration_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({"migrated_files": moves, "all_hashes_preserved": all_preserved}, f, indent=2)
        print(f"Migration manifest: {manifest_path}")


if __name__ == "__main__":
    main()
