#!/usr/bin/env python3
"""Query a deterministic artifact inventory index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.governance.artifact_inventory import build_inventory_index, load_inventory_index
from backend.governance.artifact_policy import load_artifact_policy
from backend.governance.query_engine import query_inventory


def _table(rows: list[dict[str, object]]) -> str:
    lines = [
        "# CPT Inventory Query",
        "",
        "| Artifact | Type | Schema | Fingerprint | Path |",
        "|---------|------|--------|-------------|------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['artifact_id']} | {row['artifact_type']} | {row['schema_version']} | {row['fingerprint']} | {row['relative_path']} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Query an artifact inventory index.")
    parser.add_argument("--workspace", default=None, help="Workspace root to scan.")
    parser.add_argument("--index", default=None, help="Inventory index path.")
    parser.add_argument("--policy", default="configs/artifact_policy.yaml", help="Artifact policy path.")
    parser.add_argument("--artifact-type", default=None)
    parser.add_argument("--schema-version", default=None)
    parser.add_argument("--fingerprint", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--retention-status", default=None)
    parser.add_argument("--parent", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    policy = load_artifact_policy(Path(args.policy)) if Path(args.policy).exists() else None
    if args.index:
        index = load_inventory_index(Path(args.index))
    elif args.workspace:
        index = build_inventory_index(Path(args.workspace), policy=policy)
    else:
        raise SystemExit("Provide --workspace or --index.")
    entries = query_inventory(
        index,
        artifact_type=args.artifact_type,
        schema_version=args.schema_version,
        fingerprint=args.fingerprint,
        tag=args.tag,
        parent_id=args.parent,
        retention_status=args.retention_status,
    )
    payload = {
        "index_fingerprint": index.inventory_fingerprint,
        "entry_count": len(entries),
        "entries": [entry.to_dict() for entry in entries],
    }
    output = Path(args.output) if args.output else Path("inventory_query.json")
    if args.json and args.markdown:
        json_path = output if output.suffix else output.with_suffix(".json")
        md_path = output.with_suffix(".md")
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(_table(payload["entries"]), encoding="utf-8")
    elif args.markdown and not args.json:
        output.write_text(_table(payload["entries"]), encoding="utf-8")
    else:
        output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"entry_count": len(entries), "index_fingerprint": index.inventory_fingerprint}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
