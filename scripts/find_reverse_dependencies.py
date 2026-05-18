#!/usr/bin/env python3
"""Find downstream artifacts that depend on a given artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.governance.artifact_inventory import build_inventory_index, load_inventory_index
from backend.governance.artifact_policy import load_artifact_policy
from backend.governance.reverse_dependencies import build_reverse_dependency_index, find_reverse_dependencies


def main() -> int:
    parser = argparse.ArgumentParser(description="Find reverse dependencies for an artifact.")
    parser.add_argument("--workspace", default=None, help="Workspace root to scan.")
    parser.add_argument("--inventory", default=None, help="Inventory index path.")
    parser.add_argument("--artifact-id", required=True, help="Artifact ID to inspect.")
    parser.add_argument("--policy", default="configs/artifact_policy.yaml", help="Artifact policy path.")
    parser.add_argument("--output", default=None, help="Output path.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    policy = load_artifact_policy(Path(args.policy)) if Path(args.policy).exists() else None
    if args.inventory:
        index = load_inventory_index(Path(args.inventory))
    elif args.workspace:
        index = build_inventory_index(Path(args.workspace), policy=policy)
    else:
        raise SystemExit("Provide --workspace or --inventory.")
    reverse_index = build_reverse_dependency_index(index.entries)
    result = find_reverse_dependencies(args.artifact_id, reverse_index)
    payload = {
        "artifact_id": args.artifact_id,
        "index_fingerprint": index.inventory_fingerprint,
        "result": result.to_dict(),
    }
    output = Path(args.output) if args.output else Path("reverse_dependencies.json")
    if args.json and args.markdown:
        json_path = output if output.suffix else output.with_suffix(".json")
        md_path = output.with_suffix(".md")
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(
            "\n".join(
                [
                    "# CPT Reverse Dependencies",
                    "",
                    f"- Root Artifact: {args.artifact_id}",
                    f"- Dependent Count: {result.dependent_count}",
                    f"- Dependency Depth: {result.dependency_depth}",
                ]
            ),
            encoding="utf-8",
        )
    elif args.markdown and not args.json:
        output.write_text(
            "\n".join(
                [
                    "# CPT Reverse Dependencies",
                    "",
                    f"- Root Artifact: {args.artifact_id}",
                    f"- Dependent Count: {result.dependent_count}",
                    f"- Dependency Depth: {result.dependency_depth}",
                ]
            ),
            encoding="utf-8",
        )
    else:
        output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"artifact_id": args.artifact_id, "dependent_count": result.dependent_count, "index_fingerprint": index.inventory_fingerprint}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
