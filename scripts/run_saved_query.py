#!/usr/bin/env python3
"""Run a saved deterministic inventory query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.governance.artifact_inventory import build_inventory_index, load_inventory_index
from backend.governance.artifact_policy import load_artifact_policy
from backend.governance.saved_queries import execute_saved_query, load_query, save_query


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a saved query.")
    parser.add_argument("--query", required=True, help="Saved query JSON path.")
    parser.add_argument("--workspace", default=None, help="Workspace root to scan.")
    parser.add_argument("--inventory", default=None, help="Inventory index path.")
    parser.add_argument("--policy", default="configs/artifact_policy.yaml", help="Artifact policy path.")
    parser.add_argument("--output", default=None, help="Output path.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    policy = load_artifact_policy(Path(args.policy)) if Path(args.policy).exists() else None
    query = load_query(Path(args.query))
    if args.inventory:
        inventory = load_inventory_index(Path(args.inventory))
    elif args.workspace:
        inventory = build_inventory_index(Path(args.workspace), policy=policy)
    else:
        raise SystemExit("Provide --workspace or --inventory.")
    result = execute_saved_query(query, inventory)
    payload = {
        "query_name": query["query_name"],
        "query_fingerprint": query["query_fingerprint"],
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "entry_count": len(result),
        "entries": [entry.to_dict() for entry in result],
    }
    output = Path(args.output) if args.output else Path("saved_query_result.json")
    if args.json and args.markdown:
        json_path = output if output.suffix else output.with_suffix(".json")
        md_path = output.with_suffix(".md")
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(
            "\n".join(
                [
                    "# CPT Saved Query Result",
                    "",
                    f"- Query: {query['query_name']}",
                    f"- Matches: {len(result)}",
                    f"- Inventory: {inventory.inventory_fingerprint}",
                ]
            ),
            encoding="utf-8",
        )
    elif args.markdown and not args.json:
        output.write_text(
            "\n".join(
                [
                    "# CPT Saved Query Result",
                    "",
                    f"- Query: {query['query_name']}",
                    f"- Matches: {len(result)}",
                    f"- Inventory: {inventory.inventory_fingerprint}",
                ]
            ),
            encoding="utf-8",
        )
    else:
        output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"query_name": query["query_name"], "entry_count": len(result), "inventory_fingerprint": inventory.inventory_fingerprint}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
