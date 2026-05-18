#!/usr/bin/env python3
"""Generate a deterministic workspace summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.governance.artifact_inventory import build_inventory_index, load_inventory_index
from backend.governance.artifact_policy import load_artifact_policy
from backend.reporting.workspace_summary import build_workspace_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a workspace summary.")
    parser.add_argument("--workspace", default=None, help="Workspace root to scan.")
    parser.add_argument("--index", default=None, help="Inventory index path.")
    parser.add_argument("--policy", default="configs/artifact_policy.yaml", help="Artifact policy path.")
    parser.add_argument("--output", default=None, help="Output path.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    policy = load_artifact_policy(Path(args.policy)) if Path(args.policy).exists() else None
    if args.index:
        index = load_inventory_index(Path(args.index))
    elif args.workspace:
        index = build_inventory_index(Path(args.workspace), policy=policy)
    else:
        raise SystemExit("Provide --workspace or --index.")
    summary = build_workspace_summary(Path(index.workspace_root), policy=policy, index=index)
    output = Path(args.output) if args.output else Path(index.workspace_root) / "workspace_summary.json"
    if args.json and args.markdown:
        json_path = output if output.suffix else output.with_suffix(".json")
        md_path = output.with_suffix(".md")
        json_path.write_text(summary.to_json(), encoding="utf-8")
        md_path.write_text(summary.to_markdown(), encoding="utf-8")
    elif args.markdown and not args.json:
        output.write_text(summary.to_markdown(), encoding="utf-8")
    else:
        output.write_text(summary.to_json(), encoding="utf-8")
    print(json.dumps({"written": str(output), "summary_fingerprint": summary.to_dict()["summary_fingerprint"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
