#!/usr/bin/env python3
"""Export an artifact lineage graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.governance.artifact_inventory import build_inventory_index, load_inventory_index
from backend.governance.artifact_policy import load_artifact_policy
from backend.governance.lineage_graph import build_lineage_graph, save_lineage_graph


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a lineage graph.")
    parser.add_argument("--workspace", default=None, help="Workspace root to scan.")
    parser.add_argument("--index", default=None, help="Inventory index path.")
    parser.add_argument("--policy", default="configs/artifact_policy.yaml", help="Artifact policy path.")
    parser.add_argument("--output", required=True, help="Output path.")
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
    graph = build_lineage_graph(index)
    output = Path(args.output)
    if args.json and args.markdown:
        json_path = output if output.suffix else output.with_suffix(".json")
        md_path = output.with_suffix(".md")
        save_lineage_graph(graph, json_path)
        md_path.write_text(graph.to_markdown(), encoding="utf-8")
    elif args.markdown and not args.json:
        output.write_text(graph.to_markdown(), encoding="utf-8")
    else:
        save_lineage_graph(graph, output)
    print(json.dumps({"written": str(output), "graph_fingerprint": graph.graph_fingerprint}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
