#!/usr/bin/env python3
"""Analyze the impact of an artifact change or removal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.governance.artifact_inventory import build_inventory_index, load_inventory_index
from backend.governance.artifact_policy import load_artifact_policy
from backend.governance.impact_analysis import analyze_artifact_impact
from backend.governance.lineage_graph import build_lineage_graph


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze artifact impact.")
    parser.add_argument("--workspace", default=None, help="Workspace root to scan.")
    parser.add_argument("--inventory", default=None, help="Inventory index path.")
    parser.add_argument("--artifact-id", required=True, help="Artifact ID to analyze.")
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
    graph = build_lineage_graph(index)
    result = analyze_artifact_impact(args.artifact_id, index, graph)
    payload = result.to_dict()
    output = Path(args.output) if args.output else Path("impact_analysis.json")
    if args.json and args.markdown:
        json_path = output if output.suffix else output.with_suffix(".json")
        md_path = output.with_suffix(".md")
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(
            "\n".join(
                [
                    "# CPT Artifact Impact Analysis",
                    "",
                    f"- Artifact: {args.artifact_id}",
                    f"- Impacted Artifacts: {len(result.impacted_artifacts)}",
                    f"- Report Invalidations: {result.report_invalidations}",
                    f"- Checkpoint Invalidations: {result.checkpoint_invalidations}",
                ]
            ),
            encoding="utf-8",
        )
    elif args.markdown and not args.json:
        output.write_text(
            "\n".join(
                [
                    "# CPT Artifact Impact Analysis",
                    "",
                    f"- Artifact: {args.artifact_id}",
                    f"- Impacted Artifacts: {len(result.impacted_artifacts)}",
                    f"- Report Invalidations: {result.report_invalidations}",
                    f"- Checkpoint Invalidations: {result.checkpoint_invalidations}",
                ]
            ),
            encoding="utf-8",
        )
    else:
        output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"artifact_id": args.artifact_id, "impact_fingerprint": result.impact_fingerprint}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
