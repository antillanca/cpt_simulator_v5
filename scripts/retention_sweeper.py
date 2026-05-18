#!/usr/bin/env python3
"""Deterministic artifact retention sweeper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.governance.artifact_policy import artifact_policy_fingerprint, load_artifact_policy
from backend.governance.retention_sweeper import build_retention_plan, execute_retention_plan, scan_retention_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic artifact retention sweep.")
    parser.add_argument("--root", default=".", help="Root directory to scan.")
    parser.add_argument("--policy", default="configs/artifact_policy.yaml", help="Artifact policy path.")
    parser.add_argument("--output", default=None, help="Output path.")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; no deletion.")
    parser.add_argument("--execute", action="store_true", help="Allow deletion according to plan.")
    parser.add_argument("--yes", action="store_true", help="Confirm destructive execution.")
    parser.add_argument("--json", action="store_true", help="Write JSON output.")
    parser.add_argument("--markdown", action="store_true", help="Write Markdown output.")
    args = parser.parse_args()

    policy = load_artifact_policy(Path(args.policy))
    root = Path(args.root)
    candidates = scan_retention_candidates(root, policy)
    plan = build_retention_plan(candidates, policy)

    if args.execute and not args.yes:
        raise SystemExit("Refusing to delete artifacts without --yes.")
    dry_run = True if args.dry_run or not args.execute else False
    result = execute_retention_plan(plan, dry_run=dry_run)
    payload = {
        "root": str(root),
        "policy_schema_version": policy.schema_version,
        "policy_fingerprint": artifact_policy_fingerprint(policy),
        "dry_run": dry_run,
        "plan": [candidate.to_dict() for candidate in plan],
        "result": result.to_dict(),
    }
    output = Path(args.output) if args.output else root / "retention_plan.json"
    written: list[Path]
    if args.json and args.markdown:
        json_path = output if output.suffix else output.with_suffix(".json")
        md_path = output.with_suffix(".md")
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(
            "\n".join(
                [
                    "# CPT Retention Sweep",
                    "",
                    f"- Root: {payload['root']}",
                    f"- Dry Run: {payload['dry_run']}",
                    f"- Scanned: {payload['result']['scanned']}",
                    f"- Retained: {payload['result']['retained']}",
                    f"- Flagged: {payload['result']['flagged']}",
                    f"- Deleted: {payload['result']['deleted']}",
                    f"- Reclaimed Bytes: {payload['result']['reclaimed_bytes']}",
                ]
            ),
            encoding="utf-8",
        )
        written = [json_path, md_path]
    elif args.markdown and not args.json:
        output.write_text(
            "\n".join(
                [
                    "# CPT Retention Sweep",
                    "",
                    f"- Root: {payload['root']}",
                    f"- Dry Run: {payload['dry_run']}",
                    f"- Scanned: {payload['result']['scanned']}",
                    f"- Retained: {payload['result']['retained']}",
                    f"- Flagged: {payload['result']['flagged']}",
                    f"- Deleted: {payload['result']['deleted']}",
                    f"- Reclaimed Bytes: {payload['result']['reclaimed_bytes']}",
                ]
            ),
            encoding="utf-8",
        )
        written = [output]
    else:
        output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        written = [output]
    print(json.dumps({"written": [str(path) for path in written], "dry_run": dry_run, "deleted": result.deleted}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
