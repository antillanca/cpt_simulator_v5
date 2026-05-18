#!/usr/bin/env python3
"""Generate a deterministic artifact retention report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.governance.artifact_policy import load_artifact_policy
from backend.reporting.retention_report import build_retention_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a retention report.")
    parser.add_argument("--root", default=".", help="Root directory to scan.")
    parser.add_argument("--policy", default="configs/artifact_policy.yaml", help="Artifact policy path.")
    parser.add_argument("--output", default=None, help="Output path.")
    parser.add_argument("--json", action="store_true", help="Write JSON output.")
    parser.add_argument("--markdown", action="store_true", help="Write Markdown output.")
    args = parser.parse_args()

    policy = load_artifact_policy(Path(args.policy))
    report = build_retention_report(Path(args.root), policy)
    output = Path(args.output) if args.output else Path(args.root) / "retention_report.json"
    written: list[Path]
    if args.json and args.markdown:
        json_path = output if output.suffix else output.with_suffix(".json")
        md_path = output.with_suffix(".md")
        json_path.write_text(report.to_json(), encoding="utf-8")
        md_path.write_text(report.to_markdown(), encoding="utf-8")
        written = [json_path, md_path]
    elif args.markdown and not args.json:
        output.write_text(report.to_markdown(), encoding="utf-8")
        written = [output]
    else:
        output.write_text(report.to_json(), encoding="utf-8")
        written = [output]
    print(json.dumps({"written": [str(path) for path in written], "report_fingerprint": report.to_dict()["report_fingerprint"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
