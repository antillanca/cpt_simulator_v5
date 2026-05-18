#!/usr/bin/env python3
"""Compare two compact evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.governance.artifact_policy import load_artifact_policy
from backend.reporting.eval_diff import diff_eval_reports
from backend.reporting.report_builder import validate_evaluation_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two evaluation reports.")
    parser.add_argument("--baseline", required=True, help="Baseline report JSON path.")
    parser.add_argument("--candidate", required=True, help="Candidate report JSON path.")
    parser.add_argument("--output", default=None, help="Output file path.")
    parser.add_argument("--policy", default="configs/artifact_policy.yaml", help="Artifact policy path.")
    parser.add_argument("--strict-policy", action="store_true", help="Fail on policy mismatches.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Write JSON output.")
    parser.add_argument("--markdown", action="store_true", help="Write Markdown output.")
    args = parser.parse_args()

    policy = load_artifact_policy(Path(args.policy))
    baseline_payload = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate_payload = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    validate_evaluation_report(baseline_payload, policy=policy, strict_policy=args.strict_policy)
    validate_evaluation_report(candidate_payload, policy=policy, strict_policy=args.strict_policy)
    diff = diff_eval_reports(Path(args.baseline), Path(args.candidate))
    output = Path(args.output) if args.output else Path(args.candidate).with_suffix(".diff.json")
    if args.json and args.markdown:
        json_path = output if output.suffix else output.with_suffix(".json")
        md_path = output.with_suffix(".md")
        json_path.write_text(diff.to_json(), encoding="utf-8")
        md_path.write_text(diff.to_markdown(), encoding="utf-8")
        written = [json_path, md_path]
    elif args.markdown:
        output.write_text(diff.to_markdown(), encoding="utf-8")
        written = [output]
    else:
        output.write_text(diff.to_json(), encoding="utf-8")
        written = [output]

    print(json.dumps({"written": [str(path) for path in written], "same_fingerprint": diff.same_fingerprint, "policy_schema_version": policy.schema_version}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
