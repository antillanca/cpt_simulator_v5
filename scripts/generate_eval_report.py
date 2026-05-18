#!/usr/bin/env python3
"""Generate compact deterministic evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.governance.artifact_policy import load_artifact_policy
from backend.reporting.report_builder import build_evaluation_report


def _write_output(report, output: Path, *, markdown: bool, json_mode: bool, dual: bool) -> list[Path]:
    paths: list[Path] = []
    if dual:
        json_path = output if output.suffix else output.with_suffix(".json")
        md_path = output.with_suffix(".md")
        report.save(json_path, markdown=False)
        report.save(md_path, markdown=True)
        return [json_path, md_path]
    report.save(output, markdown=markdown and not json_mode)
    return [output]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate compact evaluation reports.")
    parser.add_argument("--input", required=True, help="Path to evaluation run JSON.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    parser.add_argument("--dataset-manifest", default=None, help="Optional dataset manifest path.")
    parser.add_argument("--output", required=True, help="Output file or base path.")
    parser.add_argument("--policy", default="configs/artifact_policy.yaml", help="Artifact policy path.")
    parser.add_argument("--strict-policy", action="store_true", help="Fail on policy mismatches.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Write JSON output.")
    parser.add_argument("--markdown", action="store_true", help="Write Markdown output.")
    args = parser.parse_args()

    policy = load_artifact_policy(Path(args.policy))
    evaluation_run = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report = build_evaluation_report(
        evaluation_run,
        checkpoint_path=args.checkpoint,
        dataset_manifest_path=args.dataset_manifest,
        policy=policy,
        strict_policy=args.strict_policy,
        seed=args.seed,
    )

    output = Path(args.output)
    dual = args.json and args.markdown
    if not args.json and not args.markdown:
        report.save(output, markdown=False)
        paths = [output]
    else:
        paths = _write_output(report, output, markdown=args.markdown, json_mode=args.json, dual=dual)

    print(json.dumps({"written": [str(path) for path in paths], "report_fingerprint": report.to_dict()["report_fingerprint"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
