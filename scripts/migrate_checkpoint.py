#!/usr/bin/env python3
"""Migrate governed checkpoints to a target schema version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.governance.artifact_policy import load_artifact_policy
from backend.neural.checkpoints import migrate_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate a checkpoint artifact.")
    parser.add_argument("--path", "--checkpoint", dest="checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--target-version", default="2.7.6")
    parser.add_argument("--output", default=None, help="Output path.")
    parser.add_argument("--policy", default="configs/artifact_policy.yaml", help="Artifact policy path.")
    parser.add_argument("--strict-policy", action="store_true", help="Fail on policy mismatches.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Write JSON output.")
    parser.add_argument("--markdown", action="store_true", help="Write Markdown output.")
    args = parser.parse_args()

    policy = load_artifact_policy(Path(args.policy))
    result = migrate_checkpoint(Path(args.checkpoint), args.target_version, dry_run=args.dry_run, policy=policy, strict_policy=args.strict_policy)
    output = Path(args.output) if args.output else Path(args.checkpoint).with_name(f"{Path(args.checkpoint).stem}.migrated_{args.target_version}{Path(args.checkpoint).suffix}")
    written: list[Path]
    if args.json and args.markdown:
        json_path = output if output.suffix else output.with_suffix(".json")
        md_path = output.with_suffix(".md")
        json_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        md_path.write_text("\n".join([
            "# Checkpoint Migration",
            "",
            f"- Source Version: {result.source_version}",
            f"- Target Version: {result.target_version}",
            f"- Dry Run: {result.dry_run}",
            f"- Migration Fingerprint: {result.migration_fingerprint}",
            f"- Policy Schema: {policy.schema_version}",
            f"- Seed: {args.seed}",
        ]), encoding="utf-8")
        written = [json_path, md_path]
    elif args.markdown and not args.json:
        lines = [
            "# Checkpoint Migration",
            "",
            f"- Source Version: {result.source_version}",
            f"- Target Version: {result.target_version}",
            f"- Dry Run: {result.dry_run}",
            f"- Migration Fingerprint: {result.migration_fingerprint}",
            f"- Policy Schema: {policy.schema_version}",
            f"- Seed: {args.seed}",
        ]
        output.write_text("\n".join(lines), encoding="utf-8")
        written = [output]
    else:
        output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        written = [output]
    print(json.dumps({"written": [str(path) for path in written], "migration_fingerprint": result.migration_fingerprint}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
