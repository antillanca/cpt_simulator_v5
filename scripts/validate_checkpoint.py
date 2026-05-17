#!/usr/bin/env python3
"""Validate governed checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.governance.artifact_policy import load_artifact_policy
from backend.neural.checkpoints import CHECKPOINT_SCHEMA_VERSION, infer_checkpoint_version, validate_checkpoint_payload
from backend.neural.tiny_experiments import load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a checkpoint artifact.")
    parser.add_argument("--path", "--checkpoint", dest="checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--output", default=None, help="Output path.")
    parser.add_argument("--policy", default="configs/artifact_policy.yaml", help="Artifact policy path.")
    parser.add_argument("--strict-policy", action="store_true", help="Fail on policy mismatches.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Write JSON output.")
    parser.add_argument("--markdown", action="store_true", help="Write Markdown output.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    policy = load_artifact_policy(Path(args.policy))
    model, tokenizer, payload = load_checkpoint(checkpoint_path)
    errors = validate_checkpoint_payload(payload, allow_legacy=True, policy=policy, strict_policy=args.strict_policy)
    validation = {
        "checkpoint": str(checkpoint_path),
        "schema_version": payload.get("schema_version", ""),
        "model_type": payload.get("model_type", ""),
        "model_config": payload.get("model_config", {}),
        "training_config": payload.get("training_config", payload.get("config", {})),
        "dataset_manifest_hash": payload.get("dataset_manifest_hash", ""),
        "snapshot_hash": payload.get("snapshot_hash", ""),
        "weights_hash": payload.get("weights_hash", ""),
        "optimizer_state_hash": payload.get("optimizer_state_hash"),
        "eval_fingerprint": payload.get("eval_fingerprint"),
        "curriculum_coverage": payload.get("curriculum_coverage", {}),
        "seed": payload.get("seed", 0),
        "created_at": payload.get("created_at", 0.0),
        "artifact_fingerprint": payload.get("artifact_fingerprint", ""),
        "validation_errors": errors,
        "policy_schema_version": policy.schema_version,
        "tokenizer_size": len(tokenizer.itos),
        "infer_version": infer_checkpoint_version(payload),
        "model_class": model.__class__.__name__,
        "seed_override": args.seed,
    }
    output = Path(args.output) if args.output else checkpoint_path.with_suffix(".validation.json")
    written: list[Path]
    if args.json and args.markdown:
        json_path = output if output.suffix else output.with_suffix(".json")
        md_path = output.with_suffix(".md")
        json_path.write_text(json.dumps(validation, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        md_path.write_text("\n".join([
            "# Checkpoint Validation",
            "",
            f"- Checkpoint: {validation['checkpoint']}",
            f"- Schema Version: {validation['schema_version']}",
            f"- Model Type: {validation['model_type']}",
            f"- Artifact Fingerprint: {validation['artifact_fingerprint']}",
            f"- Policy Schema: {validation['policy_schema_version']}",
            f"- Validation Errors: {len(validation['validation_errors'])}",
        ]), encoding="utf-8")
        written = [json_path, md_path]
    elif args.markdown and not args.json:
        lines = [
            "# Checkpoint Validation",
            "",
            f"- Checkpoint: {validation['checkpoint']}",
            f"- Schema Version: {validation['schema_version']}",
            f"- Model Type: {validation['model_type']}",
            f"- Artifact Fingerprint: {validation['artifact_fingerprint']}",
            f"- Policy Schema: {validation['policy_schema_version']}",
            f"- Validation Errors: {len(validation['validation_errors'])}",
        ]
        output.write_text("\n".join(lines), encoding="utf-8")
        written = [output]
    else:
        output.write_text(json.dumps(validation, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        written = [output]
    print(json.dumps({"written": [str(path) for path in written], "schema_version": validation["schema_version"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
