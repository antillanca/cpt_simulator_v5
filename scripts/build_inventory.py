#!/usr/bin/env python3
"""Build a deterministic artifact inventory index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.governance.artifact_inventory import build_inventory_index, save_inventory_index
from backend.governance.artifact_policy import load_artifact_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an artifact inventory index.")
    parser.add_argument("--workspace", required=True, help="Workspace root to scan.")
    parser.add_argument("--index", default=None, help="Index output path.")
    parser.add_argument("--policy", default="configs/artifact_policy.yaml", help="Artifact policy path.")
    args = parser.parse_args()

    policy = load_artifact_policy(Path(args.policy)) if Path(args.policy).exists() else None
    index = build_inventory_index(Path(args.workspace), policy=policy)
    output = Path(args.index) if args.index else Path(args.workspace) / "inventory_index.json"
    save_inventory_index(index, output)
    print(json.dumps({"written": str(output), "inventory_fingerprint": index.inventory_fingerprint}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
