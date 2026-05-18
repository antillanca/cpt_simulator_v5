#!/usr/bin/env python3
"""Archive an explicit set of CPT artifacts into a deterministic bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.governance.artifact_policy import load_artifact_policy
from backend.governance.archive_tooling import create_artifact_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive artifacts into a bundle.")
    parser.add_argument("--path", action="append", dest="paths", required=True, help="Artifact path to include.")
    parser.add_argument("--output", required=True, help="Bundle output path.")
    parser.add_argument("--policy", default="configs/artifact_policy.yaml", help="Artifact policy path.")
    args = parser.parse_args()

    policy = load_artifact_policy(Path(args.policy))
    bundle_path, manifest = create_artifact_bundle([Path(path) for path in args.paths], Path(args.output), policy=policy)
    print(json.dumps({"bundle_path": str(bundle_path), "bundle_fingerprint": manifest["bundle_fingerprint"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
