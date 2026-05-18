#!/usr/bin/env python3
"""Query the milestone registry. Example: python scripts/query_milestones.py --status complete --tag neural"""
import argparse
from pathlib import Path
from backend.governance.milestones import load_milestones
from backend.governance.milestone_queries import query_milestones


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version")
    parser.add_argument("--status")
    parser.add_argument("--tag")
    parser.add_argument("--dependency")
    parser.add_argument("--artifact")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    milestones = load_milestones(Path("docs/milestones.yaml"))
    results = query_milestones(
        milestones,
        version=args.version,
        status=args.status,
        tag=args.tag,
        dependency=args.dependency,
        artifact=args.artifact,
    )
    if args.json:
        import json
        print(json.dumps([m.__dict__ for m in results], indent=2, default=str))
    else:
        for m in results:
            print(f"{m.version}: {m.title} ({m.status})")


if __name__ == "__main__":
    main()
