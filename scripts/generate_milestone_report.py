#!/usr/bin/env python3
"""
Generate milestone report in Markdown and JSON from docs/milestones.yaml.
Usage: python scripts/generate_milestone_report.py [--json] [--output-dir workspace/reports]
"""
import argparse
import json
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import yaml

# Reuse the MilestoneRecord and functions from backend/governance/milestones.py
# For simplicity, we inline a minimal version here (adjust imports as needed)
try:
    from backend.governance.milestones import load_milestones, compute_milestone_fingerprint
except ImportError:
    # Fallback: implement here if module not ready
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class MilestoneRecord:
        version: str
        title: str
        status: str
        summary: str
        technical_impact: str
        date: str | None
        tags: tuple[str, ...]
        dependencies: tuple[str, ...]
        artifacts: tuple[str, ...]
        doc_refs: tuple[str, ...]
        fingerprint: str

    def load_milestones(path: Path):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        items = []
        for row in data["milestones"]:
            items.append(MilestoneRecord(
                version=str(row["version"]),
                title=str(row["title"]),
                status=str(row["status"]),
                summary=str(row["summary"]),
                technical_impact=str(row["technical_impact"]),
                date=row.get("date"),
                tags=tuple(row.get("tags", ())),
                dependencies=tuple(row.get("dependencies", ())),
                artifacts=tuple(row.get("artifacts", ())),
                doc_refs=tuple(row.get("doc_refs", ())),
                fingerprint="",
            ))
        items = tuple(sorted(items, key=lambda x: x.version))
        fp = compute_milestone_fingerprint(items)
        return tuple(MilestoneRecord(**{**m.__dict__, "fingerprint": fp}) for m in items)

    def compute_milestone_fingerprint(milestones):
        payload = [
            {
                "version": m.version,
                "title": m.title,
                "status": m.status,
                "summary": m.summary,
                "technical_impact": m.technical_impact,
                "date": m.date,
                "tags": list(m.tags),
                "dependencies": list(m.dependencies),
                "artifacts": list(m.artifacts),
                "doc_refs": list(m.doc_refs),
            }
            for m in milestones
        ]
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def generate_markdown(milestones):
    md = "# CPT Milestone Chronicle\n\n"
    md += f"Generated at: {datetime.now(timezone.utc).isoformat()}\n"
    md += f"Registry fingerprint: `{milestones[0].fingerprint}`\n\n"
    md += "## Completed Milestones\n\n"
    md += "| Version | Title | Status | Technical Impact |\n"
    md += "|--------|-------|--------|------------------|\n"
    for m in milestones:
        if m.status == "complete":
            md += f"| {m.version} | {m.title} | {m.status} | {m.technical_impact} |\n"
    md += "\n## Planned Milestones\n\n"
    md += "| Version | Title | Status | Technical Impact |\n"
    md += "|--------|-------|--------|------------------|\n"
    for m in milestones:
        if m.status != "complete":
            md += f"| {m.version} | {m.title} | {m.status} | {m.technical_impact} |\n"
    md += "\n## Notes for Future Agents\n"
    md += "- Do not rewrite core truth.\n"
    md += "- Preserve backward compatibility.\n"
    md += "- Prefer deterministic artifacts and explicit migrations.\n"
    return md


def generate_json(milestones):
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": milestones[0].fingerprint,
        "milestones": [
            {
                "version": m.version,
                "title": m.title,
                "status": m.status,
                "summary": m.summary,
                "technical_impact": m.technical_impact,
                "date": m.date,
                "tags": list(m.tags),
                "dependencies": list(m.dependencies),
                "artifacts": list(m.artifacts),
                "doc_refs": list(m.doc_refs),
            }
            for m in milestones
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Output JSON instead of Markdown")
    parser.add_argument("--output-dir", default="workspace/reports", help="Directory for output")
    args = parser.parse_args()

    milestones_path = Path("docs/milestones.yaml")
    if not milestones_path.exists():
        sys.exit("docs/milestones.yaml not found. Run this script from repo root.")

    milestones = load_milestones(milestones_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.json:
        content = generate_json(milestones)
        ext = "json"
    else:
        content = generate_markdown(milestones)
        ext = "md"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"milestone_report_{timestamp}.{ext}"
    out_path.write_text(content, encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
