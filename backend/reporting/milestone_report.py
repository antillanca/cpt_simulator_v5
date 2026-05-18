# backend/reporting/milestone_report.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backend.governance.milestones import load_milestones, MilestoneRecord


def generate_markdown(milestones: tuple[MilestoneRecord, ...]) -> str:
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


def generate_json(milestones: tuple[MilestoneRecord, ...]) -> str:
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


def generate_milestone_report(output_dir: Path, fmt: str = "md") -> Path:
    milestones_path = Path("docs/milestones.yaml")
    milestones = load_milestones(milestones_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        content = generate_json(milestones)
        ext = "json"
    else:
        content = generate_markdown(milestones)
        ext = "md"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"milestone_report_{timestamp}.{ext}"
    out_path.write_text(content, encoding="utf-8")
    return out_path
