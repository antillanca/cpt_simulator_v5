# backend/governance/milestones.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import yaml

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


def compute_milestone_fingerprint(milestones: tuple[MilestoneRecord, ...]) -> str:
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


def load_milestones(path: Path) -> tuple[MilestoneRecord, ...]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = []
    for row in data["milestones"]:
        items.append(
            MilestoneRecord(
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
            )
        )
    items = tuple(sorted(items, key=lambda x: x.version))
    fp = compute_milestone_fingerprint(items)
    return tuple(
        MilestoneRecord(
            **{**m.__dict__, "fingerprint": fp}
        )
        for m in items
    )
