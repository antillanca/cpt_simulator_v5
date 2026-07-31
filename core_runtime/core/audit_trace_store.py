"""CORE v3.3 — Audit Trace Store.

Persist trustworthiness audits separately from memory and exact cache.

SAFETY GUARANTEE:
  - Audit trace store has NO influence on runtime outputs
  - It is a write-only observability layer from the runtime's perspective
  - Same input -> same audit ordering
  - Atomic persistence — no data loss on crash
  - No influence on runtime outputs
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core_runtime.core.trustworthiness_runtime import TrustworthinessAudit
from core_runtime.core.explainability_runtime import ExecutionExplanation
from core_runtime.core.failure_report import FailureReport


# ---------------------------------------------------------------------------
# AuditTraceRecord — composite of audit + explanation + optional report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditTraceRecord:
    """Immutable audit trail record combining audit, explanation, and
    optional failure report."""
    audit: TrustworthinessAudit
    explanation: ExecutionExplanation
    report: FailureReport | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "audit": {
                "task_hash": self.audit.task_hash,
                "trust_level": self.audit.trust_level.value,
                "flags": [f.value for f in self.audit.flags],
                "confidence_score": self.audit.confidence_score,
                "uncertainty_score": self.audit.uncertainty_score,
                "projection_iterations": self.audit.projection_iterations,
                "final_residual": self.audit.final_residual,
                "escalation_required": self.audit.escalation_required,
                "deterministic": self.audit.deterministic,
                "fingerprint": self.audit.fingerprint(),
            },
            "explanation": self.explanation.to_dict(),
        }
        if self.report is not None:
            d["report"] = self.report.to_dict()
        else:
            d["report"] = None
        return d


# ---------------------------------------------------------------------------
# AuditTraceStore — deterministic, atomic, ordered
# ---------------------------------------------------------------------------

class AuditTraceStore:
    """Persist audits separately from memory and exact cache.

    Invariants:
    - Deterministic ordering (insertion order + task_hash tiebreak)
    - Atomic file writes (write-to-temp, then rename)
    - No influence on runtime outputs
    - Queryable by task_hash
    """

    def __init__(self) -> None:
        self._records: dict[str, AuditTraceRecord] = {}
        self._order: list[str] = []

    def append(
        self,
        audit: TrustworthinessAudit,
        explanation: ExecutionExplanation,
        report: FailureReport | None = None,
    ) -> None:
        """Add an audit trace record. Overwrites if same task_hash."""
        record = AuditTraceRecord(
            audit=audit,
            explanation=explanation,
            report=report,
        )
        if audit.task_hash in self._records:
            self._records[audit.task_hash] = record
        else:
            self._records[audit.task_hash] = record
            self._order.append(audit.task_hash)

    def get(self, task_hash: str) -> AuditTraceRecord | None:
        """Retrieve audit trace by task hash."""
        return self._records.get(task_hash)

    def contains(self, task_hash: str) -> bool:
        """Check if task hash has an audit trace."""
        return task_hash in self._records

    def all_records(self) -> list[AuditTraceRecord]:
        """Return all records in deterministic insertion order."""
        return [self._records[h] for h in self._order]

    def __len__(self) -> int:
        return len(self._records)

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def trust_level_distribution(self) -> dict[str, int]:
        """Count records by trust level."""
        dist: dict[str, int] = {}
        for rec in self.all_records():
            level = rec.audit.trust_level.value
            dist[level] = dist.get(level, 0) + 1
        return dist

    def flag_distribution(self) -> dict[str, int]:
        """Count occurrences of each trust flag."""
        dist: dict[str, int] = {}
        for rec in self.all_records():
            for flag in rec.audit.flags:
                dist[flag.value] = dist.get(flag.value, 0) + 1
        return dist

    def escalation_rate(self) -> float:
        """Fraction of records with escalation_required=True."""
        if not self._records:
            return 0.0
        escalated = sum(
            1 for r in self.all_records()
            if r.audit.escalation_required
        )
        return escalated / len(self._records)

    def avg_confidence(self) -> float:
        """Average confidence score across all records."""
        if not self._records:
            return 0.0
        total = sum(r.audit.confidence_score for r in self.all_records())
        return total / len(self._records)

    def avg_residual(self) -> float:
        """Average final residual across all records."""
        if not self._records:
            return 0.0
        total = sum(r.audit.final_residual for r in self.all_records())
        return total / len(self._records)

    # -----------------------------------------------------------------------
    # Persistence — atomic JSONL
    # -----------------------------------------------------------------------

    def export_jsonl(self, path: str | Path) -> None:
        """Write all records to JSONL atomically."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(rec.to_dict(), sort_keys=True)
            for rec in self.all_records()
        ]
        content = "\n".join(lines) + ("\n" if lines else "")
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)

    def load_jsonl(self, path: str | Path) -> None:
        """Load records from JSONL file. Appends to existing."""
        path = Path(path)
        if not path.exists():
            return
        from core_runtime.core.trustworthiness_runtime import (
            TrustLevel,
            TrustFlag,
            TrustworthinessAudit,
        )
        from core_runtime.core.explainability_runtime import ExecutionExplanation
        from core_runtime.core.failure_report import FailureReport

        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            audit_d = d["audit"]
            audit = TrustworthinessAudit(
                task_hash=audit_d["task_hash"],
                trust_level=TrustLevel(audit_d["trust_level"]),
                flags=tuple(TrustFlag(f) for f in audit_d["flags"]),
                confidence_score=audit_d["confidence_score"],
                uncertainty_score=audit_d["uncertainty_score"],
                projection_iterations=audit_d["projection_iterations"],
                final_residual=audit_d["final_residual"],
                escalation_required=audit_d["escalation_required"],
                deterministic=audit_d.get("deterministic", True),
                metadata=None,
            )
            expl_d = d["explanation"]
            explanation = ExecutionExplanation(
                task_hash=expl_d["task_hash"],
                summary=expl_d["summary"],
                key_factors=tuple(expl_d["key_factors"]),
                topology_family=expl_d["topology_family"],
                confidence_score=expl_d["confidence_score"],
                trust_level=expl_d["trust_level"],
                projected_iterations=expl_d["projected_iterations"],
                residual=expl_d["residual"],
                explanation_type=expl_d["explanation_type"],
                metadata=expl_d.get("metadata", {}),
            )
            report = None
            if d.get("report") is not None:
                rp = d["report"]
                report = FailureReport(
                    task_hash=rp["task_hash"],
                    error_type=rp["error_type"],
                    conditions=rp["conditions"],
                    probable_cause=rp["probable_cause"],
                    recommended_action=rp["recommended_action"],
                    topology_signature=rp["topology_signature"],
                    confidence_score=rp["confidence_score"],
                    residual=rp["residual"],
                    projection_iterations=rp["projection_iterations"],
                    deterministic=rp.get("deterministic", True),
                )
            self.append(audit, explanation, report)
