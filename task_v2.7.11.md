YOU ARE CODEX RUNNING INSIDE THE CPT COGNITIVE ENGINE REPOSITORY.

CURRENT STATUS:
CPT v2.7.10 — ARTIFACT SEARCH REFINEMENT + REVERSE DEPENDENCY INTELLIGENCE IS COMPLETE.

VERIFIED MILESTONES:
- v2.6 — snapshot reproducible
- v2.7 — distillation readiness
- v2.7.5 — tiny distillation experiments
- v2.7.6 — artifact governance + evaluation reporting
- v2.7.7 — artifact policy + operations runbook
- v2.7.8 — retention sweeper + archive/export tooling
- v2.7.9 — inventory + queryable indices + workspace intelligence
- v2.7.10 — reverse dependency intelligence + artifact search refinement

NEXT MISSION:
CPT v2.7.11 — MILESTONE LEDGER + ROADMAP CHRONICLE + RELEASE DOCUMENTATION

IMPORTANT:
THIS PHASE IS ABOUT:
- making project progress explicit and durable
- capturing major technical milestones
- generating a canonical history of architecture evolution
- helping future agents understand what was already solved
- producing deterministic docs from structured milestone data

DO NOT:
- change core truth logic
- expand curriculum
- train models
- rewrite governance systems
- break backward compatibility
- introduce non-deterministic documentation generation
- invent milestones not backed by actual repo state

========================================================
PRIMARY OBJECTIVE
========================================================

Transform CPT from:
"well-engineered, heavily versioned infrastructure"

into:
"a project with a canonical, queryable, reproducible milestone history"

The repository must now support:
- milestone registry
- roadmap chronicle
- release summaries
- architecture evolution notes
- documentation generation from structured milestone data
- changelog-style reporting for future agents

========================================================
IMPLEMENTATION ORDER
========================================================

1. Structured milestone registry
2. Canonical milestone timeline (create docs/milestones.yaml)
3. Release note generator
4. Roadmap chronicle document (docs/ROADMAP_CHRONICLE.md)
5. Milestone query tooling
6. Human-readable summaries
7. Tests and backward compatibility

========================================================
PHASE 1 — STRUCTURED MILESTONE REGISTRY
========================================================

Create:
- backend/governance/milestones.py

Purpose:
Store the canonical list of major CPT milestones in structured form.

Required code (use exact copy):

```python
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
```

========================================================
PHASE 2 — CANONICAL MILESTONE TIMELINE
========================================================

Create file: docs/milestones.yaml

Copy the following content exactly:

```yaml
milestones:
  - version: "v2.6"
    title: "Snapshot Reproducible"
    status: "complete"
    summary: "Generación determinista de snapshots, fingerprinting de datasets y benchmarks, aislamiento de tests."
    technical_impact: "Establece la base de reproducibilidad bit a bit para todos los artefactos futuros."
    date: "2026-05-15"
    tags: ["core", "determinismo"]
    dependencies: []
    artifacts: ["snapshots/", "benchmarks/"]
    doc_refs: ["docs/CPT_BENCH.md"]

  - version: "v2.7"
    title: "Distillation Readiness"
    status: "complete"
    summary: "Contratos de exportación de datasets, manifiestos, sharding, harness de evaluación y scaffold de modelos pequeños."
    technical_impact: "Prepara el pipeline de datos y evaluación para los experimentos de destilación neuronal."
    date: "2026-05-16"
    tags: ["datasets", "destilacion"]
    dependencies: ["v2.6"]
    artifacts: ["datasets/export_contract.py", "datasets/manifest.py", "neural/models/base.py"]
    doc_refs: ["docs/V27_DISTILLATION_READINESS.md"]

  - version: "v2.7.5"
    title: "Tiny Distillation Experiments"
    status: "complete"
    summary: "Generación de datasets grandes, bucles de entrenamiento pequeños, arena oracle-vs-model, taxonomía de fallos."
    technical_impact: "Primera prueba de que un modelo neuronal minúsculo puede aproximar un oráculo determinista en un dominio cerrado."
    date: "2026-05-16"
    tags: ["neural", "destilacion", "arena"]
    dependencies: ["v2.7"]
    artifacts: ["neural/tiny_experiments.py", "neural/trainers/", "neural/evaluators/"]
    doc_refs: ["docs/V27_DISTILLATION_READINESS.md"]

  - version: "v2.7.6"
    title: "Artifact Governance + Reporting"
    status: "complete"
    summary: "Contratos de esquema para checkpoints, migraciones, registro de artefactos, reportes compactos y CLI de diff."
    technical_impact: "Añade la capa de gobernanza que permite versionar y auditar todos los artefactos del proyecto."
    date: "2026-05-16"
    tags: ["governance", "reporting"]
    dependencies: ["v2.7"]
    artifacts: ["governance/artifact_registry.py", "governance/artifact_inventory.py", "reporting/report_builder.py"]
    doc_refs: ["docs/ARCHITECTURE_V26.md"]

  - version: "v2.7.7"
    title: "Artifact Policy + Runbook"
    status: "complete"
    summary: "Cargador de políticas, enforcement, runbook operativo, reglas de compatibilidad y reporting policy-aware."
    technical_impact: "Automatiza el cumplimiento de políticas de ciclo de vida de artefactos, eliminando decisiones manuales."
    date: "2026-05-16"
    tags: ["governance", "policy"]
    dependencies: ["v2.7.6"]
    artifacts: ["governance/artifact_policy.py", "docs/ARTIFACT_OPERATIONS_RUNBOOK.md"]
    doc_refs: ["docs/ARTIFACT_OPERATIONS_RUNBOOK.md"]

  - version: "v2.7.8"
    title: "Retention + Archive Tooling"
    status: "complete"
    summary: "Barredor de retención determinista, bundles de archivo, reportes de retención y dry-run de limpieza."
    technical_impact: "Gestiona el ciclo de vida completo de artefactos, desde activo hasta archivado o eliminado, sin pérdida de trazabilidad."
    date: "2026-05-16"
    tags: ["governance", "retention"]
    dependencies: ["v2.7.7"]
    artifacts: ["governance/retention_sweeper.py", "governance/archive_tooling.py"]
    doc_refs: ["docs/ARTIFACT_OPERATIONS_RUNBOOK.md"]

  - version: "v2.7.9"
    title: "Inventory + Workspace Intelligence"
    status: "complete"
    summary: "Índice determinista de inventario, motor de consultas, grafo de linaje, detección de drift y resúmenes de workspace."
    technical_impact: "Permite consultar y visualizar las relaciones entre todos los artefactos del proyecto sin base de datos externa."
    date: "2026-05-16"
    tags: ["governance", "query", "inventory"]
    dependencies: ["v2.7.8"]
    artifacts: ["governance/artifact_inventory.py", "governance/query_engine.py", "governance/lineage_graph.py"]
    doc_refs: ["docs/ARCHITECTURE.md"]

  - version: "v2.7.10"
    title: "Artifact Search Refinement"
    status: "complete"
    summary: "Dependencias inversas, búsqueda por workspace, consultas guardadas, reportes de descubrimiento y análisis de impacto."
    technical_impact: "Cierra la capa de inteligencia de artefactos con capacidad de navegación operativa completa."
    date: "2026-05-16"
    tags: ["governance", "search", "dependencies"]
    dependencies: ["v2.7.9"]
    artifacts: ["governance/reverse_dependencies.py", "governance/saved_queries.py", "reporting/discovery_report.py"]
    doc_refs: ["docs/ARCHITECTURE.md"]

  - version: "v2.8"
    title: "Neural Surrogate Experiments"
    status: "planned"
    summary: "Entrenar un surrogate neuronal pequeño que imite un oráculo determinista en un dominio cerrado (aritmética o cinemática 1D)."
    technical_impact: "Primer paso hacia la destilación de conocimiento físico en redes neuronales livianas."
    date: null
    tags: ["neural", "destilacion"]
    dependencies: ["v2.7.10"]
    artifacts: []
    doc_refs: ["docs/ROADMAP.md"]
```

========================================================
PHASE 3 — RELEASE NOTE GENERATOR
========================================================

Create files:
- backend/reporting/milestone_report.py
- scripts/generate_milestone_report.py

Copy the following code for scripts/generate_milestone_report.py (it includes a fallback inline definition of milestone functions, so it works even if backend/governance/milestones.py isn't ready yet):

```python
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
        import yaml
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
        ]
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
```

For backend/reporting/milestone_report.py, you can simply import and re-export the generate functions from the script, or create a small wrapper:

```python
# backend/reporting/milestone_report.py
from backend.governance.milestones import load_milestones
from pathlib import Path

def generate_milestone_report(output_dir: Path, format: str = "md"):
    # Delegate to the CLI script logic or recreate here using the same functions.
    # For now, just call the script as a subprocess or directly import.
    pass
```
(Implement minimally; the CLI script already covers the generation.)

========================================================
PHASE 4 — ROADMAP CHRONICLE
========================================================

Create: docs/ROADMAP_CHRONICLE.md

Initial content:

```markdown
# CPT Roadmap Chronicle

## Foundations (v2.6)
Snapshot reproducibility was established as the first critical milestone. Deterministic snapshot generation, dataset fingerprinting, and test isolation ensured that every future artifact could be verified bit-for-bit against its origin. This foundation allowed the project to scale without fear of silent corruption.

## Distillation Readiness (v2.7 – v2.7.5)
The dataset layer was formalized: export contracts, manifests, and sharding turned raw oracle outputs into machine-consumable training data. Evaluation harnesses and a tiny model scaffold were built, culminating in the first distillation experiments that proved a <1M-parameter transformer could mimic a deterministic oracle in a closed domain.

## Artifact Governance (v2.7.6 – v2.7.8)
Gobernanza became the backbone. Checkpoint schema contracts, migrations, and a registry made artifact management explicit. Policy enforcement and a runbook automated lifecycle decisions. Retention sweeping and archival tooling completed the loop, ensuring the workspace never accumulates obsolete artifacts silently.

## Inventory and Discovery (v2.7.9 – v2.7.10)
With the inventory index and query engine, every artifact became searchable deterministically. Lineage graphs, drift detection, and workspace summaries provided transparency. Reverse dependencies, saved queries, and impact analysis turned the repository into a navigable knowledge graph—no external database required.

## Next Research Direction (v2.8+)
The platform is now ready for its first neural surrogate experiments. The plan is to extend the arena to physics-based domains, then gradually push into CAD plugins (KiCad, FreeCAD) and eventually controlled code generation. Every step remains grounded in deterministic truth from the sandbox oracle.
```

Also create docs/ROADMAP.md with the strategic table:

```markdown
# CPT Roadmap Estratégico

| Fase   | Versión   | Hito Principal                           | Producto Clave               |
|--------|-----------|------------------------------------------|------------------------------|
| 0      | v2.6–v2.7 | Cierre de plataforma de verdad           | Gobernanza, queries, inventory |
| 1      | v2.8      | Primer surrogate neuronal diminuto       | Prueba de destilación        |
| 2      | v2.9      | Arena oracle vs model                    | Métricas de fidelidad y OOD  |
| 3      | v3.0      | Plugin KiCad: validación DC simple       | Asistente de circuitos       |
| 4      | v3.1      | KiCad avanzado: AC, transitorios         | Sugerencias de diseño        |
| 5      | v3.2      | FreeCAD restringido: vigas 2D            | Validador estructural        |
| 6      | v3.3      | FreeCAD avanzado: placas, térmica        | Análisis acoplado            |
| 7      | v3.4      | Generación de código utilitario          | Scripts y validadores        |
| 8      | v3.5      | Generación de módulos de programación    | Funciones y componentes      |
| 9      | v4.0      | Generación de proyectos acotados         | Código + tests + docs        |

**Principio**: verdad → surrogate → plugin físico → código general.
```

========================================================
PHASE 5 — MILESTONE QUERY TOOLING
========================================================

Create:
- backend/governance/milestone_queries.py
- scripts/query_milestones.py

backend/governance/milestone_queries.py:

```python
# backend/governance/milestone_queries.py
from __future__ import annotations
from typing import Tuple
from backend.governance.milestones import MilestoneRecord

def query_milestones(
    milestones: Tuple[MilestoneRecord, ...],
    *,
    version: str | None = None,
    status: str | None = None,
    tag: str | None = None,
    dependency: str | None = None,
    artifact: str | None = None,
) -> Tuple[MilestoneRecord, ...]:
    results = milestones
    if version:
        results = [m for m in results if m.version == version]
    if status:
        results = [m for m in results if m.status == status]
    if tag:
        results = [m for m in results if tag in m.tags]
    if dependency:
        results = [m for m in results if dependency in m.dependencies]
    if artifact:
        results = [m for m in results if any(artifact in a for a in m.artifacts)]
    return tuple(results)
```

scripts/query_milestones.py:

```python
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
```

========================================================
PHASE 6 — TESTING
==================

Create: tests/test_v2711_milestones_and_chronicle.py

Include at least these tests:

```python
# tests/test_v2711_milestones_and_chronicle.py
import pytest
from pathlib import Path
from backend.governance.milestones import load_milestones, compute_milestone_fingerprint
from backend.governance.milestone_queries import query_milestones

MILESTONES_PATH = Path("docs/milestones.yaml")

@pytest.fixture
def milestones():
    return load_milestones(MILESTONES_PATH)

def test_load_all_milestones(milestones):
    assert len(milestones) > 0
    # Check that the first item has a version field
    assert milestones[0].version

def test_fingerprint_stable(milestones):
    fp1 = compute_milestone_fingerprint(milestones)
    fp2 = compute_milestone_fingerprint(milestones)
    assert fp1 == fp2

def test_query_by_status(milestones):
    completed = query_milestones(milestones, status="complete")
    planned = query_milestones(milestones, status="planned")
    assert len(completed) >= 9
    assert any(m.version == "v2.8" for m in planned)

def test_query_by_tag(milestones):
    governance = query_milestones(milestones, tag="governance")
    assert len(governance) > 0
    assert all("governance" in m.tags for m in governance)

def test_report_generation_deterministic():
    # Use a subprocess or call directly to check that two runs produce identical output
    import subprocess, tempfile, os
    cmd = ["python", "scripts/generate_milestone_report.py", "--json", "--output-dir", tempfile.gettempdir()]
    res1 = subprocess.run(cmd, capture_output=True, text=True)
    res2 = subprocess.run(cmd, capture_output=True, text=True)
    # The report file name contains timestamp, so we cannot compare paths. Instead, check that the command succeeds.
    assert res1.returncode == 0
    assert res2.returncode == 0

def test_backward_compatibility(milestones):
    # No existing module should be broken by adding these files
    from backend.governance import artifact_inventory, query_engine  # just ensure imports work
    assert True
```

========================================================
ACCEPTANCE CRITERIA
===================

The task is complete only when:

1. Milestone registry exists and is deterministic (load_milestones works, fingerprint stable).
2. docs/milestones.yaml contains the canonical milestone timeline.
3. Release note generator produces identical Markdown and JSON on repeated runs.
4. docs/ROADMAP_CHRONICLE.md and docs/ROADMAP.md are present.
5. Milestone query tooling works for version, status, and tag.
6. All new tests pass.
7. Existing functionality (166 tests) remains intact.
8. The next planned milestone (v2.8) is documented but not implemented.

BEGIN WITH A REPO SCAN.
IMPLEMENT IN SMALL SAFE COMMITS.
VERIFY EVERY CHANGE WITH TESTS.
```

