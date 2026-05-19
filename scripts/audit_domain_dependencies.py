#!/usr/bin/env python3
"""CPT CORE -- Domain Dependency Audit.

Identifies circuit-specific imports across the codebase to determine
what must move into domains/circuits/ and what stays in core/.

Outputs: workspace/runtime_reports/domain_dependency_audit.json
"""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CIRCUIT_SYMBOLS = {
    "MNA", "KCL", "KVL", "circuit_arena", "topology_families",
    "CircuitGraph", "CanonicalCircuitGraph", "Circuit", "CircuitSolution",
    "PhysicsProjection", "dc_solver", "solve_dc_circuit",
}

CIRCUIT_IMPORTS = {
    "backend.circuits",
    "backend.core_runtime.oracle_protocol",
    "backend.core_runtime.projection_runtime",
    "backend.core_runtime.surrogate_runtime",
    "backend.core_runtime.confidence_runtime",
}


@dataclass
class ModuleAudit:
    filepath: str
    is_circuit_dependent: bool = False
    circuit_imports: list[str] = field(default_factory=list)
    circuit_symbols: list[str] = field(default_factory=list)
    recommendation: str = "core"


def audit_file(filepath: Path) -> ModuleAudit:
    audit = ModuleAudit(filepath=str(filepath.relative_to(REPO_ROOT)))
    try:
        source = filepath.read_text(encoding="utf-8")
    except Exception:
        return audit

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return audit

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(ci) for ci in CIRCUIT_IMPORTS):
                    audit.circuit_imports.append(alias.name)
                    audit.is_circuit_dependent = True
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                if any(node.module.startswith(ci) for ci in CIRCUIT_IMPORTS):
                    audit.circuit_imports.append(node.module)
                    audit.is_circuit_dependent = True

        if isinstance(node, ast.Name) and node.id in CIRCUIT_SYMBOLS:
            audit.circuit_symbols.append(node.id)
            audit.is_circuit_dependent = True
        if isinstance(node, ast.Attribute) and node.attr in CIRCUIT_SYMBOLS:
            audit.circuit_symbols.append(node.attr)
            audit.is_circuit_dependent = True

    audit.recommendation = "circuits" if audit.is_circuit_dependent else "core"
    return audit


def main() -> None:
    targets = list(REPO_ROOT.glob("backend/runtime/*.py"))
    targets += list(REPO_ROOT.glob("backend/core_runtime/*.py"))
    targets += list(REPO_ROOT.glob("backend/core_spec/*.py"))

    audits = []
    core_modules = []
    circuit_modules = []

    for filepath in sorted(targets):
        if filepath.name == "__init__.py":
            continue
        audit = audit_file(filepath)
        entry = {
            "filepath": audit.filepath,
            "is_circuit_dependent": audit.is_circuit_dependent,
            "circuit_imports": audit.circuit_imports,
            "circuit_symbols": audit.circuit_symbols,
            "recommendation": audit.recommendation,
        }
        audits.append(entry)
        if audit.is_circuit_dependent:
            circuit_modules.append(audit.filepath)
        else:
            core_modules.append(audit.filepath)

    report = {
        "total_modules_audited": len(audits),
        "core_modules": core_modules,
        "circuit_modules": circuit_modules,
        "core_count": len(core_modules),
        "circuit_count": len(circuit_modules),
        "details": audits,
    }

    out_path = REPO_ROOT / "workspace" / "runtime_reports" / "domain_dependency_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Domain Dependency Audit")
    print(f"Modules audited: {len(audits)}")
    print(f"Core (domain-agnostic): {len(core_modules)}")
    print(f"Circuit-dependent: {len(circuit_modules)}")
    for m in circuit_modules:
        print(f"  CIRCUIT: {m}")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
