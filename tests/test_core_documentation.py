"""CORE v3.1 -- Documentation completeness tests.

Verify that all required documentation files exist and contain key sections.
"""

from __future__ import annotations

import os
import pytest


DOCS_DIR = "docs"
REQUIRED_DOCS = [
    "CORE_ARCHITECTURE.md",
    "CORE_PRINCIPLES.md",
    "PAPER_POSITIONING.md",
    "V215_ADAPTIVE_RUNTIME_SCHEDULING.md",
    "V215_STABILITY_GUARANTEES.md",
    "THIRD_DOMAIN_PLAN.md",
    "MIGRATION_LOG.md",
    "CORE_RELEASE_README.md",
]


class TestRequiredDocsExist:

    @pytest.mark.parametrize("docname", REQUIRED_DOCS)
    def test_doc_exists(self, docname):
        path = os.path.join(DOCS_DIR, docname)
        assert os.path.exists(path), f"Missing required doc: {path}"


class TestArchitectureDoc:

    def test_has_positioning_statement(self):
        path = os.path.join(DOCS_DIR, "CORE_ARCHITECTURE.md")
        with open(path) as f:
            content = f.read()
        assert "deterministic hybrid runtime" in content.lower()

    def test_has_observability_section(self):
        path = os.path.join(DOCS_DIR, "CORE_ARCHITECTURE.md")
        with open(path) as f:
            content = f.read()
        assert "observ" in content.lower()


class TestPrinciplesDoc:

    def test_has_positioning_statement(self):
        path = os.path.join(DOCS_DIR, "CORE_PRINCIPLES.md")
        with open(path) as f:
            content = f.read()
        assert "deterministic hybrid runtime" in content.lower()

    def test_has_determinism_guarantee(self):
        path = os.path.join(DOCS_DIR, "CORE_PRINCIPLES.md")
        with open(path) as f:
            content = f.read()
        assert "determinist" in content.lower()


class TestStabilityGuarantees:

    def test_has_frozen_apis(self):
        path = os.path.join(DOCS_DIR, "V215_STABILITY_GUARANTEES.md")
        with open(path) as f:
            content = f.read()
        assert "frozen" in content.lower() or "API" in content


class TestPaperPositioning:

    def test_has_positioning_statement(self):
        path = os.path.join(DOCS_DIR, "PAPER_POSITIONING.md")
        with open(path) as f:
            content = f.read()
        assert "deterministic hybrid runtime" in content.lower()

    def test_has_cpt_reference(self):
        path = os.path.join(DOCS_DIR, "PAPER_POSITIONING.md")
        with open(path) as f:
            content = f.read()
        # Should reference CPT somewhere
        assert "CPT" in content or "circuit" in content.lower()


class TestThirdDomainPlan:

    def test_has_domain_name(self):
        path = os.path.join(DOCS_DIR, "THIRD_DOMAIN_PLAN.md")
        with open(path) as f:
            content = f.read()
        assert "propositional" in content.lower() or "symbolic" in content.lower() or "domain" in content.lower()

    def test_has_implementation_note(self):
        path = os.path.join(DOCS_DIR, "THIRD_DOMAIN_PLAN.md")
        with open(path) as f:
            content = f.read()
        assert "implement" in content.lower()
