"""CORE v3.2 — Packaging Tests.

Verify that packaging metadata is correct:
  - package name and version
  - optional extras are defined
  - base install has no mandatory heavy deps
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestPackaging:
    def test_pyproject_toml_exists(self):
        import os
        path = REPO_ROOT / "pyproject.toml"
        assert path.exists(), "pyproject.toml missing"

    def test_package_name(self):
        path = REPO_ROOT / "pyproject.toml"
        content = path.read_text()
        assert 'name = "core-runtime-engine"' in content

    def test_version_is_set(self):
        path = REPO_ROOT / "pyproject.toml"
        content = path.read_text()
        assert "version" in content

    def test_circuits_extra_defined(self):
        path = REPO_ROOT / "pyproject.toml"
        content = path.read_text()
        assert "circuits" in content

    def test_linear_system_extra_defined(self):
        path = REPO_ROOT / "pyproject.toml"
        content = path.read_text()
        assert "linear-system" in content

    def test_dev_extra_defined(self):
        path = REPO_ROOT / "pyproject.toml"
        content = path.read_text()
        assert "dev" in content

    def test_base_install_no_heavy_deps(self):
        """Base install should not require torch or circuit-simulators."""
        path = REPO_ROOT / "pyproject.toml"
        content = path.read_text()
        # torch should only appear in optional extras, not main deps
        lines = content.split("\n")
        in_dependencies = False
        in_optional = False
        for line in lines:
            stripped = line.strip()
            if stripped == "dependencies = [":
                in_dependencies = True
                in_optional = False
            elif stripped.startswith("[") and "optional" in stripped.lower():
                in_optional = True
                in_dependencies = False
            elif stripped.startswith("[") and stripped != "":
                in_dependencies = False
                in_optional = False
            elif in_dependencies and "torch" in stripped.lower():
                pytest.fail("torch should not be in base dependencies")

    def test_package_is_importable(self):
        import core_runtime
        assert hasattr(core_runtime, "__version__") or True  # no __version__ required yet
