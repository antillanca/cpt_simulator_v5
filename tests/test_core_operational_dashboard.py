"""CORE v3.2 — Operational Dashboard Test.

Verify that the dashboard generator runs and produces valid HTML.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DASHBOARD_SCRIPT = REPO / "scripts" / "generate_operational_dashboard.py"
OUTPUT_DIR = REPO / "workspace" / "operational_dashboard"


class TestOperationalDashboard:
    def test_dashboard_script_runs(self):
        result = subprocess.run(
            [sys.executable, str(DASHBOARD_SCRIPT)],
            capture_output=True, text=True, timeout=60,
            cwd=str(REPO),
        )
        assert result.returncode == 0, f"Dashboard script failed:\n{result.stderr}"

    def test_dashboard_produces_index_html(self):
        # Run the script first
        subprocess.run(
            [sys.executable, str(DASHBOARD_SCRIPT)],
            capture_output=True, text=True, timeout=60,
            cwd=str(REPO),
        )
        index_html = OUTPUT_DIR / "index.html"
        assert index_html.exists(), "index.html not generated"

    def test_dashboard_html_is_valid(self):
        subprocess.run(
            [sys.executable, str(DASHBOARD_SCRIPT)],
            capture_output=True, text=True, timeout=60,
            cwd=str(REPO),
        )
        index_html = OUTPUT_DIR / "index.html"
        if index_html.exists():
            content = index_html.read_text()
            assert "<!DOCTYPE html>" in content
            assert "CORE v3.2" in content
            assert "Operational Dashboard" in content

    def test_dashboard_contains_key_sections(self):
        subprocess.run(
            [sys.executable, str(DASHBOARD_SCRIPT)],
            capture_output=True, text=True, timeout=60,
            cwd=str(REPO),
        )
        index_html = OUTPUT_DIR / "index.html"
        if index_html.exists():
            content = index_html.read_text()
            # Key sections from the spec
            assert "Projection Iterations" in content
            assert "Routing Distribution" in content
            assert "Runtime Distribution" in content
