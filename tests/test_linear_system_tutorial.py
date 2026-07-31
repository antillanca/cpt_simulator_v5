"""CORE v3.2 — Linear System Tutorial Test.

Verify that the tutorial example runs end-to-end without errors.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TUTORIAL_SCRIPT = REPO / "examples" / "01_linear_system_walkthrough.py"


class TestLinearSystemTutorial:
    def test_tutorial_runs(self):
        result = subprocess.run(
            [sys.executable, str(TUTORIAL_SCRIPT)],
            capture_output=True, text=True, timeout=60,
            cwd=str(REPO),
        )
        assert result.returncode == 0, f"Tutorial failed:\n{result.stderr}\n{result.stdout}"

    def test_tutorial_produces_output(self):
        result = subprocess.run(
            [sys.executable, str(TUTORIAL_SCRIPT)],
            capture_output=True, text=True, timeout=60,
            cwd=str(REPO),
        )
        output = result.stdout
        # Key sections the tutorial should display
        assert "Step 1" in output or "DomainTask" in output or "LinearSystemTask" in output
        assert "projection" in output.lower() or "Projection" in output

    def test_tutorial_is_deterministic(self):
        """Running the tutorial twice produces the same output."""
        r1 = subprocess.run(
            [sys.executable, str(TUTORIAL_SCRIPT)],
            capture_output=True, text=True, timeout=60,
            cwd=str(REPO),
        )
        r2 = subprocess.run(
            [sys.executable, str(TUTORIAL_SCRIPT)],
            capture_output=True, text=True, timeout=60,
            cwd=str(REPO),
        )
        # Key lines should match (not comparing full output due to timing)
        lines1 = [l for l in r1.stdout.split("\n") if "residual" in l.lower() or "iterations" in l.lower()]
        lines2 = [l for l in r2.stdout.split("\n") if "residual" in l.lower() or "iterations" in l.lower()]
        assert lines1 == lines2, "Tutorial output is not deterministic"
