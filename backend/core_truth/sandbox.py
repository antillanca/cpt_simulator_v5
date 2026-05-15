"""Deterministic Lua sandbox runner.

This is the canonical sandbox implementation. Compatibility wrappers in
``backend.sandbox`` should delegate here.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from backend.config import (
    SANDBOX_CPUS,
    SANDBOX_IMAGE,
    SANDBOX_MEMORY,
    SANDBOX_PIDS_LIMIT,
    SANDBOX_TIMEOUT_MS,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
LUA_RUNNER = BASE_DIR / "sandbox" / "lua" / "sandbox_runner.lua"


@dataclass(frozen=True)
class SandboxResult:
    status: str
    payload: dict[str, Any]


class SecureSandboxManager:
    """Run Lua rules in a hardened Docker sandbox."""

    def __init__(self, image_name: Optional[str] = None, runner_path: Optional[Path] = None):
        self.image_name = image_name or SANDBOX_IMAGE
        self.runner_path = runner_path or LUA_RUNNER

    def run_rule(
        self,
        rule_text: str,
        initial_state: dict | None = None,
        timeout_ms: int | None = None,
        frames: int = 1,
        collect_trace: bool = False,
    ) -> dict:
        """Execute a Lua rule in an isolated Docker container."""

        if initial_state is None:
            initial_state = {}

        timeout = (timeout_ms or SANDBOX_TIMEOUT_MS) / 1000.0
        payload = json.dumps(
            {
                "particle": initial_state,
                "rule": rule_text,
                "frames": frames,
                "collect_trace": collect_trace,
            }
        )

        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            SANDBOX_MEMORY,
            "--cpus",
            str(SANDBOX_CPUS),
            "--pids-limit",
            str(SANDBOX_PIDS_LIMIT),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "-i",
            self.image_name,
            "lua5.4",
            "/sandbox/sandbox_runner.lua",
        ]

        try:
            result = subprocess.run(
                cmd,
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                logger.warning("Sandbox exit code %s: %s", result.returncode, stderr[:240])
                try:
                    return json.loads(stderr)
                except json.JSONDecodeError:
                    return {
                        "status": "error",
                        "message": f"Sandbox exit {result.returncode}: {stderr[:240]}",
                    }

            stdout = result.stdout.strip()
            if not stdout:
                return {"status": "error", "message": "Sandbox produced no output."}

            return json.loads(stdout)

        except subprocess.TimeoutExpired:
            return {"status": "error", "message": f"Sandbox timeout ({timeout}s)"}
        except json.JSONDecodeError as exc:
            return {"status": "error", "message": f"Invalid JSON from sandbox: {exc}"}
        except FileNotFoundError:
            return {"status": "error", "message": "Docker not found. Is docker installed and in PATH?"}
        except Exception as exc:
            return {"status": "error", "message": f"Sandbox error: {exc}"}


sandbox_manager = SecureSandboxManager()

