"""Hermes tooling assistant wrapper.

This layer keeps Hermes useful as an auxiliary tool while preventing it from
touching core truth components or merging changes automatically.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from backend.tooling.permissions import default_policy

logger = logging.getLogger(__name__)


@dataclass
class HermesActionResult:
    allowed: bool
    approved: bool
    stdout: str = ""
    stderr: str = ""
    message: str = ""


class IntelligentToolingAssistant:
    def __init__(self, policy=default_policy):
        self.policy = policy

    async def analyze_logs(self, logs: str, skill: str = "cpt-debugger", model: str | None = None) -> HermesActionResult:
        decision = self.policy.can_hermes_perform("analyze_logs")
        if not decision.allowed:
            return HermesActionResult(False, False, message=decision.reason)
        return await self._invoke(skill, model, logs)

    async def suggest_patch(self, prompt: str, target_path: str | None = None, skill: str = "cpt-debugger", model: str | None = None) -> HermesActionResult:
        decision = self.policy.can_hermes_perform("apply_patch", target_path=target_path)
        if not decision.allowed:
            return HermesActionResult(False, False, message=decision.reason)
        approved = decision.requires_human_approval and self.policy.human_approved("apply_patch")
        if decision.requires_human_approval and not approved:
            return HermesActionResult(True, False, message="Human approval required before patch execution.")
        return await self._invoke(skill, model, prompt)

    async def create_pr_draft(self, prompt: str, skill: str = "cpt-debugger", model: str | None = None) -> HermesActionResult:
        return await self._invoke(skill, model, prompt)

    async def _invoke(self, skill: str, model: str | None, prompt: str) -> HermesActionResult:
        try:
            args = ["hermes", "--skills", skill]
            if model:
                args.extend(["-m", model])
            args.extend(["-z", prompt])
            
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            stdout_text = stdout.decode(errors="replace")
            stderr_text = stderr.decode(errors="replace")
            if process.returncode == 0:
                return HermesActionResult(True, True, stdout=stdout_text, stderr=stderr_text)
            
            # Si falló, capturamos el error
            error_msg = stderr_text[:500] or stdout_text[:500] or "Unknown error"
            logger.error("Hermes failed (rc=%s): %s", process.returncode, error_msg)
            return HermesActionResult(True, False, stdout=stdout_text, stderr=stderr_text, message=error_msg)
        except FileNotFoundError:
            return HermesActionResult(False, False, message="Hermes binary not found.")
        except Exception as exc:
            logger.error("Hermes invocation failed: %s", exc)
            return HermesActionResult(False, False, message=str(exc))


hermes_assistant = IntelligentToolingAssistant()
