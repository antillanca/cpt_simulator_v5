"""Permission policy for LLM and Hermes tooling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


CORE_TRUTH_PREFIXES = (
    "backend/core_truth/",
    "backend/verifiers/",
    "backend/dsl/",
)

LLM_ALLOWED_ACTIONS = {
    "generate_documentation",
    "suggest_refactor",
    "generate_tests",
    "verbalize_dataset",
    "verbalize_dataset_question",
    "generate_linguistic_variants",
    "create_wrapper",
    "analyze_logs",
    "detect_regressions",
    "optimize_dataset",
    "debugging_help",
    "generate_oracle_metadata",
}

LLM_DENIED_ACTIONS = {
    "modify_core_truth",
    "modify_verifiers",
    "modify_invariants",
    "modify_dsl_compiler",
    "approve_rules",
    "merge",
    "bypass_verification",
    "self_approve_physical_laws",
}

HERMES_DENIED_ACTIONS = {
    "merge",
    "modify_core_truth",
    "modify_verifiers",
    "modify_invariants",
    "bypass_verification",
}


def _normalise_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def path_is_core_truth(path: str | Path) -> bool:
    normalised = _normalise_path(path)
    return normalised.startswith(CORE_TRUTH_PREFIXES)


@dataclass
class PermissionDecision:
    allowed: bool
    reason: str = ""
    requires_human_approval: bool = False


@dataclass
class PermissionPolicy:
    """Simple, explicit policy for high-risk tooling operations."""

    denied_prefixes: tuple[str, ...] = CORE_TRUTH_PREFIXES
    human_approval_actions: set[str] = field(default_factory=lambda: {"merge", "apply_patch", "patch_core"})
    core_truth_actions: set[str] = field(
        default_factory=lambda: {
            "modify_core_truth",
            "modify_verifiers",
            "modify_invariants",
            "modify_dsl_compiler",
            "approve_rules",
            "bypass_verification",
            "self_approve_physical_laws",
        }
    )

    def can_llm_perform(self, action: str, target_path: str | Path | None = None) -> PermissionDecision:
        if action in LLM_DENIED_ACTIONS:
            return PermissionDecision(False, f"LLMs cannot perform '{action}'.")
        if target_path is not None and path_is_core_truth(target_path):
            return PermissionDecision(False, f"Target path '{target_path}' is protected core truth.")
        if action in self.core_truth_actions:
            return PermissionDecision(False, f"Action '{action}' requires human-governed core-truth authority.")
        if action in LLM_ALLOWED_ACTIONS:
            return PermissionDecision(True)
        return PermissionDecision(False, f"Action '{action}' is not in the LLM allowlist.")

    def can_hermes_perform(self, action: str, target_path: str | Path | None = None) -> PermissionDecision:
        if action in HERMES_DENIED_ACTIONS:
            return PermissionDecision(False, f"Hermes cannot perform '{action}'.")
        if target_path is not None and path_is_core_truth(target_path):
            return PermissionDecision(False, f"Target path '{target_path}' is protected core truth.")
        if action in self.core_truth_actions:
            return PermissionDecision(False, f"Action '{action}' is blocked from Hermes automation.")
        if action in self.human_approval_actions:
            return PermissionDecision(True, requires_human_approval=True)
        return PermissionDecision(True)

    def human_approved(self, action: str) -> bool:
        from os import getenv

        if action not in self.human_approval_actions:
            return True
        return getenv("CPT_HERMES_HUMAN_APPROVAL", "0") == "1"


default_policy = PermissionPolicy()
