"""Compatibility wrapper for the hardened sandbox manager."""

from backend.core_truth.sandbox import SecureSandboxManager, sandbox_manager


class SandboxManager(SecureSandboxManager):
    pass

