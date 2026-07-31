"""CORE v3.2 — Runtime Configuration with Feature Flags.

Frozen feature flags for future capabilities. All default to False.
These flags exist in config, are testable, are safe to ignore by current
runtime, and do NOT activate anything yet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    """Frozen runtime configuration with feature flags.

    All flags default to False (disabled). The current runtime does not
    read or act on these flags — they exist only as a declaration of
    future capability boundaries. When a capability is implemented, the
    corresponding flag will gate its activation.

    IMPORTANT: These flags must NOT be removed or changed to default=True
    without updating the principle regression tests.
    """

    # --- Future capability flags (all disabled) ---

    enable_lora_experts: bool = False
    """Enable LoRA expert adapters in the surrogate layer.
    NOT IMPLEMENTED — reserved for future learning systems."""

    enable_replay: bool = False
    """Enable experience replay from operational experience data.
    NOT IMPLEMENTED — reserved for future learning systems."""

    enable_continual_training: bool = False
    """Enable continual training of surrogates from new experience.
    NOT IMPLEMENTED — reserved for future learning systems."""

    enable_distributed_execution: bool = False
    """Enable distributed execution across multiple nodes.
    NOT IMPLEMENTED — reserved for future scaling."""

    # --- Operational flags (active, safe defaults) ---

    enable_exact_cache: bool = True
    """Enable exact cache hit resolution. Always True in v3.2."""

    enable_semantic_retrieval: bool = True
    """Enable semantic retrieval from experience memory. Always True in v3.2."""

    enable_warmstart_projection: bool = True
    """Enable warmstart from retrieved experience. Always True in v3.2."""

    enable_degraded_execution: bool = True
    """Enable degraded execution fallback. Always True in v3.2."""

    # --- Sandbox flags ---

    enable_training_sandbox: bool = False
    """Enable the training sandbox boundary.
    NOT IMPLEMENTED — reserved for future training isolation."""

    sandbox_require_promotion: bool = True
    """Require explicit promotion before sandbox artifacts go live.
    Always True — this is a safety guarantee."""


# --- Singleton default config ---

DEFAULT_CONFIG = RuntimeConfig()


def get_config() -> RuntimeConfig:
    """Get the current runtime configuration.

    Returns the frozen default config. In the future, this may
    load from environment variables or a config file, but the
    returned object will always be frozen.
    """
    return DEFAULT_CONFIG
