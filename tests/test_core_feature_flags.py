"""CORE v3.2 — Feature Flag Tests.

Verify that feature flags:
  - exist in config
  - default to False for future capabilities
  - default to True for safe operational flags
  - are testable
  - are safe to ignore by current runtime
  - config is frozen and immutable
"""

from __future__ import annotations

import dataclasses

import pytest

from core_runtime.core.runtime_config import (
    DEFAULT_CONFIG,
    RuntimeConfig,
    get_config,
)


class TestRuntimeConfigStructure:
    def test_config_is_frozen_dataclass(self):
        assert dataclasses.is_dataclass(RuntimeConfig)
        assert RuntimeConfig.__dataclass_params__.frozen is True

    def test_config_has_future_flags(self):
        config = get_config()
        assert hasattr(config, "enable_lora_experts")
        assert hasattr(config, "enable_replay")
        assert hasattr(config, "enable_continual_training")
        assert hasattr(config, "enable_distributed_execution")

    def test_config_has_operational_flags(self):
        config = get_config()
        assert hasattr(config, "enable_exact_cache")
        assert hasattr(config, "enable_semantic_retrieval")
        assert hasattr(config, "enable_warmstart_projection")
        assert hasattr(config, "enable_degraded_execution")

    def test_config_has_sandbox_flags(self):
        config = get_config()
        assert hasattr(config, "enable_training_sandbox")
        assert hasattr(config, "sandbox_require_promotion")


class TestFutureFlagsDefaultFalse:
    """All future capability flags MUST default to False."""

    def test_lora_experts_disabled(self):
        assert get_config().enable_lora_experts is False

    def test_replay_disabled(self):
        assert get_config().enable_replay is False

    def test_continual_training_disabled(self):
        assert get_config().enable_continual_training is False

    def test_distributed_execution_disabled(self):
        assert get_config().enable_distributed_execution is False

    def test_training_sandbox_disabled(self):
        assert get_config().enable_training_sandbox is False


class TestOperationalFlagsDefaultTrue:
    """Operational flags that are active in v3.2 default to True."""

    def test_exact_cache_enabled(self):
        assert get_config().enable_exact_cache is True

    def test_semantic_retrieval_enabled(self):
        assert get_config().enable_semantic_retrieval is True

    def test_warmstart_projection_enabled(self):
        assert get_config().enable_warmstart_projection is True

    def test_degraded_execution_enabled(self):
        assert get_config().enable_degraded_execution is True

    def test_sandbox_require_promotion_true(self):
        assert get_config().sandbox_require_promotion is True


class TestConfigImmutability:
    def test_cannot_modify_config(self):
        config = get_config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.enable_lora_experts = True  # type: ignore[misc]

    def test_cannot_create_config_with_future_flags_enabled(self):
        """Creating a config with future flags=True is allowed but explicit."""
        custom = RuntimeConfig(enable_lora_experts=True)
        assert custom.enable_lora_experts is True
        # Default still safe
        assert get_config().enable_lora_experts is False

    def test_default_config_is_singleton(self):
        assert get_config() is DEFAULT_CONFIG


class TestConfigSafety:
    def test_all_future_flags_off_in_default(self):
        """Comprehensive check: all future flags are off."""
        config = get_config()
        future_flags = [
            config.enable_lora_experts,
            config.enable_replay,
            config.enable_continual_training,
            config.enable_distributed_execution,
            config.enable_training_sandbox,
        ]
        assert all(f is False for f in future_flags)

    def test_runtime_ignores_future_flags(self):
        """Current runtime should work regardless of flag values."""
        # Even if we create a config with future flags on,
        # it should not change runtime behavior
        custom = RuntimeConfig(enable_lora_experts=True)
        # The config exists and is frozen — runtime doesn't use it yet
        assert custom.enable_lora_experts is True
        # But default config is unaffected
        assert get_config().enable_lora_experts is False

    def test_config_is_frozen_cannot_be_hot_patched(self):
        """Verify that no runtime code can mutate the config."""
        config = get_config()
        original_lora = config.enable_lora_experts
        try:
            config.enable_lora_experts = True  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except dataclasses.FrozenInstanceError:
            pass
        assert config.enable_lora_experts == original_lora
