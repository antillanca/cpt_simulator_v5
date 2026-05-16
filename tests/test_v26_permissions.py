from backend.tooling.permissions import default_policy


def test_llm_can_generate_oracle_metadata():
    decision = default_policy.can_llm_perform("generate_oracle_metadata")
    assert decision.allowed is True


def test_llm_cannot_bypass_verification():
    decision = default_policy.can_llm_perform("bypass_verification")
    assert decision.allowed is False

