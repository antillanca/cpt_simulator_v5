from backend.tooling.permissions import default_policy


def test_core_truth_is_protected():
    decision = default_policy.can_llm_perform("suggest_refactor", "backend/core_truth/sandbox.py")
    assert decision.allowed is False


def test_llm_cannot_modify_invariants():
    decision = default_policy.can_llm_perform("modify_invariants")
    assert decision.allowed is False

