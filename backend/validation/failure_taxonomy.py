"""Stable failure taxonomy for oracle-vs-model comparison."""

from __future__ import annotations

from backend.validation.oracle_arena import ArenaResult


def classify_failure(result: ArenaResult) -> str | None:
    """Return a stable failure category or None if the sample passed."""

    if result.exact_match and result.replay_consistency and not result.invariant_violation:
        return None
    if not result.replay_consistency:
        return "replay_instability"
    if result.invariant_violation:
        return "invariant_violation"
    if not result.struct_match:
        return "structure_mismatch"
    if result.trace_consistency == 0.0 and result.trajectory_deviation > 0:
        return "hallucinated_step"
    if result.trace_consistency < 0.5:
        return "trace_divergence"
    if result.answer_consistency == 0.0 and result.trace_consistency >= 0.5:
        return "causal_discontinuity"
    if result.trajectory_deviation > 0:
        return "compositional_failure"
    if result.answer_consistency < 1.0:
        return "extrapolation_collapse"
    return "trace_divergence"
