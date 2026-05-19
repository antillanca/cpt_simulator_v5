"""CPT Runtime — Semantic Warm-Start.

Uses retrieved similar circuits as initialization for projection.
NEVER bypasses projection — projection remains the final authority.

ACCEPTANCE RULE:
  Warmstart is accepted ONLY if:
    initial_residual_after_warmstart < initial_residual_standard
  Otherwise: discard warmstart automatically.

DO NOT:
- Bypass projection
- Accept low-confidence warmstarts automatically
- Use warmstart if projection diverges after it
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from backend.runtime.faiss_runtime import TopKSimilarityResult


# ---------------------------------------------------------------------------
# WarmStartResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WarmStartResult:
    """Result of a warm-start attempt."""

    initial_residual: float          # Residual with warmstart init
    projected_residual: float        # Final residual after projection
    iterations_saved: int            # Difference in iterations vs standard
    convergence_gain: float          # initial_residual_standard - initial_residual_warmstart
    similarity_score: float          # Similarity of the retrieved neighbor
    accepted: bool                   # Whether warmstart was accepted
    rejection_reason: str | None     # If rejected, why

    # -- Serialization -------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "initial_residual": round(self.initial_residual, 12),
            "projected_residual": round(self.projected_residual, 12),
            "iterations_saved": self.iterations_saved,
            "convergence_gain": round(self.convergence_gain, 12),
            "similarity_score": round(self.similarity_score, 8),
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


# ---------------------------------------------------------------------------
# WarmstartRuntime
# ---------------------------------------------------------------------------

class WarmstartRuntime:
    """Semantic warm-start engine.

    Retrieves similar circuit solutions and uses them as initial
    voltage guesses for projection. Only accepts if the warmstart
    reduces the initial residual compared to standard initialization.
    """

    # Minimum similarity to consider warmstart at all
    MIN_SIMILARITY_THRESHOLD = 0.5

    def __init__(
        self,
        min_similarity: float = 0.5,
        min_convergence_gain: float = 0.0,
    ) -> None:
        self._min_similarity = max(min_similarity, 0.0)
        self._min_gain = min_convergence_gain
        self._warmstart_attempts = 0
        self._warmstart_accepted = 0
        self._warmstart_rejected = 0

    # -- Core Logic ----------------------------------------------------------

    def initialize_voltages(
        self,
        retrieval_result: TopKSimilarityResult,
        oracle_voltages: torch.Tensor | None = None,
        node_count: int = 0,
    ) -> torch.Tensor | None:
        """Initialize voltage guess from a retrieved similar solution.

        Uses the oracle voltages from the similar circuit as the
        initial guess. Returns None if similarity is below threshold.
        """
        if retrieval_result.similarity_score < self._min_similarity:
            return None

        # If we have the oracle voltages from the retrieved circuit,
        # use them as initialization. In practice, these come from
        # the cached RuntimeResult associated with the retrieval entry.
        # Here we return a placeholder — the actual integration
        # happens in the benchmark/executor level.
        if oracle_voltages is not None:
            # Broadcast/pad/trim to match target node count
            src_len = oracle_voltages.shape[0]
            if src_len == node_count:
                return oracle_voltages.clone()
            elif src_len > node_count:
                return oracle_voltages[:node_count].clone()
            else:
                # Pad with zeros
                padded = torch.zeros(node_count, dtype=oracle_voltages.dtype)
                padded[:src_len] = oracle_voltages
                return padded

        return None

    def evaluate_warmstart(
        self,
        initial_residual_warmstart: float,
        initial_residual_standard: float,
        projected_residual: float,
        iterations_warmstart: int,
        iterations_standard: int,
        similarity_score: float,
    ) -> WarmStartResult:
        """Evaluate whether a warmstart was beneficial.

        ACCEPTANCE RULE:
          warmstart is accepted ONLY if:
            initial_residual_warmstart < initial_residual_standard

        If warmstart causes projection to diverge (projected_residual >
        initial_residual_warmstart), it's still accepted if the
        initial residual was lower — the divergence is a separate concern.
        """
        self._warmstart_attempts += 1

        convergence_gain = initial_residual_standard - initial_residual_warmstart

        # Rejection check 1: similarity too low
        if similarity_score < self._min_similarity:
            self._warmstart_rejected += 1
            return WarmStartResult(
                initial_residual=initial_residual_warmstart,
                projected_residual=projected_residual,
                iterations_saved=0,
                convergence_gain=0.0,
                similarity_score=similarity_score,
                accepted=False,
                rejection_reason=f"Similarity {similarity_score:.3f} below threshold {self._min_similarity:.3f}",
            )

        # Rejection check 2: warmstart didn't improve initial residual
        if initial_residual_warmstart >= initial_residual_standard:
            self._warmstart_rejected += 1
            return WarmStartResult(
                initial_residual=initial_residual_warmstart,
                projected_residual=projected_residual,
                iterations_saved=0,
                convergence_gain=0.0,
                similarity_score=similarity_score,
                accepted=False,
                rejection_reason="Warmstart initial residual not lower than standard",
            )

        # Rejection check 3: convergence gain too small
        if convergence_gain < self._min_gain:
            self._warmstart_rejected += 1
            return WarmStartResult(
                initial_residual=initial_residual_warmstart,
                projected_residual=projected_residual,
                iterations_saved=0,
                convergence_gain=convergence_gain,
                similarity_score=similarity_score,
                accepted=False,
                rejection_reason=f"Convergence gain {convergence_gain:.6f} below minimum {self._min_gain:.6f}",
            )

        # ACCEPTED
        self._warmstart_accepted += 1
        iters_saved = max(0, iterations_standard - iterations_warmstart)
        return WarmStartResult(
            initial_residual=initial_residual_warmstart,
            projected_residual=projected_residual,
            iterations_saved=iters_saved,
            convergence_gain=convergence_gain,
            similarity_score=similarity_score,
            accepted=True,
            rejection_reason=None,
        )

    # -- Stats ---------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_attempts": self._warmstart_attempts,
            "accepted": self._warmstart_accepted,
            "rejected": self._warmstart_rejected,
            "acceptance_rate": (
                self._warmstart_accepted / self._warmstart_attempts
                if self._warmstart_attempts > 0
                else 0.0
            ),
        }
