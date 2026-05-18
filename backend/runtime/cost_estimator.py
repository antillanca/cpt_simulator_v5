"""CPT Runtime — Execution Cost Estimation.

PURE HEURISTICS ONLY. No trained model yet.

Estimates execution cost based on:
- node count
- edge count
- topology family
- current sources
- resistance dynamic range
- prior retrieval statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# ExecutionCostEstimate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionCostEstimate:
    """Immutable cost estimate for a task execution."""

    estimated_projection_iterations: int
    estimated_runtime_ms: float
    estimated_memory_cost: float
    estimated_difficulty: str  # "trivial", "easy", "moderate", "hard", "extreme"
    estimated_confidence: float

    # -- Validation ----------------------------------------------------------

    def __post_init__(self) -> None:
        valid_difficulties = {"trivial", "easy", "moderate", "hard", "extreme"}
        if self.estimated_difficulty not in valid_difficulties:
            raise ValueError(
                f"Invalid difficulty: {self.estimated_difficulty}, "
                f"expected one of {valid_difficulties}"
            )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "estimated_projection_iterations": self.estimated_projection_iterations,
            "estimated_runtime_ms": round(self.estimated_runtime_ms, 3),
            "estimated_memory_cost": round(self.estimated_memory_cost, 3),
            "estimated_difficulty": self.estimated_difficulty,
            "estimated_confidence": round(self.estimated_confidence, 8),
        }


# ---------------------------------------------------------------------------
# CostEstimator
# ---------------------------------------------------------------------------

class CostEstimator:
    """Heuristic execution cost estimator.

    Uses graph topology, component ranges, and prior statistics
    to estimate execution cost. DETERMINISTIC: same inputs → same estimate.
    """

    # Base cost parameters (tuned heuristically)
    _BASE_RUNTIME_MS = 5.0
    _PER_NODE_RUNTIME_MS = 0.5
    _PER_EDGE_RUNTIME_MS = 0.3
    _PROJ_ITER_PER_NODE = 0.5
    _OOD_MULTIPLIER = 2.0
    _MEMORY_COST_PER_ENTRY = 0.001  # KB

    def __init__(
        self,
        prior_stats: dict[str, Any] | None = None,
    ) -> None:
        self._prior_stats = prior_stats or {}
        # Historical averages per topology
        self._topo_avg_iters: dict[str, float] = self._prior_stats.get(
            "topology_avg_iterations", {}
        )

    def estimate(
        self,
        node_count: int,
        edge_count: int,
        topology_family: str = "unknown",
        current_sources: int = 0,
        resistance_range: tuple[float, float] = (1.0, 1.0),
        likely_ood: bool = False,
        confidence: float = 1.0,
    ) -> ExecutionCostEstimate:
        """Estimate execution cost for a task.

        Returns a deterministic cost estimate.
        """
        # --- Projection iterations ---
        # Base: proportional to graph size
        base_iters = max(3, int(node_count * self._PROJ_ITER_PER_NODE))

        # Historical adjustment
        topo_avg = self._topo_avg_iters.get(topology_family)
        if topo_avg is not None and topo_avg > 0:
            # Blend with historical average (50/50)
            base_iters = int(0.5 * base_iters + 0.5 * topo_avg)

        # OOD multiplier
        if likely_ood:
            base_iters = int(base_iters * self._OOD_MULTIPLIER)

        # Confidence adjustment: low confidence → more iterations
        confidence_factor = 1.0 + (1.0 - confidence) * 0.5
        base_iters = int(base_iters * confidence_factor)

        # Current sources add complexity
        base_iters += current_sources

        # Cap at reasonable maximum
        proj_iters = min(max(base_iters, 1), 50)

        # --- Runtime estimate ---
        runtime = (
            self._BASE_RUNTIME_MS
            + node_count * self._PER_NODE_RUNTIME_MS
            + edge_count * self._PER_EDGE_RUNTIME_MS
            + proj_iters * 0.5  # Projection iteration cost
        )
        if likely_ood:
            runtime *= self._OOD_MULTIPLIER

        # --- Memory cost ---
        mem_cost = (
            node_count * self._MEMORY_COST_PER_ENTRY
            + edge_count * self._MEMORY_COST_PER_ENTRY
        )

        # --- Difficulty ---
        r_min, r_max = resistance_range
        dynamic_range = r_max / max(r_min, 1e-12)
        difficulty = self._classify_difficulty(
            node_count, edge_count, dynamic_range, likely_ood, confidence
        )

        return ExecutionCostEstimate(
            estimated_projection_iterations=proj_iters,
            estimated_runtime_ms=round(runtime, 3),
            estimated_memory_cost=round(mem_cost, 3),
            estimated_difficulty=difficulty,
            estimated_confidence=round(confidence, 8),
        )

    def _classify_difficulty(
        self,
        node_count: int,
        edge_count: int,
        resistance_dynamic_range: float,
        likely_ood: bool,
        confidence: float,
    ) -> str:
        """Classify execution difficulty (deterministic)."""
        # Score from 0 (trivial) to 4 (extreme)
        score = 0

        # Graph size
        if node_count <= 3:
            pass  # trivial
        elif node_count <= 10:
            score += 1
        elif node_count <= 50:
            score += 2
        else:
            score += 3

        # Resistance dynamic range
        if resistance_dynamic_range > 1000:
            score += 1
        elif resistance_dynamic_range > 100:
            score += 0.5

        # OOD penalty
        if likely_ood:
            score += 1

        # Low confidence penalty
        if confidence < 0.3:
            score += 1
        elif confidence < 0.5:
            score += 0.5

        # Map score to difficulty
        if score < 0.5:
            return "trivial"
        elif score < 1.5:
            return "easy"
        elif score < 2.5:
            return "moderate"
        elif score < 3.5:
            return "hard"
        else:
            return "extreme"

    def update_prior_stats(self, topology_family: str, actual_iterations: int) -> None:
        """Update historical statistics with actual results."""
        current = self._topo_avg_iters.get(topology_family, 0.0)
        # Exponential moving average (alpha=0.3)
        alpha = 0.3
        self._topo_avg_iters[topology_family] = alpha * actual_iterations + (1 - alpha) * current
