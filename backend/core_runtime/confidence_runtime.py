"""CPT Core Runtime — Confidence Estimation.

Lightweight DETERMINISTIC confidence estimation using heuristics only.
NO stochastic dropout, NO random sampling, NO ensembles.

Inputs: topology family, graph size, resistance dynamic range,
        projection effort history, nearest historical failure rate,
        raw KCL residual before projection.
Output: ConfidenceEstimate (score, estimated iterations, likely OOD).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from backend.core_runtime.task_runtime import RuntimeTask


# ---------------------------------------------------------------------------
# ConfidenceEstimate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfidenceEstimate:
    """Deterministic confidence estimate for a task.

    confidence_score:            0.0 (low) to 1.0 (high).
    estimated_projection_iterations:  How many iterations projection will need.
    likely_ood:                  True if the task looks out-of-distribution.
    """
    confidence_score: float
    estimated_projection_iterations: int
    likely_ood: bool

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(f"confidence_score must be in [0,1], got {self.confidence_score}")
        if self.estimated_projection_iterations < 0:
            raise ValueError(f"estimated_projection_iterations must be >= 0, got {self.estimated_projection_iterations}")


# ---------------------------------------------------------------------------
# ConfidenceRuntime — deterministic heuristic estimator
# ---------------------------------------------------------------------------

class ConfidenceRuntime:
    """Estimate task confidence from deterministic heuristics.

    Heuristics used (all deterministic):
    1. Graph size:            larger graphs → lower confidence
    2. Resistance range:      high dynamic range → lower confidence
    3. Topology family:       some families are inherently harder
    4. Historical failure:    past failures for similar topology → lower confidence
    5. KCL residual:          high residual → OOD / low confidence
    6. Projection effort:     past high-iteration projections → lower confidence
    """

    # Known topology difficulty (0=easy, 1=hard)
    _TOPOLOGY_DIFFICULTY: dict[str, float] = {
        "radial": 0.1,
        "mesh": 0.3,
        "bridge": 0.4,
        "ladder": 0.2,
        "unknown": 0.5,
    }

    # Graph size thresholds
    _SMALL_GRAPH = 5     # nodes
    _LARGE_GRAPH = 20    # nodes

    # Resistance dynamic range thresholds
    _LOW_RANGE = 2.0     # ratio max/min
    _HIGH_RANGE = 100.0

    # KCL residual thresholds
    _LOW_KCL = 1e-6
    _HIGH_KCL = 1e-2

    def __init__(
        self,
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        """history: list of past execution dicts with keys:
        topology_family, projection_iterations, failure_type, kcl_residual
        """
        self._history = history or []

    def estimate(
        self,
        task: RuntimeTask,
        *,
        graph_size: int = 0,
        resistance_range: float = 1.0,
        topology_family: str = "unknown",
        kcl_residual: float = 0.0,
    ) -> ConfidenceEstimate:
        """Compute deterministic confidence estimate."""
        score = 1.0

        # Factor 1: Graph size (larger → less confident)
        if graph_size > 0:
            size_factor = self._graph_size_factor(graph_size)
            score *= size_factor

        # Factor 2: Resistance dynamic range (higher → less confident)
        range_factor = self._resistance_range_factor(resistance_range)
        score *= range_factor

        # Factor 3: Topology difficulty
        topo_diff = self._TOPOLOGY_DIFFICULTY.get(topology_family, 0.5)
        score *= (1.0 - 0.5 * topo_diff)

        # Factor 4: Historical failure rate for this topology
        failure_rate = self._historical_failure_rate(topology_family)
        score *= (1.0 - 0.4 * failure_rate)

        # Factor 5: KCL residual (higher → less confident, likely OOD)
        if kcl_residual > 0:
            kcl_factor = self._kcl_factor(kcl_residual)
            score *= kcl_factor

        # Factor 6: Past projection effort for this topology
        avg_proj = self._historical_avg_projection(topology_family)
        if avg_proj > 10:
            score *= 0.85

        # Clamp
        score = max(0.0, min(1.0, score))

        # Likely OOD?
        likely_ood = score < 0.4 or kcl_residual > self._HIGH_KCL

        # Estimate projection iterations
        if score > 0.7:
            est_iters = 3
        elif score > 0.4:
            est_iters = 8
        else:
            est_iters = 18

        # Adjust by historical average if available
        if avg_proj > 0:
            est_iters = max(est_iters, int(avg_proj * 0.8))

        return ConfidenceEstimate(
            confidence_score=round(score, 4),
            estimated_projection_iterations=est_iters,
            likely_ood=likely_ood,
        )

    def add_history(self, entry: dict[str, Any]) -> None:
        """Add a past execution to the history."""
        self._history.append(entry)

    # -- internal heuristics ------------------------------------------------

    def _graph_size_factor(self, n: int) -> float:
        if n <= self._SMALL_GRAPH:
            return 1.0
        if n >= self._LARGE_GRAPH:
            return 0.6
        # Linear interpolation
        t = (n - self._SMALL_GRAPH) / (self._LARGE_GRAPH - self._SMALL_GRAPH)
        return 1.0 - 0.4 * t

    def _resistance_range_factor(self, ratio: float) -> float:
        if ratio <= self._LOW_RANGE:
            return 1.0
        if ratio >= self._HIGH_RANGE:
            return 0.5
        t = (ratio - self._LOW_RANGE) / (self._HIGH_RANGE - self._LOW_RANGE)
        return 1.0 - 0.5 * t

    def _kcl_factor(self, residual: float) -> float:
        if residual <= self._LOW_KCL:
            return 1.0
        if residual >= self._HIGH_KCL:
            return 0.3
        log_residual = math.log10(max(residual, 1e-15))
        log_low = math.log10(self._LOW_KCL)
        log_high = math.log10(self._HIGH_KCL)
        t = (log_residual - log_low) / (log_high - log_low)
        return 1.0 - 0.7 * max(0.0, min(1.0, t))

    def _historical_failure_rate(self, topology_family: str) -> float:
        relevant = [e for e in self._history if e.get("topology_family") == topology_family]
        if not relevant:
            return 0.0
        failures = sum(1 for e in relevant if e.get("failure_type") is not None)
        return failures / len(relevant)

    def _historical_avg_projection(self, topology_family: str) -> float:
        relevant = [e for e in self._history if e.get("topology_family") == topology_family]
        if not relevant:
            return 0.0
        iters = [e.get("projection_iterations", 0) for e in relevant]
        return sum(iters) / len(iters)
