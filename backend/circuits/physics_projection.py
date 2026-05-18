"""Hybrid Physics Correction Layer for CPT v2.9F.

Deterministic iterative residual projection that transforms the GNN surrogate
from a pure voltage regressor into a physics-consistent coarse solver.

Pipeline:
 1. GNN predicts V_pred (initial solution)
 2. PhysicsProjection.project() iteratively reduces residuals:
    - KCL residual: current imbalance at each node
    - KVL residual: voltage-drop mismatch around cycles
    - Power residual: global power conservation mismatch
 3. VirtualNodeProjection adds global residual coupling:
    - Aggregates residual drift globally (R_global = mean)
    - Redistributes correction relative to global mean
    - Transforms chain-like propagation into star-like
    - Reduces effective spectral radius for radial topologies

All operations are:
 - deterministic (no randomness, fixed order)
 - torch-only (differentiable for future training)
 - lightweight (no extra learned parameters)
 - NaN/Inf safe with clamping

v2.9F specification:
 V_corrected = V - alpha * residual

v2.9F Phase 3 — Virtual Node:
 R_global = mean(residuals)
 V_i = V_i - beta * (residual_i - R_global)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from backend.circuits.graph_dataset import CircuitGraph
from backend.circuits.models import Circuit


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectionConfig:
    """Deterministic configuration for the physics projection layer."""

    alpha_kcl: float = 0.1          # KCL correction step size
    alpha_kvl: float = 0.05         # KVL correction step size
    alpha_power: float = 0.05       # Power correction step size
    steps: int = 3                  # Number of iterative refinement steps
    clamp_value: float = 1e4        # Max absolute voltage to prevent divergence
    min_resistance: float = 1e-6    # Floor for resistance to avoid division by zero
    kcl_enabled: bool = True
    kvl_enabled: bool = True
    power_enabled: bool = False     # Power correction is global & less stable; off by default
    # v2.9F Phase 3: SOR over-relaxation (kept for backwards compatibility)
    omega: float = 1.0
    # v2.9F Phase 3A: Virtual Node Projection
    virtual_node_enabled: bool = True
    virtual_conductance: float = 0.1   # G_virtual — conductance of virtual edges
    blend_factor: float = 0.5          # beta — how much global residual correction to apply
    virtual_momentum: float = 0.0      # momentum for exponential residual memory (0=disabled)


# ---------------------------------------------------------------------------
# Virtual Node Projection
# ---------------------------------------------------------------------------

class VirtualNodeProjection:
    """Global residual communication hub for physics projection.

    The virtual node:
    - connects to ALL real nodes (conceptually)
    - aggregates residual drift globally: R_global = mean(residuals)
    - redistributes correction relative to global mean:
      V_i = V_i - beta * (residual_i - R_global)

    This creates:
    - global coupling between all nodes
    - long-range information flow (star-like propagation)
    - reduced effective spectral radius for chain topologies

    Optional: exponential residual memory for drift accumulation awareness.
    global_memory = momentum * global_memory + R_global_current

    All operations are O(N), deterministic, and torch-only.
    """

    def __init__(
        self,
        enabled: bool = True,
        virtual_conductance: float = 0.1,
        blend_factor: float = 0.5,
        momentum: float = 0.0,
    ) -> None:
        self.enabled = enabled
        self.virtual_conductance = virtual_conductance
        self.blend_factor = blend_factor
        self.momentum = momentum
        self.global_memory: Optional[torch.Tensor] = None

    def reset_memory(self) -> None:
        """Reset accumulated global memory."""
        self.global_memory = None

    def apply(
        self,
        voltages: torch.Tensor,
        residuals: torch.Tensor,
        circuit: Circuit,
        graph: CircuitGraph,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply virtual node correction to voltages.

        Args:
            voltages: Current voltage estimates (num_nodes,)
            residuals: KCL residuals (already scaled by 1/G_ii)
            circuit: Circuit domain model
            graph: CircuitGraph with node info

        Returns:
            (corrected_voltages, metrics_dict)
        """
        if not self.enabled or voltages.numel() == 0:
            return voltages, {"r_global": 0.0, "r_memory": 0.0, "virtual_delta": 0.0}

        # Identify fixed nodes (voltage source nodes + ground)
        node_idx: Dict[str, int] = {name: i for i, name in enumerate(graph.node_names)}
        vs_nodes: set = set()
        for vs in circuit.voltage_sources:
            vs_nodes.add(vs.positive)
            vs_nodes.add(vs.negative)

        # Compute global residual mean (excluding fixed nodes)
        free_mask = torch.ones(voltages.size(0), dtype=torch.bool)
        for name in vs_nodes:
            if name in node_idx and name != circuit.ground_node:
                free_mask[node_idx[name]] = False

        n_free = free_mask.sum().item()
        if n_free == 0:
            return voltages, {"r_global": 0.0, "r_memory": 0.0, "virtual_delta": 0.0}

        free_residuals = residuals[free_mask]
        r_global = free_residuals.mean()

        # Update exponential memory
        if self.momentum > 0.0:
            if self.global_memory is None:
                self.global_memory = r_global.clone()
            else:
                self.global_memory = self.momentum * self.global_memory + (1.0 - self.momentum) * r_global
            effective_r_global = self.global_memory
        else:
            effective_r_global = r_global

        # Apply virtual node correction:
        # V_i = V_i - beta * (residual_i - R_global)
        # This subtracts the global drift component, leaving only the
        # node-specific deviation from the mean, then corrects that.
        correction = self.blend_factor * (residuals - effective_r_global)

        # Zero out correction for fixed nodes
        correction = correction * free_mask.float()

        v_corrected = voltages - correction

        metrics = {
            "r_global": r_global.item(),
            "r_memory": effective_r_global.item() if effective_r_global.numel() > 0 else 0.0,
            "virtual_delta": correction.abs().max().item(),
        }

        return v_corrected, metrics


# ---------------------------------------------------------------------------
# Residual computation helpers
# ---------------------------------------------------------------------------

def _node_conductance_diagonal(
    graph: CircuitGraph,
    circuit: Circuit,
) -> torch.Tensor:
    """Compute diagonal of the conductance matrix G_ii for each node.

    G_ii = sum of conductances connected to node i.
    This is the diagonal of the Jacobian dKCL/dV, used for Newton-like
    scaling of the KCL correction (Jacobi preconditioning).

    Returns: (num_nodes,) tensor of G_ii values.
    """
    node_names = graph.node_names
    n = len(node_names)
    if n == 0:
        return torch.zeros((0,))

    g_diag = torch.zeros((n,), dtype=torch.float32)
    node_idx: Dict[str, int] = {name: i for i, name in enumerate(node_names)}

    for resistor in circuit.resistors:
        r = max(resistor.resistance_ohm, ProjectionConfig.min_resistance)
        g = 1.0 / r
        va_idx = node_idx.get(resistor.node_a)
        vb_idx = node_idx.get(resistor.node_b)
        if va_idx is not None:
            g_diag[va_idx] += g
        if vb_idx is not None:
            g_diag[vb_idx] += g

    # Zero out for voltage-source nodes (they are fixed)
    vs_nodes: set = set()
    for vs in circuit.voltage_sources:
        vs_nodes.add(vs.positive)
        vs_nodes.add(vs.negative)
    for node in vs_nodes:
        if node in node_idx and node != circuit.ground_node:
            g_diag[node_idx[node]] = 1.0  # avoid div-by-zero; residual is 0 anyway

    return g_diag.clamp(min=1e-12)  # safety floor


def _node_kcl_residual(
    voltages: torch.Tensor,
    graph: CircuitGraph,
    circuit: Circuit,
    scale_by_conductance: bool = True,
) -> torch.Tensor:
    """Compute KCL residual at each node: net current leaving.

    Convention: residual_i > 0 means net current leaves node i
    (KCL violation: should be 0 for interior nodes).

    When scale_by_conductance=True (default), the current residual is
    divided by G_ii (diagonal of conductance matrix), yielding a
    Newton-like voltage correction. This makes alpha ~ O(1) meaningful
    regardless of circuit resistance scale.

    Correction: V_i -= alpha * residual_i
    - If residual > 0 (too much leaving), voltage must decrease
    - If residual < 0 (too much entering), voltage must increase

    Nodes attached to voltage sources get residual=0.
    """
    node_names = graph.node_names
    n = len(node_names)
    if n == 0 or voltages.numel() == 0:
        return voltages.new_zeros(voltages.shape)

    residuals = voltages.new_zeros(voltages.shape)

    # Build node index
    node_idx: Dict[str, int] = {name: i for i, name in enumerate(node_names)}

    # Identify nodes attached to voltage sources (skip those)
    vs_nodes: set = set()
    for vs in circuit.voltage_sources:
        vs_nodes.add(vs.positive)
        vs_nodes.add(vs.negative)

    # Compute current balance per node
    for resistor in circuit.resistors:
        r = max(resistor.resistance_ohm, ProjectionConfig.min_resistance)
        va_idx = node_idx.get(resistor.node_a)
        vb_idx = node_idx.get(resistor.node_b)

        if va_idx is not None and vb_idx is not None:
            current = (voltages[va_idx] - voltages[vb_idx]) / r
            residuals[va_idx] += current
            residuals[vb_idx] -= current
        elif va_idx is not None:
            current = (voltages[va_idx] - 0.0) / r
            residuals[va_idx] += current
        elif vb_idx is not None:
            current = (voltages[vb_idx] - 0.0) / r
            residuals[vb_idx] += current

    # Current sources
    for cs in circuit.current_sources:
        pos_idx = node_idx.get(cs.positive)
        neg_idx = node_idx.get(cs.negative)
        if pos_idx is not None:
            residuals[pos_idx] -= cs.current  # current enters this node
        if neg_idx is not None:
            residuals[neg_idx] += cs.current  # current leaves this node

    # Zero out residuals at voltage-source nodes (unknown current)
    for node in vs_nodes:
        if node in node_idx and node != circuit.ground_node:
            residuals[node_idx[node]] = 0.0

    # Scale by 1/G_ii for Newton-like correction (voltage-space residual)
    if scale_by_conductance:
        g_diag = _node_conductance_diagonal(graph, circuit).to(voltages.dtype)
        residuals = residuals / g_diag

    return residuals


def _enforce_boundary_conditions(
    voltages: torch.Tensor,
    circuit: Circuit,
    graph: CircuitGraph,
) -> torch.Tensor:
    """Enforce fixed voltage source values to prevent zero-initialization stall."""
    if voltages.numel() == 0:
        return voltages
    node_idx = {name: i for i, name in enumerate(graph.node_names)}
    for vs in circuit.voltage_sources:
        pos_idx = node_idx.get(vs.positive)
        neg_idx = node_idx.get(vs.negative)
        if vs.negative == circuit.ground_node and pos_idx is not None:
            voltages[pos_idx] = float(vs.voltage)
        elif vs.positive == circuit.ground_node and neg_idx is not None:
            voltages[neg_idx] = float(-vs.voltage)
        elif pos_idx is not None and neg_idx is not None:
            voltages[pos_idx] = voltages[neg_idx] + float(vs.voltage)
    return voltages


def _cycle_kvl_residual(
    voltages: torch.Tensor,
    graph: CircuitGraph,
) -> torch.Tensor:
    """Compute KVL residual for each fundamental cycle.

    For cycle c:
      residual_c = sum_e (sign_ce * V_drop_e)

    Returns: (num_cycles,) tensor of cycle residuals.
    If no cycles exist, returns empty tensor.
    """
    if graph.cycle_matrix.numel() == 0 or graph.cycle_matrix.size(0) == 0:
        return voltages.new_zeros((0,))

    component_edges = graph.component_edge_index
    if component_edges.numel() == 0:
        return voltages.new_zeros((0,))

    src = component_edges[0].to(torch.long)
    dst = component_edges[1].to(torch.long)
    num_edges = src.numel()
    edge_drops = voltages.new_zeros((num_edges,))

    src_mask = src >= 0
    dst_mask = dst >= 0
    if src_mask.any():
        edge_drops[src_mask] += voltages[src[src_mask]]
    if dst_mask.any():
        edge_drops[dst_mask] -= voltages[dst[dst_mask]]

    cycle_residuals = torch.matmul(
        graph.cycle_matrix.to(edge_drops.dtype), edge_drops
    )
    return cycle_residuals


def _power_residual(
    voltages: torch.Tensor,
    graph: CircuitGraph,
    circuit: Circuit,
) -> torch.Tensor:
    """Compute global power conservation residual.

    power_residual = P_supplied - P_dissipated

    Returns scalar tensor.
    """
    if voltages.numel() == 0:
        return voltages.new_zeros(())

    node_idx: Dict[str, int] = {name: i for i, name in enumerate(graph.node_names)}

    # Power supplied by voltage sources
    source_power = voltages.new_zeros(())
    for vs in circuit.voltage_sources:
        vp = voltages[node_idx[vs.positive]] if vs.positive in node_idx else voltages.new_zeros(())
        vn = voltages[node_idx[vs.negative]] if vs.negative in node_idx else voltages.new_zeros(())
        current_leaving = voltages.new_zeros(())
        for r in circuit.resistors:
            if r.node_a == vs.positive:
                va = voltages[node_idx[r.node_a]] if r.node_a in node_idx else voltages.new_zeros(())
                vb = voltages[node_idx[r.node_b]] if r.node_b in node_idx else voltages.new_zeros(())
                current_leaving = current_leaving + (va - vb) / max(r.resistance_ohm, ProjectionConfig.min_resistance)
            elif r.node_b == vs.positive:
                va = voltages[node_idx[r.node_a]] if r.node_a in node_idx else voltages.new_zeros(())
                vb = voltages[node_idx[r.node_b]] if r.node_b in node_idx else voltages.new_zeros(())
                current_leaving = current_leaving + (vb - va) / max(r.resistance_ohm, ProjectionConfig.min_resistance)
        for cs in circuit.current_sources:
            if cs.positive == vs.positive:
                current_leaving = current_leaving + cs.current
            elif cs.negative == vs.positive:
                current_leaving = current_leaving - cs.current
        vs_current = -current_leaving
        source_power = source_power + (vn - vp) * vs_current

    # Power supplied by current sources
    for cs in circuit.current_sources:
        vp = voltages[node_idx[cs.positive]] if cs.positive in node_idx else voltages.new_zeros(())
        vn = voltages[node_idx[cs.negative]] if cs.negative in node_idx else voltages.new_zeros(())
        source_power = source_power + (vp - vn) * cs.current

    # Power dissipated by resistors
    dissipated_power = voltages.new_zeros(())
    for r in circuit.resistors:
        va = voltages[node_idx[r.node_a]] if r.node_a in node_idx else voltages.new_zeros(())
        vb = voltages[node_idx[r.node_b]] if r.node_b in node_idx else voltages.new_zeros(())
        current = (va - vb) / max(r.resistance_ohm, ProjectionConfig.min_resistance)
        dissipated_power = dissipated_power + (va - vb) * current

    return source_power - dissipated_power


# ---------------------------------------------------------------------------
# Cycle-based KVL voltage correction
# ---------------------------------------------------------------------------

def _apply_kvl_correction(
    voltages: torch.Tensor,
    graph: CircuitGraph,
    cycle_residuals: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Distribute KVL correction across nodes participating in each cycle."""
    if cycle_residuals.numel() == 0 or graph.cycle_matrix.numel() == 0:
        return voltages

    component_edges = graph.component_edge_index
    if component_edges.numel() == 0:
        return voltages

    src = component_edges[0].to(torch.long)
    dst = component_edges[1].to(torch.long)
    n_nodes = voltages.size(0)

    cycle_count = voltages.new_zeros((n_nodes,))
    node_correction = voltages.new_zeros((n_nodes,))

    for c in range(cycle_residuals.size(0)):
        row = graph.cycle_matrix[c]
        nz = (row != 0).nonzero(as_tuple=False).squeeze(1)
        if nz.numel() == 0:
            continue

        residual_c = cycle_residuals[c]

        participating_nodes: set = set()
        for eid in nz.tolist():
            s = src[eid].item()
            d = dst[eid].item()
            if s >= 0:
                participating_nodes.add(s)
            if d >= 0:
                participating_nodes.add(d)

        if not participating_nodes:
            continue

        correction_per_node = -alpha * residual_c / len(participating_nodes)
        for ni in participating_nodes:
            node_correction[ni] += correction_per_node
            cycle_count[ni] += 1

    mask = cycle_count > 0
    if mask.any():
        node_correction[mask] = node_correction[mask] / cycle_count[mask]

    return voltages + node_correction


# ---------------------------------------------------------------------------
# Power correction
# ---------------------------------------------------------------------------

def _apply_power_correction(
    voltages: torch.Tensor,
    power_residual: torch.Tensor,
    alpha: float,
    circuit: Circuit,
    graph: CircuitGraph,
) -> torch.Tensor:
    """Apply a global power correction by shifting all non-fixed voltages."""
    if abs(power_residual.item()) < 1e-12:
        return voltages

    node_idx: Dict[str, int] = {name: i for i, name in enumerate(graph.node_names)}
    n = voltages.size(0)

    v_rms = voltages.pow(2).mean().sqrt().clamp(min=1e-6)
    scale_correction = -alpha * power_residual / (n * v_rms.pow(2).clamp(min=1e-6))

    vs_nodes: set = set()
    for vs in circuit.voltage_sources:
        vs_nodes.add(vs.positive)
        vs_nodes.add(vs.negative)

    correction = voltages.new_zeros(voltages.shape)
    for i, name in enumerate(graph.node_names):
        if name not in vs_nodes and name != circuit.ground_node:
            correction[i] = scale_correction * voltages[i]

    return voltages + correction


# ---------------------------------------------------------------------------
# Main PhysicsProjection class
# ---------------------------------------------------------------------------

class PhysicsProjection:
    """Deterministic iterative physics correction layer with virtual node.

    Takes GNN-predicted voltages and iteratively reduces physics residuals:
      V_corrected = V - alpha * residual

    With virtual node enabled (default):
      R_global = mean(residuals)
      V_i = V_i - beta * (residual_i - R_global)

    Usage:
      projector = PhysicsProjection(config)
      V_corrected = projector.project(graph, circuit, V_pred)
    """

    def __init__(self, config: ProjectionConfig | None = None) -> None:
        self.config = config or ProjectionConfig()
        cfg = self.config
        self.virtual_node = VirtualNodeProjection(
            enabled=cfg.virtual_node_enabled,
            virtual_conductance=cfg.virtual_conductance,
            blend_factor=cfg.blend_factor,
            momentum=cfg.virtual_momentum,
        )

    def project(
        self,
        graph: CircuitGraph,
        circuit: Circuit,
        voltages: torch.Tensor,
    ) -> torch.Tensor:
        """Apply iterative physics projection with virtual node.

        Args:
            graph: CircuitGraph with cycle_matrix and component_edge_index.
            circuit: Circuit domain model with component definitions.
            voltages: Predicted voltages (num_nodes,) tensor.

        Returns:
            Corrected voltages (num_nodes,) tensor, same shape.
        """
        v = voltages.clone()
        cfg = self.config

        v = _enforce_boundary_conditions(v, circuit, graph)

        # Reset virtual node memory for fresh projection
        self.virtual_node.reset_memory()

        for step in range(cfg.steps):
            # --- KCL correction (with SOR over-relaxation) ---
            if cfg.kcl_enabled:
                kcl_res = _node_kcl_residual(v, graph, circuit)
                # Standard Jacobi/SOR step
                v = v - cfg.omega * cfg.alpha_kcl * kcl_res

                # --- Virtual Node correction (after KCL, before KVL) ---
                if cfg.virtual_node_enabled:
                    v, _ = self.virtual_node.apply(v, kcl_res, circuit, graph)

            # --- KVL correction ---
            if cfg.kvl_enabled and graph.cycle_matrix.numel() > 0:
                kvl_res = _cycle_kvl_residual(v, graph)
                v = _apply_kvl_correction(v, graph, kvl_res, cfg.alpha_kvl)

            # --- Power correction ---
            if cfg.power_enabled:
                p_res = _power_residual(v, graph, circuit)
                v = _apply_power_correction(v, p_res, cfg.alpha_power, circuit, graph)

            # --- Safety clamping ---
            v = torch.clamp(v, min=-cfg.clamp_value, max=cfg.clamp_value)

            # --- NaN safety ---
            if torch.isnan(v).any() or torch.isinf(v).any():
                v = torch.where(
                    torch.isfinite(v),
                    v,
                    voltages,
                )
                break
                
            v = _enforce_boundary_conditions(v, circuit, graph)

        return v

    def project_step_metrics(
        self,
        graph: CircuitGraph,
        circuit: Circuit,
        voltages: torch.Tensor,
    ) -> List[Dict[str, float]]:
        """Project and collect per-step metrics for analysis.

        Returns list of dicts, one per step, with:
        - kcl_max_residual
        - kvl_max_residual
        - power_residual
        - voltage_delta (max change from previous step)
        - r_global (virtual node global residual mean)
        - r_memory (virtual node memory value)
        - virtual_delta (max virtual node correction magnitude)
        """
        v = voltages.clone()
        cfg = self.config
        metrics: List[Dict[str, float]] = []

        v = _enforce_boundary_conditions(v, circuit, graph)

        # Reset virtual node memory
        self.virtual_node.reset_memory()

        for step in range(cfg.steps):
            v_prev = v.clone()

            if cfg.kcl_enabled:
                kcl_res = _node_kcl_residual(v, graph, circuit)
                v = v - cfg.omega * cfg.alpha_kcl * kcl_res
            else:
                kcl_res = v.new_zeros(v.shape)

            # Virtual node correction
            vn_metrics = {"r_global": 0.0, "r_memory": 0.0, "virtual_delta": 0.0}
            if cfg.virtual_node_enabled:
                v, vn_metrics = self.virtual_node.apply(v, kcl_res, circuit, graph)

            if cfg.kvl_enabled and graph.cycle_matrix.numel() > 0:
                kvl_res = _cycle_kvl_residual(v, graph)
                v = _apply_kvl_correction(v, graph, kvl_res, cfg.alpha_kvl)
            else:
                kvl_res = v.new_zeros((0,))

            p_res = _power_residual(v, graph, circuit) if cfg.power_enabled else v.new_zeros(())

            v = torch.clamp(v, min=-cfg.clamp_value, max=cfg.clamp_value)
            if torch.isnan(v).any() or torch.isinf(v).any():
                v = torch.where(torch.isfinite(v), v, voltages)
                break
                
            v = _enforce_boundary_conditions(v, circuit, graph)

            delta = (v - v_prev).abs().max().item()
            metrics.append({
                "step": step,
                "kcl_max_residual": kcl_res.abs().max().item() if kcl_res.numel() > 0 else 0.0,
                "kvl_max_residual": kvl_res.abs().max().item() if kvl_res.numel() > 0 else 0.0,
                "power_residual": abs(p_res.item()) if p_res.numel() > 0 else 0.0,
                "voltage_delta": delta,
                **vn_metrics,
            })

        return metrics
