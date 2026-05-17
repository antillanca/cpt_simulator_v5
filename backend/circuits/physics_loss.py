"""Physics-informed surrogate loss for DC circuit training."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn

from backend.circuits.graph_dataset import CircuitGraph, GraphBatch
from backend.circuits.losses import kcl_penalty, voltage_loss
from backend.circuits.models import Circuit


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def _component_edge_count(graph: CircuitGraph) -> int:
    return int(graph.component_edge_index.size(1))


def _as_list(value: Any | None, length: int) -> list[Any | None]:
    if value is None:
        return [None] * length
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _edge_voltage_drop(voltages: torch.Tensor, component_edge_index: torch.Tensor) -> torch.Tensor:
    if component_edge_index.numel() == 0:
        return voltages.new_zeros((0,))

    src = component_edge_index[0].to(torch.long)
    dst = component_edge_index[1].to(torch.long)
    drops = voltages.new_zeros((src.numel(),))

    src_mask = src >= 0
    dst_mask = dst >= 0
    if src_mask.any():
        drops[src_mask] += voltages[src[src_mask]]
    if dst_mask.any():
        drops[dst_mask] -= voltages[dst[dst_mask]]
    return drops


def _source_current_leaving_positive(
    circuit: Circuit,
    node_names: Sequence[str],
    voltages: torch.Tensor,
    node: str,
) -> torch.Tensor:
    index = {name: idx for idx, name in enumerate(node_names)}
    current_sum = voltages.new_zeros(())

    for resistor in circuit.resistors:
        if resistor.node_a == node:
            v_a = voltages[index[resistor.node_a]]
            v_b = voltages[index[resistor.node_b]] if resistor.node_b in index else voltages.new_zeros(())
            current_sum = current_sum + (v_a - v_b) / resistor.resistance_ohm
        elif resistor.node_b == node:
            v_a = voltages[index[resistor.node_a]] if resistor.node_a in index else voltages.new_zeros(())
            v_b = voltages[index[resistor.node_b]]
            current_sum = current_sum + (v_b - v_a) / resistor.resistance_ohm

    for source in circuit.current_sources:
        if source.positive == node:
            current_sum = current_sum + source.current
        elif source.negative == node:
            current_sum = current_sum - source.current

    return -current_sum


def _power_balance_loss(circuit: Circuit, node_names: Sequence[str], voltages: torch.Tensor) -> torch.Tensor:
    if voltages.numel() == 0:
        return _zero_like(voltages)

    index = {name: idx for idx, name in enumerate(node_names)}
    source_power = voltages.new_zeros(())
    dissipated_power = voltages.new_zeros(())

    for source in circuit.voltage_sources:
        vp = voltages[index[source.positive]] if source.positive in index else voltages.new_zeros(())
        vn = voltages[index[source.negative]] if source.negative in index else voltages.new_zeros(())
        inferred = _source_current_leaving_positive(circuit, node_names, voltages, source.positive)
        source_power = source_power + (vn - vp) * inferred

    for source in circuit.current_sources:
        vp = voltages[index[source.positive]] if source.positive in index else voltages.new_zeros(())
        vn = voltages[index[source.negative]] if source.negative in index else voltages.new_zeros(())
        source_power = source_power + (vp - vn) * source.current

    for resistor in circuit.resistors:
        va = voltages[index[resistor.node_a]] if resistor.node_a in index else voltages.new_zeros(())
        vb = voltages[index[resistor.node_b]] if resistor.node_b in index else voltages.new_zeros(())
        current = (va - vb) / resistor.resistance_ohm
        dissipated_power = dissipated_power + ((va - vb) * current)

    return (source_power - dissipated_power).pow(2)


class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_kcl: float = 0.1, lambda_kvl: float = 0.1, lambda_power: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_kcl = float(lambda_kcl)
        self.lambda_kvl = float(lambda_kvl)
        self.lambda_power = float(lambda_power)

    def forward(
        self,
        pred_voltages: torch.Tensor,
        target_voltages: torch.Tensor,
        graph_batch: CircuitGraph | GraphBatch,
        circuit: Circuit | Sequence[Circuit] | None = None,
    ) -> torch.Tensor:
        loss_v = self.compute_voltage_loss(pred_voltages, target_voltages)
        loss_kcl = self.compute_kcl_loss(pred_voltages, graph_batch, circuit=circuit)
        loss_kvl = self.compute_kvl_loss(pred_voltages, graph_batch)
        loss_power = self.compute_power_loss(pred_voltages, graph_batch, circuit=circuit)
        return loss_v + self.lambda_kcl * loss_kcl + self.lambda_kvl * loss_kvl + self.lambda_power * loss_power

    def compute_voltage_loss(self, pred_voltages: torch.Tensor, target_voltages: torch.Tensor) -> torch.Tensor:
        return self.mse(pred_voltages, target_voltages)

    def _single_kcl_loss(
        self,
        circuit: Circuit | None,
        node_names: Sequence[str],
        voltages: torch.Tensor,
    ) -> torch.Tensor:
        if circuit is None or voltages.numel() == 0:
            return _zero_like(voltages)
        return kcl_penalty(circuit, tuple(node_names), voltages)

    def compute_kcl_loss(
        self,
        pred_voltages: torch.Tensor,
        graph_batch: CircuitGraph | GraphBatch,
        *,
        circuit: Circuit | Sequence[Circuit] | None = None,
    ) -> torch.Tensor:
        if isinstance(graph_batch, CircuitGraph):
            circuits = _as_list(circuit, 1)
            return self._single_kcl_loss(circuits[0], graph_batch.node_names, pred_voltages)

        if pred_voltages.numel() == 0 or graph_batch.node_features.numel() == 0:
            return _zero_like(pred_voltages)

        circuits_list = _as_list(circuit, len(graph_batch.num_nodes))
        losses: list[torch.Tensor] = []
        for i, nnodes in enumerate(graph_batch.num_nodes):
            if nnodes == 0:
                continue
            pred = pred_voltages[i, :nnodes]
            node_names = tuple(circuits_list[i].all_nodes) if circuits_list[i] is not None else tuple(f"n{j}" for j in range(nnodes))
            # If caller provides a circuit list we use the circuit-aware KCL.
            losses.append(self._single_kcl_loss(circuits_list[i], node_names, pred))
        if not losses:
            return _zero_like(pred_voltages)
        return torch.stack(losses).mean()

    def _single_kvl_loss(self, voltages: torch.Tensor, graph: CircuitGraph) -> torch.Tensor:
        if graph.cycle_matrix.numel() == 0 or graph.cycle_matrix.size(0) == 0:
            return _zero_like(voltages)
        component_edges = graph.component_edge_index
        edge_drops = _edge_voltage_drop(voltages, component_edges)
        if edge_drops.numel() == 0:
            return _zero_like(voltages)
        cycle_sums = torch.matmul(graph.cycle_matrix.to(edge_drops.dtype), edge_drops)
        return cycle_sums.pow(2).mean()

    def compute_kvl_loss(
        self,
        pred_voltages: torch.Tensor,
        graph_batch: CircuitGraph | GraphBatch,
    ) -> torch.Tensor:
        if isinstance(graph_batch, CircuitGraph):
            return self._single_kvl_loss(pred_voltages, graph_batch)

        if pred_voltages.numel() == 0 or graph_batch.node_features.numel() == 0:
            return _zero_like(pred_voltages)

        losses: list[torch.Tensor] = []
        for i, nnodes in enumerate(graph_batch.num_nodes):
            ncomp = graph_batch.num_component_edges[i]
            ncyc = graph_batch.num_cycles[i]
            if nnodes == 0:
                continue
            graph = CircuitGraph(
                node_features=graph_batch.node_features[i, :nnodes],
                edge_index=graph_batch.edge_index[i],
                edge_features=graph_batch.edge_features[i],
                target_voltages=graph_batch.target_voltages[i, :nnodes],
                node_names=tuple(f"n{j}" for j in range(nnodes)),
                fingerprint="",
                component_edge_index=graph_batch.component_edge_index[i, :, :ncomp],
                cycle_matrix=graph_batch.cycle_matrix[i, :ncyc, :ncomp],
            )
            losses.append(self._single_kvl_loss(pred_voltages[i, :nnodes], graph))
        if not losses:
            return _zero_like(pred_voltages)
        return torch.stack(losses).mean()

    def _single_power_loss(
        self,
        circuit: Circuit | None,
        node_names: Sequence[str],
        voltages: torch.Tensor,
    ) -> torch.Tensor:
        if circuit is None or voltages.numel() == 0:
            return _zero_like(voltages)
        return _power_balance_loss(circuit, node_names, voltages)

    def compute_power_loss(
        self,
        pred_voltages: torch.Tensor,
        graph_batch: CircuitGraph | GraphBatch,
        *,
        circuit: Circuit | Sequence[Circuit] | None = None,
    ) -> torch.Tensor:
        if isinstance(graph_batch, CircuitGraph):
            circuits = _as_list(circuit, 1)
            return self._single_power_loss(circuits[0], graph_batch.node_names, pred_voltages)

        if pred_voltages.numel() == 0 or graph_batch.node_features.numel() == 0:
            return _zero_like(pred_voltages)

        circuits_list = _as_list(circuit, len(graph_batch.num_nodes))
        losses: list[torch.Tensor] = []
        for i, nnodes in enumerate(graph_batch.num_nodes):
            if nnodes == 0:
                continue
            pred = pred_voltages[i, :nnodes]
            node_names = tuple(circuits_list[i].all_nodes) if circuits_list[i] is not None else tuple(f"n{j}" for j in range(nnodes))
            losses.append(self._single_power_loss(circuits_list[i], node_names, pred))
        if not losses:
            return _zero_like(pred_voltages)
        return torch.stack(losses).mean()
