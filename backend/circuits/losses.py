"""Invariant-aware losses for circuit surrogate training."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from backend.circuits.models import Circuit


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def _node_lookup(node_names: tuple[str, ...]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(node_names)}


def voltage_loss(predicted_voltages: torch.Tensor, target_voltages: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted_voltages, target_voltages)


def kcl_penalty(circuit: Circuit, node_names: tuple[str, ...], predicted_voltages: torch.Tensor) -> torch.Tensor:
    if predicted_voltages.numel() == 0:
        return _zero_like(predicted_voltages)

    index = _node_lookup(node_names)
    residuals: list[torch.Tensor] = []

    for node in node_names:
        if any(vs.positive == node or vs.negative == node for vs in circuit.voltage_sources):
            continue

        current_sum = predicted_voltages.new_zeros(())
        v_node = predicted_voltages[index[node]]
        v_ground = predicted_voltages.new_zeros(())

        for resistor in circuit.resistors:
            if resistor.node_a == node:
                v_a = v_node
                v_b = predicted_voltages[index[resistor.node_b]] if resistor.node_b in index else v_ground
                current_sum = current_sum - (v_a - v_b) / resistor.resistance_ohm
            elif resistor.node_b == node:
                v_a = predicted_voltages[index[resistor.node_a]] if resistor.node_a in index else v_ground
                v_b = v_node
                current_sum = current_sum + (v_a - v_b) / resistor.resistance_ohm

        for source in circuit.current_sources:
            if source.positive == node:
                current_sum = current_sum + source.current
            elif source.negative == node:
                current_sum = current_sum - source.current

        residuals.append(current_sum)

    if not residuals:
        return _zero_like(predicted_voltages)
    return torch.stack([residual.pow(2) for residual in residuals]).mean()


def kvl_penalty(circuit: Circuit, node_names: tuple[str, ...], predicted_voltages: torch.Tensor) -> torch.Tensor:
    if predicted_voltages.numel() == 0:
        return _zero_like(predicted_voltages)

    index = _node_lookup(node_names)
    residuals: list[torch.Tensor] = []
    for source in circuit.voltage_sources:
        v_pos = predicted_voltages[index[source.positive]] if source.positive in index else predicted_voltages.new_zeros(())
        v_neg = predicted_voltages[index[source.negative]] if source.negative in index else predicted_voltages.new_zeros(())
        residuals.append(v_pos - v_neg - source.voltage)

    if not residuals:
        return _zero_like(predicted_voltages)
    return torch.stack([residual.pow(2) for residual in residuals]).mean()


def invariant_aware_loss(
    predicted_voltages: torch.Tensor,
    target_voltages: torch.Tensor,
    circuit: Circuit,
    node_names: tuple[str, ...],
    *,
    kcl_weight: float = 0.1,
    kvl_weight: float = 0.1,
) -> torch.Tensor:
    base = voltage_loss(predicted_voltages, target_voltages)
    return base + (kcl_weight * kcl_penalty(circuit, node_names, predicted_voltages)) + (
        kvl_weight * kvl_penalty(circuit, node_names, predicted_voltages)
    )

