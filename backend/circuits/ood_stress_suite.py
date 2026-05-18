"""Deterministic OOD stress suite generating pathologically challenging circuit typologies."""

from __future__ import annotations
import math
from typing import Dict, List, Tuple
from backend.circuits.models import Circuit, Resistor, VoltageSource

def generate_dense_grid(n: int, m: int, resistance: float = 10.0, source_voltage: float = 100.0) -> Circuit:
    """Generates a dense grid/mesh circuit of size n x m.
    
    Contains a voltage source at the top-left (1,1) and ground at the bottom-right (n,m).
    This creates an extremely high cycle count which tests KCL constraint propagation.
    """
    resistors = []
    
    # Connect grid nodes: node names are "r_c" (row, col)
    for r in range(1, n + 1):
        for c in range(1, m + 1):
            curr_node = f"{r}_{c}"
            
            # Horizontal resistor to right neighbor
            if c < m:
                right_node = f"{r}_{c+1}"
                resistors.append(Resistor(f"Rh_{r}_{c}", curr_node, right_node, resistance))
                
            # Vertical resistor to bottom neighbor
            if r < n:
                bottom_node = f"{r+1}_{c}"
                resistors.append(Resistor(f"Rv_{r}_{c}", curr_node, bottom_node, resistance))
                
    # Add voltage source from input to (1, 1)
    # The ground node is canonical "0", we place it at the bottom-right (n, m)
    # So we rename node f"{n}_{m}" to "0" or just connect a resistor or wire.
    # To keep it elegant, we define f"{n}_{m}" as the ground node "0".
    # All occurrences of f"{n}_{m}" will be replaced by "0"
    adjusted_resistors = []
    for r in resistors:
        node_a = "0" if r.node_a == f"{n}_{m}" else r.node_a
        node_b = "0" if r.node_b == f"{n}_{m}" else r.node_b
        adjusted_resistors.append(Resistor(r.name, node_a, node_b, r.resistance_ohm))
        
    # Source connected between "1_1" and ground "0"
    v_sources = [VoltageSource("Vsrc", "1_1", "0", source_voltage)]
    
    return Circuit(
        name=f"stress_grid_{n}x{m}",
        resistors=tuple(adjusted_resistors),
        voltage_sources=tuple(v_sources),
        ground_node="0"
    )

def generate_ladder_network(stages: int, r_series: float = 10.0, r_shunt: float = 100.0, source_voltage: float = 100.0) -> Circuit:
    """Generates a pathologically long series‑shunt ladder network.
    
    Tests GNN capability to propagate signals across very long chain receptivity paths.
    """
    resistors = []
    
    # A ladder network has stages. Ground node is canonical "0".
    # Node names are "n1", "n2", ..., "n_stages" along the top rail.
    # Shunt resistors go from "n_i" to ground "0".
    # Series resistors go from "n_i" to "n_{i+1}".
    for i in range(1, stages + 1):
        top_node = f"n{i}"
        
        # Shunt resistor to ground
        resistors.append(Resistor(f"Rshunt_{i}", top_node, "0", r_shunt))
        
        # Series resistor to next node
        if i < stages:
            next_node = f"n{i+1}"
            resistors.append(Resistor(f"Rseries_{i}", top_node, next_node, r_series))
            
    # Source at the input of the first stage: between "n1" and ground "0"
    v_sources = [VoltageSource("Vsrc", "n1", "0", source_voltage)]
    
    return Circuit(
        name=f"stress_ladder_{stages}_stages",
        resistors=tuple(resistors),
        voltage_sources=tuple(v_sources),
        ground_node="0"
    )

def generate_cycle_dominant_loops(num_loops: int, loop_size: int = 4, resistance: float = 10.0, source_voltage: float = 100.0) -> Circuit:
    """Generates interlocking ring/loop topologies.
    
    Provides highly redundant parallel paths that challenge message-passing loops.
    """
    resistors = []
    
    # Ground node is "0".
    # We construct interlocking loops.
    # Loop 1: nodes 1_1, 1_2, ..., 1_{loop_size-1}, 0
    # Loop i: nodes i_1, i_2, ..., i_{loop_size-1}, (i-1)_1
    for loop_idx in range(1, num_loops + 1):
        prev_anchor = "0" if loop_idx == 1 else f"{loop_idx-1}_1"
        
        # Build loop nodes
        loop_nodes = [prev_anchor] + [f"{loop_idx}_{j}" for j in range(1, loop_size)]
        
        # Connect nodes in a ring/cycle
        for idx in range(len(loop_nodes)):
            na = loop_nodes[idx]
            nb = loop_nodes[(idx + 1) % len(loop_nodes)]
            resistors.append(Resistor(f"Rloop_{loop_idx}_{idx}", na, nb, resistance))
            
    # Source connected between "1_1" and ground "0"
    v_sources = [VoltageSource("Vsrc", "1_1", "0", source_voltage)]
    
    return Circuit(
        name=f"stress_loops_{num_loops}x{loop_size}",
        resistors=tuple(resistors),
        voltage_sources=tuple(v_sources),
        ground_node="0"
    )
