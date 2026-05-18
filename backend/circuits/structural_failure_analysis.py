"""Phase 6 — Structural Failure Analysis

Analyzes projection failures by computing structural graph metrics for each circuit family
and correlating them with projection effectiveness.

CLI:
    python -m backend.circuits.structural_failure_analysis
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import torch

from backend.circuits.graph_dataset import circuit_to_graph
from backend.circuits.models import Circuit
from backend.circuits.physics_projection import PhysicsProjection, ProjectionConfig
from backend.circuits.run_circuit_arena import _build_topology_families


def graph_structural_metrics(circuit: Circuit) -> Dict[str, float]:
    """Compute structural properties of the circuit graph.

    Returns a dictionary with keys:
        - diameter
        - cycle_count
        - bridge_centrality (average betweenness of bridge edges)
        - avg_node_eccentricity
        - ladder_depth (for ladder families, longest path across rails)
    """
    # Build a simple undirected graph where nodes are circuit nodes and edges are resistors
    G = nx.Graph()
    for r in circuit.resistors:
        G.add_edge(r.node_a, r.node_b)
    # Ensure ground node is present
    G.add_node(circuit.ground_node)

    # Diameter (longest shortest path) – for disconnected graph, use infinite
    if nx.is_connected(G):
        diameter = nx.diameter(G)
    else:
        # Use max finite component diameter
        diameter = max(nx.diameter(G.subgraph(c).copy()) for c in nx.connected_components(G))

    # Cycle count: cyclomatic number = E - V + components
    E = G.number_of_edges()
    V = G.number_of_nodes()
    components = nx.number_connected_components(G)
    cycle_count = E - V + components

    # Bridge centrality: compute betweenness of bridge edges, average
    bridges = list(nx.bridges(G))
    if bridges:
        betweenness = nx.edge_betweenness_centrality(G, weight=None)
        bridge_centrality = sum(betweenness[e] for e in bridges) / len(bridges)
    else:
        bridge_centrality = 0.0

    # Node eccentricity (max distance from node to others) – average over nodes
    ecc = nx.eccentricity(G) if nx.is_connected(G) else {n: 0 for n in G.nodes()}
    avg_node_eccentricity = sum(ecc.values()) / len(ecc)

    # Ladder depth heuristic: longest path that alternates between two rails if such pattern exists
    # We approximate by the longest simple path length in the graph (NP‑hard) using approximation.
    try:
        ladder_depth = len(nx.algorithms.approximation.longest_simple_path(G)) - 1
    except Exception:
        ladder_depth = 0

    return {
        'diameter': float(diameter),
        'cycle_count': float(cycle_count),
        'bridge_centrality': float(bridge_centrality),
        'avg_node_eccentricity': float(avg_node_eccentricity),
        'ladder_depth': float(ladder_depth),
    }


def evaluate_family_projection_effectiveness(
    family_name: str,
    circuits: List[Circuit],
    proj_cfg: ProjectionConfig,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Run the arena projection on the family and return aggregate metrics.

    Returns a tuple of (aggregate_projection_metrics, per_circuit_structural_data).
    """
    projector = PhysicsProjection(proj_cfg)
    raw_mae_list = []
    corrected_mae_list = []
    structural_data = []
    for circuit in circuits:
        sol = solve_dc_circuit(circuit)
        graph = circuit_to_graph(circuit, sol)
        if graph.target_voltages.numel() == 0:
            continue
        v_oracle = graph.target_voltages
        # surrogate prediction (perturb with scale 1.0)
        noise = torch.randn_like(v_oracle) * 1.0
        # zero out voltage source nodes
        node_idx = {n: i for i, n in enumerate(graph.node_names)}
        for vs in circuit.voltage_sources:
            for name in (vs.positive, vs.negative):
                if name in node_idx:
                    noise[node_idx[name]] = 0.0
        v_pred = v_oracle + noise
        raw_mae = (v_pred - v_oracle).abs().mean().item()
        v_corr = projector.project(graph, circuit, v_pred)
        corr_mae = (v_corr - v_oracle).abs().mean().item()
        raw_mae_list.append(raw_mae)
        corrected_mae_list.append(corr_mae)
        structural = graph_structural_metrics(circuit)
        structural.update({'family': family_name, 'raw_mae': raw_mae, 'corr_mae': corr_mae})
        structural_data.append(structural)
    # Compute aggregate projection improvement
    improvement = [ (r - c) / r * 100 if r > 0 else 0.0 for r, c in zip(raw_mae_list, corrected_mae_list) ]
    agg = {
        'family': family_name,
        'avg_raw_mae': sum(raw_mae_list) / max(len(raw_mae_list), 1),
        'avg_corr_mae': sum(corrected_mae_list) / max(len(corrected_mae_list), 1),
        'avg_improvement_pct': sum(improvement) / max(len(improvement), 1),
    }
    return agg, structural_data


def main() -> None:
    families = _build_topology_families()
    proj_cfg = ProjectionConfig(steps=3, alpha_kcl=1.0, alpha_kvl=0.5, omega=1.0)
    overall = {}
    all_structural: List[Dict[str, float]] = []
    for name, circuits in families.items():
        agg, struct = evaluate_family_projection_effectiveness(name, circuits, proj_cfg)
        overall[name] = agg
        all_structural.extend(struct)
    # Save results JSON for further analysis
    out_path = Path('structural_failure_analysis_results.json')
    out_path.write_text(json.dumps({'family_summary': overall, 'per_circuit': all_structural}, indent=2))
    print(f'Analysis saved to {out_path}')


if __name__ == '__main__':
    main()
