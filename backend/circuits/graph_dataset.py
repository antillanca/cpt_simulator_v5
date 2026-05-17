"""Circuit-to-graph conversion for GNN surrogate training.

Converts Circuit + CircuitSolution into deterministic graph tensors:
- Nodes: electrical nodes (ground excluded, voltage=0 implicit)
- Edges: one per component
- Features: normalized per-circuit
- Targets: oracle node voltages

All orderings are deterministic (sorted by name/key).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from backend.circuits.models import Circuit, CircuitSolution

# Feature dimensions
NODE_DIM = 8
EDGE_DIM = 4


@dataclass(frozen=True)
class CircuitGraph:
    """Graph representation of a solved circuit."""

    node_features: torch.Tensor  # (num_nodes, NODE_DIM)
    edge_index: torch.Tensor  # (2, num_edges)
    edge_features: torch.Tensor  # (num_edges, EDGE_DIM)
    target_voltages: torch.Tensor  # (num_nodes,)
    node_names: Tuple[str, ...]
    fingerprint: str
    component_edge_index: torch.Tensor = field(default_factory=lambda: torch.zeros(2, 0, dtype=torch.long))
    cycle_matrix: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 0, dtype=torch.float32))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_features": self.node_features.tolist(),
            "edge_index": self.edge_index.tolist(),
            "edge_features": self.edge_features.tolist(),
            "component_edge_index": self.component_edge_index.tolist(),
            "cycle_matrix": self.cycle_matrix.tolist(),
            "target_voltages": self.target_voltages.tolist(),
            "node_names": list(self.node_names),
            "fingerprint": self.fingerprint,
        }


def _compute_node_features(
    circuit: Circuit,
    node_names: Tuple[str, ...],
    max_degree: int,
) -> torch.Tensor:
    """Compute 8-dim node features for each non-ground node.

    Features:
    0: degree / max_degree
    1: is_connected_to_voltage_source (0/1)
    2: is_connected_to_current_source (0/1)
    3: is_connected_to_ground (0/1)
    4: log(sum of incident conductances + 1) / log(max_cond + 1)
    5: number of incident resistors / max(1, max_degree)
    6: number of incident voltage sources / max(1, max_degree)
    7: number of incident current sources / max(1, max_degree)
    """
    n = len(node_names)
    node_set = set(node_names)
    features = torch.zeros(n, NODE_DIM, dtype=torch.float32)

    degree = [0] * n
    vs_connected = [0] * n
    cs_connected = [0] * n
    gnd_connected = [0] * n
    cond_sum = [0.0] * n
    r_count = [0] * n
    vs_count = [0] * n
    cs_count = [0] * n

    for idx, node in enumerate(node_names):
        for r in circuit.resistors:
            if r.node_a == node or r.node_b == node:
                degree[idx] += 1
                cond_sum[idx] += 1.0 / r.resistance_ohm
                r_count[idx] += 1
            if r.node_a == node and r.node_b == "0":
                gnd_connected[idx] = 1
            if r.node_b == node and r.node_a == "0":
                gnd_connected[idx] = 1

        for vs in circuit.voltage_sources:
            if vs.positive == node or vs.negative == node:
                degree[idx] += 1
                vs_connected[idx] = 1
                vs_count[idx] += 1
            if vs.positive == node and vs.negative == "0":
                gnd_connected[idx] = 1
            if vs.negative == node and vs.positive == "0":
                gnd_connected[idx] = 1

        for cs in circuit.current_sources:
            if cs.positive == node or cs.negative == node:
                degree[idx] += 1
                cs_connected[idx] = 1
                cs_count[idx] += 1
            if cs.positive == node and cs.negative == "0":
                gnd_connected[idx] = 1
            if cs.negative == node and cs.positive == "0":
                gnd_connected[idx] = 1

    norm_deg = max(max_degree, 1)
    max_cond = max((c for c in cond_sum), default=1.0)
    log_max_cond = math.log1p(max_cond) if max_cond > 0 else 1.0

    for idx in range(n):
        features[idx, 0] = degree[idx] / norm_deg
        features[idx, 1] = float(vs_connected[idx])
        features[idx, 2] = float(cs_connected[idx])
        features[idx, 3] = float(gnd_connected[idx])
        features[idx, 4] = math.log1p(cond_sum[idx]) / log_max_cond if log_max_cond > 0 else 0.0
        features[idx, 5] = r_count[idx] / norm_deg
        features[idx, 6] = vs_count[idx] / norm_deg
        features[idx, 7] = cs_count[idx] / norm_deg

    return features


def _compute_edge_features(
    circuit: Circuit,
    edges: List[Tuple[int, int, str, str]],
    max_resistance: float,
    max_voltage: float,
    max_current: float,
) -> torch.Tensor:
    """Compute 4-dim edge features for each component.

    Features:
    0: component type (0=resistor, 1=voltage_source, 2=current_source)
    1: resistance / max_resistance (0 if not resistor)
    2: voltage / max_voltage (0 if not voltage source)
    3: current / max_current (0 if not current source)
    """
    m = len(edges)
    features = torch.zeros(m, EDGE_DIM, dtype=torch.float32)

    # Build lookup by component name
    r_lookup = {r.name: r for r in circuit.resistors}
    vs_lookup = {vs.name: vs for vs in circuit.voltage_sources}
    cs_lookup = {cs.name: cs for cs in circuit.current_sources}

    for idx, (_, _, comp_name, comp_type) in enumerate(edges):
        if comp_type == "resistor":
            r = r_lookup[comp_name]
            features[idx, 0] = 0.0
            features[idx, 1] = r.resistance_ohm / max(max_resistance, 1e-12)
        elif comp_type == "voltage_source":
            vs = vs_lookup[comp_name]
            features[idx, 0] = 1.0
            features[idx, 2] = vs.voltage / max(max_voltage, 1e-12)
        elif comp_type == "current_source":
            cs = cs_lookup[comp_name]
            features[idx, 0] = 2.0
            features[idx, 3] = cs.current / max(max_current, 1e-12)

    return features


def _physical_edge_order(
    circuit: Circuit,
    node_index: dict[str, int],
) -> list[tuple[int, int, str, str, str, str]]:
    """Return deterministic physical component edges with ground encoded as -1."""
    edges: list[tuple[int, int, str, str, str, str]] = []

    for r in circuit.resistors:
        src = node_index.get(r.node_a, -1) if r.node_a != circuit.ground_node else -1
        dst = node_index.get(r.node_b, -1) if r.node_b != circuit.ground_node else -1
        edges.append((src, dst, r.name, "resistor", r.node_a, r.node_b))

    for vs in circuit.voltage_sources:
        src = node_index.get(vs.positive, -1) if vs.positive != circuit.ground_node else -1
        dst = node_index.get(vs.negative, -1) if vs.negative != circuit.ground_node else -1
        edges.append((src, dst, vs.name, "voltage_source", vs.positive, vs.negative))

    for cs in circuit.current_sources:
        src = node_index.get(cs.positive, -1) if cs.positive != circuit.ground_node else -1
        dst = node_index.get(cs.negative, -1) if cs.negative != circuit.ground_node else -1
        edges.append((src, dst, cs.name, "current_source", cs.positive, cs.negative))

    return edges


def _edge_traversal_sign(edge_src: str, edge_dst: str, trav_src: str, trav_dst: str) -> float:
    return 1.0 if edge_src == trav_src and edge_dst == trav_dst else -1.0


def _compute_cycle_matrix(
    node_names: Tuple[str, ...],
    physical_edges: list[tuple[int, int, str, str, str, str]],
) -> torch.Tensor:
    """Compute a deterministic fundamental cycle basis over physical component edges."""
    num_edges = len(physical_edges)
    if num_edges == 0:
        return torch.zeros(0, 0, dtype=torch.float32)

    node_order = ["0", *node_names]
    parent = {node: node for node in node_order}
    rank = {node: 0 for node in node_order}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    tree_edge_ids: list[int] = []
    non_tree_edge_ids: list[int] = []
    edge_nodes = [("0" if src < 0 else node_names[src], "0" if dst < 0 else node_names[dst]) for src, dst, *_ in physical_edges]
    edge_orientations = [(src_name, dst_name) for _, _, _, _, src_name, dst_name in physical_edges]

    for edge_id, (src, dst) in enumerate(edge_nodes):
        if src == dst:
            non_tree_edge_ids.append(edge_id)
            continue
        if union(src, dst):
            tree_edge_ids.append(edge_id)
        else:
            non_tree_edge_ids.append(edge_id)

    tree_adj: dict[str, list[tuple[str, int]]] = {node: [] for node in node_order}
    for edge_id in tree_edge_ids:
        src, dst = edge_nodes[edge_id]
        tree_adj[src].append((dst, edge_id))
        tree_adj[dst].append((src, edge_id))

    for node in node_order:
        tree_adj[node].sort(key=lambda item: (item[0], item[1]))

    parent_node: dict[str, str | None] = {node: None for node in node_order}
    parent_edge: dict[str, int | None] = {node: None for node in node_order}
    depth: dict[str, int] = {node: 0 for node in node_order}
    visited: set[str] = set()

    for root in node_order:
        if root in visited:
            continue
        visited.add(root)
        parent_node[root] = None
        parent_edge[root] = None
        depth[root] = 0
        queue: deque[str] = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor, edge_id in tree_adj[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                parent_node[neighbor] = node
                parent_edge[neighbor] = edge_id
                depth[neighbor] = depth[node] + 1
                queue.append(neighbor)

    if not non_tree_edge_ids:
        return torch.zeros(0, num_edges, dtype=torch.float32)

    cycle_rows: list[torch.Tensor] = []
    for edge_id in non_tree_edge_ids:
        src, dst = edge_nodes[edge_id]
        row = torch.zeros(num_edges, dtype=torch.float32)
        row[edge_id] = _edge_traversal_sign(*edge_orientations[edge_id], src, dst)

        # Walk dst -> LCA
        ancestors: set[str] = set()
        node = src
        while node is not None:
            ancestors.add(node)
            node = parent_node[node]

        node = dst
        path_dst: list[tuple[str, str, int]] = []
        while node not in ancestors and parent_node[node] is not None:
            parent = parent_node[node]
            edge = parent_edge[node]
            assert parent is not None and edge is not None
            path_dst.append((node, parent, edge))
            node = parent
        lca = node

        path_src: list[tuple[str, str, int]] = []
        node = src
        while node != lca and parent_node[node] is not None:
            parent = parent_node[node]
            edge = parent_edge[node]
            assert parent is not None and edge is not None
            path_src.append((node, parent, edge))
            node = parent

        for child, parent, edge in path_dst:
            row[edge] = _edge_traversal_sign(*edge_orientations[edge], child, parent)
        for child, parent, edge in reversed(path_src):
            row[edge] = _edge_traversal_sign(*edge_orientations[edge], parent, child)

        cycle_rows.append(row)

    return torch.stack(cycle_rows, dim=0) if cycle_rows else torch.zeros(0, num_edges, dtype=torch.float32)


def circuit_to_graph(circuit: Circuit, solution: CircuitSolution) -> CircuitGraph:
    """Convert a Circuit + Solution into a deterministic CircuitGraph.

    Nodes sorted alphabetically, edges sorted by component name.
    Ground node ('0') excluded — its voltage is implicitly 0.
    """
    node_names = circuit.all_nodes  # already sorted, ground excluded
    n = len(node_names)
    if n == 0:
        # Degenerate: return empty graph
        return CircuitGraph(
            node_features=torch.zeros(0, NODE_DIM, dtype=torch.float32),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_features=torch.zeros(0, EDGE_DIM, dtype=torch.float32),
            target_voltages=torch.zeros(0, dtype=torch.float32),
            node_names=(),
            fingerprint="",
            component_edge_index=torch.zeros(2, 0, dtype=torch.long),
            cycle_matrix=torch.zeros(0, 0, dtype=torch.float32),
        )

    node_index = {name: idx for idx, name in enumerate(node_names)}

    physical_edges = _physical_edge_order(circuit, node_index)

    # Build edges for message passing: (src_idx, dst_idx, comp_name, comp_type)
    # Sorted deterministically by component name/category.
    edges: List[Tuple[int, int, str, str]] = [
        (src if src >= 0 else dst, dst if dst >= 0 else src, comp_name, comp_type)
        for src, dst, comp_name, comp_type, _, _ in physical_edges
    ]

    # Compute max values for normalization
    max_degree = 0
    for node in node_names:
        deg = 0
        for r in circuit.resistors:
            if r.node_a == node or r.node_b == node:
                deg += 1
        for vs in circuit.voltage_sources:
            if vs.positive == node or vs.negative == node:
                deg += 1
        for cs in circuit.current_sources:
            if cs.positive == node or cs.negative == node:
                deg += 1
        max_degree = max(max_degree, deg)

    max_resistance = max((r.resistance_ohm for r in circuit.resistors), default=1.0)
    max_voltage = max((abs(vs.voltage) for vs in circuit.voltage_sources), default=1.0)
    max_current = max((abs(cs.current) for cs in circuit.current_sources), default=1.0)

    # Build tensors
    node_features = _compute_node_features(circuit, node_names, max_degree)

    # Edge index: bidirectional for message passing
    if edges:
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        # Add reverse edges
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        edge_feat_forward = _compute_edge_features(circuit, edges, max_resistance, max_voltage, max_current)
        edge_features = torch.cat([edge_feat_forward, edge_feat_forward], dim=0)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_features = torch.zeros(0, EDGE_DIM, dtype=torch.float32)

    component_edge_index = (
        torch.tensor([[src for src, _, _, _, _, _ in physical_edges], [dst for _, dst, _, _, _, _ in physical_edges]], dtype=torch.long)
        if physical_edges
        else torch.zeros(2, 0, dtype=torch.long)
    )
    cycle_matrix = _compute_cycle_matrix(node_names, physical_edges)

    # Target voltages (ordered by node_names)
    target_voltages = torch.tensor(
        [solution.node_voltages.get(name, 0.0) for name in node_names],
        dtype=torch.float32,
    )

    # Fingerprint
    payload = {
        "node_features": node_features.tolist(),
        "edge_index": edge_index.tolist(),
        "edge_features": edge_features.tolist(),
        "target_voltages": target_voltages.tolist(),
        "node_names": list(node_names),
    }
    fp = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return CircuitGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        target_voltages=target_voltages,
        node_names=node_names,
        fingerprint=fp,
        component_edge_index=component_edge_index,
        cycle_matrix=cycle_matrix,
    )


def dataset_to_graphs(jsonl_path: Path) -> List[CircuitGraph]:
    """Load a JSONL dataset and convert all circuits to CircuitGraphs."""
    from backend.circuits.parser import parse_netlist
    from backend.circuits.dc_solver import solve_dc_circuit

    graphs: List[CircuitGraph] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            netlist = row["netlist"]
            name = row.get("circuit_name", "unnamed")
            try:
                circuit = parse_netlist(netlist, name=name)
                solution = solve_dc_circuit(circuit)
                g = circuit_to_graph(circuit, solution)
                graphs.append(g)
            except Exception:
                continue

    return graphs


@dataclass(frozen=True)
class GraphBatch:
    """Padded batch of CircuitGraphs for GNN training."""

    node_features: torch.Tensor  # (batch_size, max_nodes, NODE_DIM)
    edge_index: torch.Tensor  # (batch_size, 2, max_edges)
    edge_features: torch.Tensor  # (batch_size, max_edges, EDGE_DIM)
    component_edge_index: torch.Tensor  # (batch_size, 2, max_component_edges)
    cycle_matrix: torch.Tensor  # (batch_size, max_cycles, max_component_edges)
    target_voltages: torch.Tensor  # (batch_size, max_nodes)
    node_mask: torch.Tensor  # (batch_size, max_nodes) — 1 for real, 0 for pad
    num_nodes: Tuple[int, ...]
    num_edges: Tuple[int, ...]
    num_component_edges: Tuple[int, ...]
    num_cycles: Tuple[int, ...]


def collate_graphs(graphs: Sequence[CircuitGraph]) -> GraphBatch:
    """Collate a list of CircuitGraphs into a padded batch.

    Padding: zero-valued for features, zero mask for node_mask.
    Edge indices are offset per graph in the batch dimension.
    """
    if not graphs:
        return GraphBatch(
            node_features=torch.zeros(0, 0, NODE_DIM, dtype=torch.float32),
            edge_index=torch.zeros(0, 2, 0, dtype=torch.long),
            edge_features=torch.zeros(0, 0, EDGE_DIM, dtype=torch.float32),
            component_edge_index=torch.zeros(0, 2, 0, dtype=torch.long),
            cycle_matrix=torch.zeros(0, 0, 0, dtype=torch.float32),
            target_voltages=torch.zeros(0, 0, dtype=torch.float32),
            node_mask=torch.zeros(0, 0, dtype=torch.float32),
            num_nodes=(),
            num_edges=(),
            num_component_edges=(),
            num_cycles=(),
        )

    max_nodes = max(g.node_features.size(0) for g in graphs)
    max_nodes = max(max_nodes, 1)  # at least 1
    max_edges = max(g.edge_index.size(1) for g in graphs)
    max_edges = max(max_edges, 0)
    max_component_edges = max(g.component_edge_index.size(1) for g in graphs)
    max_component_edges = max(max_component_edges, 0)
    max_cycles = max(g.cycle_matrix.size(0) for g in graphs)
    max_cycles = max(max_cycles, 0)
    bs = len(graphs)

    node_features = torch.zeros(bs, max_nodes, NODE_DIM, dtype=torch.float32)
    edge_index_batch = torch.zeros(bs, 2, max_edges, dtype=torch.long)
    edge_features = torch.zeros(bs, max_edges, EDGE_DIM, dtype=torch.float32)
    component_edge_index = torch.full((bs, 2, max_component_edges), -1, dtype=torch.long)
    cycle_matrix = torch.zeros(bs, max_cycles, max_component_edges, dtype=torch.float32)
    target_voltages = torch.zeros(bs, max_nodes, dtype=torch.float32)
    node_mask = torch.zeros(bs, max_nodes, dtype=torch.float32)

    num_nodes_list = []
    num_edges_list = []
    num_component_edges_list = []
    num_cycles_list = []

    for i, g in enumerate(graphs):
        nn = g.node_features.size(0)
        ne = g.edge_index.size(1)
        nce = g.component_edge_index.size(1)
        ncy = g.cycle_matrix.size(0)
        num_nodes_list.append(nn)
        num_edges_list.append(ne)
        num_component_edges_list.append(nce)
        num_cycles_list.append(ncy)

        if nn > 0:
            node_features[i, :nn, :] = g.node_features
            target_voltages[i, :nn] = g.target_voltages
            node_mask[i, :nn] = 1.0

        if ne > 0:
            edge_index_batch[i, :, :ne] = g.edge_index
            edge_features[i, :ne, :] = g.edge_features
        if nce > 0:
            component_edge_index[i, :, :nce] = g.component_edge_index
        if ncy > 0 and nce > 0:
            cycle_matrix[i, :ncy, :nce] = g.cycle_matrix

    return GraphBatch(
        node_features=node_features,
        edge_index=edge_index_batch,
        edge_features=edge_features,
        component_edge_index=component_edge_index,
        cycle_matrix=cycle_matrix,
        target_voltages=target_voltages,
        node_mask=node_mask,
        num_nodes=tuple(num_nodes_list),
        num_edges=tuple(num_edges_list),
        num_component_edges=tuple(num_component_edges_list),
        num_cycles=tuple(num_cycles_list),
    )
