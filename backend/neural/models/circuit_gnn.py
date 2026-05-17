"""Tiny GNN surrogate for DC circuit voltage prediction.

Architecture: message-passing GNN with edge conditioning.
Target: < 250k params.

Predicts node voltages for each non-ground node.
Uses signed log1p transform internally for training stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualGCNConv(nn.Module):
    """Minimal GCN-like layer with symmetric normalization."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """x: (N, in_dim), edge_index: (2, E)"""
        N = x.size(0)
        # Add self-loops
        self_loop_idx = torch.arange(N, device=x.device).unsqueeze(0).repeat(2, 1)
        full_edge_index = torch.cat([edge_index, self_loop_idx], dim=1)
        row, col = full_edge_index

        # Degree normalization
        deg = torch.zeros(N, device=x.device, dtype=torch.float32)
        deg.scatter_add_(0, col, torch.ones(full_edge_index.size(1), device=x.device, dtype=torch.float32))
        deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)

        # Normalized message passing: D^{-1/2} A D^{-1/2} X W
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = torch.zeros_like(x)
        # Accumulate: out[col] += norm * x[row]
        msg = x[row] * norm.unsqueeze(-1)
        out.scatter_add_(0, col.unsqueeze(-1).expand_as(msg), msg)

        return self.weight(out) + self.bias


class EdgeConditionedConv(nn.Module):
    """Message passing with edge feature conditioning.

    m_{ij} = MLP_edge([x_i || x_j || e_{ij}])
    x_i' = MLP_update([x_i || aggregate(m_{ij})])
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor
    ) -> torch.Tensor:
        """x: (N, node_dim), edge_index: (2, E), edge_features: (E, edge_dim)"""
        N = x.size(0)

        # For each edge, concatenate source, target, and edge features
        src, dst = edge_index  # src->dst
        # Build message: [x_src || x_dst || edge_feat]
        msg_input = torch.cat([x[src], x[dst], edge_features], dim=-1)
        messages = self.edge_mlp(msg_input)  # (E, hidden_dim)

        # Aggregate messages at destination nodes (sum)
        agg = torch.zeros(N, messages.size(1), device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # Update: [x || agg]
        update_input = torch.cat([x, agg], dim=-1)
        return self.update_mlp(update_input)


class CircuitGNN(nn.Module):
    """Basic GNN for circuit voltage prediction (no edge features)."""

    def __init__(self, node_dim: int = 8, edge_dim: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
        )
        self.conv1 = ManualGCNConv(hidden_dim, hidden_dim)
        self.conv2 = ManualGCNConv(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.node_encoder(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.head(x).squeeze(-1)  # (N,)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class EdgeAwareCircuitGNN(nn.Module):
    """Edge-aware GNN for circuit voltage prediction.

    Uses 3 edge-conditioned message passing layers for richer representation.
    """

    def __init__(self, node_dim: int = 8, edge_dim: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
        )
        self.conv1 = EdgeConditionedConv(hidden_dim, edge_dim, hidden_dim)
        self.conv2 = EdgeConditionedConv(hidden_dim, edge_dim, hidden_dim)
        self.conv3 = EdgeConditionedConv(hidden_dim, edge_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        x = self.node_encoder(x)
        x = F.relu(self.conv1(x, edge_index, edge_features))
        x = F.relu(self.conv2(x, edge_index, edge_features))
        x = F.relu(self.conv3(x, edge_index, edge_features))
        return self.head(x).squeeze(-1)  # (N,)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
