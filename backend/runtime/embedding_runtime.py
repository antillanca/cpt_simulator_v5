"""CPT Runtime — GNN Embedding Extraction.

Reuses the EXISTING frozen GNN encoder (inference-only).
Extracts the latent representation BEFORE the voltage prediction head.

DO NOT:
- Train anything new
- Modify the GNN architecture
- Enable dropout during inference
- Use non-deterministic operations
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import torch


# ---------------------------------------------------------------------------
# EmbeddingResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingResult:
    """Immutable embedding extraction result."""

    vector: tuple[float, ...]  # Fixed-length float32 embedding
    norm: float
    sha256: str
    topology_family: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "vector_sha256": self.sha256,
            "norm": round(self.norm, 8),
            "topology_family": self.topology_family,
            "dim": len(self.vector),
            "metadata": _sort_dict(self.metadata),
        }

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        topology_family: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> "EmbeddingResult":
        """Create from a tensor (float32, canonicalized)."""
        canonical = normalize_embedding(tensor)
        sha = compute_embedding_sha256(canonical)
        norm_val = float(canonical.norm().item())
        vec_tuple = tuple(round(float(v), 8) for v in canonical.detach().cpu().flatten().tolist())
        return cls(
            vector=vec_tuple,
            norm=round(norm_val, 8),
            sha256=sha,
            topology_family=topology_family,
            metadata=metadata or {},
        )


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def extract_graph_embedding(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract graph embedding from a GNN model BEFORE the prediction head.

    Supports CircuitGNN (no edge features) and EdgeAwareCircuitGNN.

    The model MUST be in eval mode (dropout disabled).
    Returns: (hidden_dim,) tensor — the graph-level embedding.
    """
    model.eval()
    with torch.no_grad():
        # Check if model has node_encoder (both CircuitGNN and EdgeAwareCircuitGNN do)
        if hasattr(model, "node_encoder"):
            h = model.node_encoder(x)
        else:
            h = x

        # Run through conv layers (before head)
        if hasattr(model, "conv1"):
            if edge_features is not None and hasattr(model.conv1, "edge_mlp"):
                # EdgeAwareCircuitGNN
                h = torch.relu(model.conv1(h, edge_index, edge_features))
                if hasattr(model, "conv2"):
                    h = torch.relu(model.conv2(h, edge_index, edge_features))
                if hasattr(model, "conv3"):
                    h = torch.relu(model.conv3(h, edge_index, edge_features))
            else:
                # CircuitGNN
                h = torch.relu(model.conv1(h, edge_index))
                if hasattr(model, "conv2"):
                    h = torch.relu(model.conv2(h, edge_index))

        # Graph-level pooling: mean over nodes
        # This gives us a fixed-size embedding regardless of graph size
        graph_embedding = h.mean(dim=0)  # (hidden_dim,)

    return graph_embedding


def normalize_embedding(tensor: torch.Tensor) -> torch.Tensor:
    """Canonicalize embedding to float32, CPU, contiguous."""
    result = tensor.detach().cpu().to(torch.float32).contiguous()
    # Deterministic: round to 6 sig figs to avoid float noise
    result = (result * 1e6).round() / 1e6
    return result


def compute_embedding_sha256(tensor: torch.Tensor) -> str:
    """Deterministic SHA-256 of a normalized embedding tensor."""
    canonical = normalize_embedding(tensor)
    # Convert to bytes deterministically
    values = canonical.flatten().tolist()
    blob = json.dumps(
        [round(float(v), 6) for v in values],
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _sort_dict(d: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k in sorted(d):
        v = d[k]
        result[k] = _sort_dict(v) if isinstance(v, dict) else v
    return result
