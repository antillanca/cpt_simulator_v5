import pytest
import torch
from backend.circuits.graph_dataset import CircuitGraph
from backend.neural.models.circuit_gnn import EdgeAwareCircuitGNN
from scripts.train_circuit_gnn import train_one_epoch, evaluate, TrainingGraph
from backend.circuits.physics_loss import PhysicsInformedLoss
from backend.circuits.models import Circuit

def make_dummy_graph(num_nodes: int, node_dim: int, edge_dim: int) -> CircuitGraph:
    return CircuitGraph(
        node_features=torch.randn(num_nodes, node_dim),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_features=torch.randn(2, edge_dim),
        target_voltages=torch.zeros(num_nodes),
        node_names=tuple([str(i) for i in range(num_nodes)]),
        fingerprint="dummy",
        cycle_matrix=torch.zeros(1, 1)
    )

def test_ablation_slicing_forward_pass():
    # Make a dummy training graph with maximum dims
    # NODE_DIM is 13, EDGE_DIM is 7
    g = make_dummy_graph(num_nodes=3, node_dim=13, edge_dim=7)
    
    # We will test all four ablation modes: baseline, norm_only, topo_only, full
    # For each, we check that train_one_epoch and evaluate run without shape mismatch errors.
    
    device = torch.device("cpu")
    loss_fn = PhysicsInformedLoss(lambda_kcl=1.0, lambda_kvl=1.0, lambda_power=0.1)
    dummy_circuit = Circuit(name="dummy")
    
    # 1. baseline: effective node_dim=8, edge_dim=4
    model_baseline = EdgeAwareCircuitGNN(node_dim=8, edge_dim=4, hidden_dim=16).to(device)
    optimizer = torch.optim.AdamW(model_baseline.parameters(), lr=1e-3)
    train_data = [TrainingGraph(graph=g, circuit=dummy_circuit, vmax=1.0)]
    
    # Check baseline training
    res = train_one_epoch(model_baseline, train_data, optimizer, loss_fn, "baseline", use_edge_features=True)
    assert "total_loss" in res
    
    # Check baseline eval
    eval_res = evaluate(model_baseline, train_data, loss_fn, "baseline", use_edge_features=True)
    assert "loss" in eval_res

    # 2. norm_only: effective node_dim=8, edge_dim=5
    model_norm = EdgeAwareCircuitGNN(node_dim=8, edge_dim=5, hidden_dim=16).to(device)
    optimizer = torch.optim.AdamW(model_norm.parameters(), lr=1e-3)
    res = train_one_epoch(model_norm, train_data, optimizer, loss_fn, "norm_only", use_edge_features=True)
    assert "total_loss" in res

    # 3. topo_only: effective node_dim=13, edge_dim=6 (original 4 + 2 cycle flags = 6)
    model_topo = EdgeAwareCircuitGNN(node_dim=13, edge_dim=6, hidden_dim=16).to(device)
    optimizer = torch.optim.AdamW(model_topo.parameters(), lr=1e-3)
    res = train_one_epoch(model_topo, train_data, optimizer, loss_fn, "topo_only", use_edge_features=True)
    assert "total_loss" in res

    # 4. full: effective node_dim=13, edge_dim=7
    model_full = EdgeAwareCircuitGNN(node_dim=13, edge_dim=7, hidden_dim=16).to(device)
    optimizer = torch.optim.AdamW(model_full.parameters(), lr=1e-3)
    res = train_one_epoch(model_full, train_data, optimizer, loss_fn, "full", use_edge_features=True)
    assert "total_loss" in res
