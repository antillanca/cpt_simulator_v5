import pytest
import torch
from backend.circuits.models import Circuit, Resistor
from backend.circuits.graph_dataset import CircuitGraph
from backend.circuits.failure_analysis import classify_failure

def test_topology_failure_taxonomy():
    # 1. Closed loop circuit (cycle_drift_failure)
    # A triangular mesh: 3 nodes (1, 2, 0)
    resistors = [
        Resistor("r1", "1", "2", 10.0),
        Resistor("r2", "2", "0", 20.0),
        Resistor("r3", "0", "1", 30.0),
    ]
    circuit_cycle = Circuit("circuit_cycle", ground_node="0", resistors=resistors)
    graph_cycle = CircuitGraph(
        node_features=torch.zeros(2, 13),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_features=torch.zeros(0, 7),
        target_voltages=torch.zeros(2),
        node_names=("1", "2"),
        fingerprint="dummy",
        cycle_matrix=torch.zeros(1, 1)
    )
    # Pred vs oracle voltages
    predicted = torch.tensor([5.0, 2.0])
    oracle = torch.tensor([5.0, 2.0])
    # KCL max violation is high
    invariant_metrics = {"kcl_max_violation": 0.5, "kvl_max_violation": 0.0, "power_conservation_violation": 0.0}
    
    result = classify_failure(circuit_cycle, graph_cycle, predicted, oracle, invariant_metrics=invariant_metrics, ood=True)
    assert result["failure_type"] == "cycle_drift_failure"
    assert "KCL drift in closed cycles" in result["reasons"]

    # 2. Large mesh circuit (dense_mesh_leakage)
    # 12 nodes, 11 resistors in a chain (no closed loops but node count > 10)
    resistors_large = [Resistor(f"r{i}", str(i), str(i+1) if i < 11 else "0", 10.0) for i in range(1, 12)]
    circuit_large = Circuit("circuit_large", ground_node="0", resistors=resistors_large)
    graph_large = CircuitGraph(
        node_features=torch.zeros(11, 13),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_features=torch.zeros(0, 7),
        target_voltages=torch.zeros(11),
        node_names=tuple(str(i) for i in range(1, 12)),
        fingerprint="dummy",
        cycle_matrix=torch.zeros(0, 1)
    )
    predicted_large = torch.zeros(11)
    oracle_large = torch.zeros(11)
    
    result_large = classify_failure(circuit_large, graph_large, predicted_large, oracle_large, invariant_metrics=invariant_metrics, ood=True)
    assert result_large["failure_type"] == "dense_mesh_leakage"
    assert "high degree node connectivity leakage" in result_large["reasons"]

    # 3. Small tree-like circuit (bridge_node_instability)
    # 3 nodes (1, 2, 0), 2 resistors in a chain
    resistors_tree = [
        Resistor("r1", "1", "2", 10.0),
        Resistor("r2", "2", "0", 20.0),
    ]
    circuit_tree = Circuit("circuit_tree", ground_node="0", resistors=resistors_tree)
    graph_tree = CircuitGraph(
        node_features=torch.zeros(2, 13),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_features=torch.zeros(0, 7),
        target_voltages=torch.zeros(2),
        node_names=("1", "2"),
        fingerprint="dummy",
        cycle_matrix=torch.zeros(0, 1)
    )
    predicted_tree = torch.zeros(2)
    oracle_tree = torch.zeros(2)
    
    result_tree = classify_failure(circuit_tree, graph_tree, predicted_tree, oracle_tree, invariant_metrics=invariant_metrics, ood=True)
    assert result_tree["failure_type"] == "bridge_node_instability"
    assert "bridge/tree node prediction instability" in result_tree["reasons"]
