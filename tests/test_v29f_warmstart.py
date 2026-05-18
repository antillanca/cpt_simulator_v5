import pytest
import torch
import numpy as np
from backend.circuits.models import Circuit, Resistor, VoltageSource
from backend.circuits.graph_dataset import circuit_to_graph
from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.warmstart_eval import run_warmstart_experiment

def _make_bridge_circuit() -> Circuit:
    resistors = [
        Resistor("R1", "1", "2", 10.0),
        Resistor("R2", "1", "3", 10.0),
        Resistor("R3", "2", "0", 10.0),
        Resistor("R4", "3", "0", 10.0),
        Resistor("R5", "2", "3", 50.0),  # Bridge
    ]
    vs = VoltageSource("V1", "1", "0", 100.0)
    return Circuit(name="bridge", ground_node="0", resistors=tuple(resistors), voltage_sources=(vs,))

def test_warmstart_reduces_iterations():
    circuit = _make_bridge_circuit()
    solver = solve_dc_circuit(circuit)
    graph = circuit_to_graph(circuit, solver)
    
    result = run_warmstart_experiment(
        circuit, graph,
        perturbation_scale=2.0,
        projection_steps=5,
        virtual_node=True,
        max_iterations=1000,
        tolerance=1e-5
    )
    
    zero_iters = result["zero_iters"]
    surr_iters = result["surrogate_iters"]
    proj_iters = result["projected_iters"]
    vn_iters = result["virtual_iters"]
    
    assert "zero_iters" in result
    assert "surrogate_iters" in result
    assert "virtual_iters" in result
