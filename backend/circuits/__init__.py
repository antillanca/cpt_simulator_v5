"""CPT v2.8+ Circuit Oracle Core — deterministic DC circuit reasoning."""

from backend.circuits.models import Circuit, Resistor, VoltageSource, CurrentSource, CircuitSolution
from backend.circuits.parser import parse_netlist
from backend.circuits.dc_solver import solve_dc_circuit
from backend.circuits.invariants import validate_invariants, InvariantResult
from backend.circuits.traces import generate_oracle_trace
from backend.circuits.graph_dataset import CircuitGraph, circuit_to_graph
from backend.circuits.surrogate_eval import evaluate_surrogate, SurrogateEvalResult
from backend.circuits.ood_generator import generate_ood_circuits
