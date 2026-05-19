# CORE Domain Model

## What Is a Domain?

A domain is a self-contained computational world that plugs into
the CORE runtime. Each domain provides:

- Tasks (what to solve)
- Oracle (ground truth solver)
- Surrogate (fast approximate solver)
- Projection (iterative refinement toward correctness)
- Evaluator (correctness measurement)
- Optional: Confidence scorer

The core runtime handles scheduling, caching, retrieval, tracing,
and routing. The domain handles the actual computation.

## Domain SDK Protocol Interfaces

```python
class DomainTask(Protocol):
    task_id: str
    domain_name: str
    def fingerprint(self) -> str: ...
    def node_count(self) -> int: ...
    def edge_count(self) -> int: ...

class DomainOracle(Protocol):
    def solve(self, task: DomainTaskBase) -> dict: ...

class DomainSurrogate(Protocol):
    def predict(self, task: DomainTaskBase) -> dict: ...

class DomainProjection(Protocol):
    def project(self, task, prediction, budget) -> dict: ...

class DomainEvaluator(Protocol):
    def evaluate(self, task, solution) -> dict: ...

class DomainConfidence(Protocol):
    def score(self, task, prediction) -> float: ...
```

## DomainTaskBase

```python
@dataclass(frozen=True)
class DomainTaskBase:
    task_id: str
    domain_name: str
    input_artifact: str
    metadata: dict[str, Any]
```

Every domain task inherits from DomainTaskBase. The core runtime
only knows DomainTaskBase, never domain-specific fields.

## Domain Registration

```python
register_domain("circuits", oracle=CircuitOracle, surrogate=..., ...)
register_domain("linear_system", oracle=LinearSystemOracle, ...)

components = get_domain_components("circuits")
domains = list_domains()
```

## Current Domains

### Circuits (v2.15.0)

The first validated domain. MNA oracle, KCL/KVL projection,
CircuitGraph surrogate, topology families, confidence scoring.
This domain validated the entire adaptive scheduling stack through
v2.14 and v2.15.

### Linear System (v0.1.0)

Proof-of-concept second domain. Solves Ax = b using numpy.
Oracle: exact solve. Surrogate: Jacobi iteration. Projection:
gradient descent on ||b - Ax||^2. Evaluator: residual norm.
Tiny but real -- proves the SDK is sufficient.

## Future Domains

The SDK is designed so future domains can plug in without
modifying the core:

- **KiCad**: circuit board layout verification
- **FreeCAD**: mechanical CAD constraint solving
- **Mathematics**: symbolic computation verification
- **Logic**: SAT/SMT solver with projection
- **Programming**: code generation with verification

Each future domain implements the same Protocol interfaces
and registers with the core runtime.

## CORE Is the Canonical Runtime

CORE is the canonical runtime. CPT is the historical
first-domain validation lineage. The circuit domain was
the first domain to validate CORE's adaptive scheduling,
trajectory analysis, and retrieval-assisted warmstart.
All guarantees proven with circuits carry forward to
future domains through the SDK.
