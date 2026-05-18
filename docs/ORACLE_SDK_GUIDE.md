# CPT Oracle SDK Guide

## Overview

The CPT Runtime allows ANY domain to be solved through a standardized
execution pipeline. You implement an **Oracle** that solves your domain
problem exactly, and the runtime handles surrogate approximation,
physics projection, evaluation, and memory registration.

## Quick Start

See `examples/oracle_template.py` for a minimal working example.

```python
from backend.core_runtime.task_runtime import RuntimeTask, RuntimeExecutor

class MyOracle:
    def solve(self, task_or_graph) -> dict:
        # Your exact solver here
        return {"voltages": tensor, "oracle_name": self.name()}

    def name(self) -> str:
        return "my_domain_oracle"

class MySurrogate:
    def predict(self, task_or_graph):
        # Your approximate model here
        return tensor

    def name(self) -> str:
        return "my_surrogate"

executor = RuntimeExecutor(
    oracle=MyOracle(),
    surrogate=MySurrogate(),
)

task = RuntimeTask(
    task_id="my_task_001",
    domain="my_domain",
    input_artifact="fingerprint_of_input",
    oracle_name="my_domain_oracle",
    surrogate_name="my_surrogate",
)

result = executor.execute(task)
```

## OracleProtocol Interface

```python
class OracleProtocol(Protocol):
    def solve(self, graph: Any) -> dict[str, Any]:
        """Return exact solution as dict with 'voltages' key (or domain equivalent)."""
        ...

    def name(self) -> str:
        """Return canonical oracle identifier."""
        ...
```

**Requirements:**
1. `solve()` must accept whatever the RuntimeExecutor passes (typically a RuntimeTask)
2. Return dict must contain a `'voltages'` key (or your domain's equivalent tensor)
3. Return dict should include `'oracle_name'` and `'latency_ms'`
4. Solution must be **deterministic** — same input always produces same output

## SurrogateProtocol Interface

```python
class SurrogateProtocol(Protocol):
    def predict(self, graph: Any) -> Any:
        """Return approximate prediction."""
        ...

    def name(self) -> str:
        """Return canonical surrogate identifier."""
        ...
```

## RuntimeTask Fields

| Field | Type | Description |
|-------|------|-------------|
| task_id | str | Unique task identifier |
| domain | str | Domain name (circuit, kicad, freecad, math, etc.) |
| input_artifact | str | Fingerprint or path to input data |
| oracle_name | str | Which oracle to use |
| surrogate_name | str | Which surrogate to use |
| projection_enabled | bool | Whether to run physics projection |
| metadata | dict | Domain-specific parameters |

## v2.13 Features

### Exact-Match Cache
```python
from backend.core_runtime.exact_cache import ExactMatchCache
from backend.core_runtime.task_hashing import compute_task_hash

cache = ExactMatchCache()
task_hash = compute_task_hash(task)

# Check cache before executing
cached = cache.get(task_hash)
if cached is not None:
    return cached  # Instant result!

# After execution, store result
cache.put(task_hash, result)
```

### Confidence-Aware Routing
```python
from backend.core_runtime.confidence_runtime import ConfidenceRuntime
from backend.core_runtime.capability_router import CapabilityRouter

confidence_rt = ConfidenceRuntime()
confidence = confidence_rt.estimate(task, graph_size=10, topology_family="mesh")

router = CapabilityRouter()
decision = router.route(task, confidence, cache_hit=False)
# decision.projection_budget -> max iterations
# decision.force_oracle -> True if oracle verification needed
```

### Execution Policy
```python
from backend.core_runtime.execution_policy import ExecutionPolicy, RecoveryHandler

policy = ExecutionPolicy(oracle_timeout_s=60.0, max_retries=3)
recovery = RecoveryHandler(policy)

# Check for degradation
reason = recovery.check_nan_output(result.surrogate_voltages, "surrogate", task)
if reason:
    degraded = recovery.make_degraded_result(task, reason, ...)
```

## Domain Integration Checklist

- [ ] Implement OracleProtocol.solve()
- [ ] Implement SurrogateProtocol.predict() (or use zero baseline)
- [ ] Define domain-specific metadata schema
- [ ] Implement canonical task hashing for your domain
- [ ] Test determinism: same input → same output, always
- [ ] Register with RuntimeExecutor
- [ ] Optionally implement ProjectionProtocol and EvaluatorProtocol

## Architecture

```
RuntimeTask
    │
    ▼
┌─────────────────────────────────────┐
│         RuntimeExecutor             │
│                                     │
│  1. Check ExactMatchCache           │  ← v2.13
│  2. EstimateConfidence              │  ← v2.13
│  3. Route via CapabilityRouter      │  ← v2.13
│  4. OracleProtocol.solve()          │
│  5. SurrogateProtocol.predict()     │
│  6. ProjectionProtocol.project()    │
│  7. RecoveryHandler checks          │  ← v2.13
│  8. EvaluatorProtocol.evaluate()    │
│  9. MemoryRuntime.register()        │
│ 10. ExactMatchCache.put()           │  ← v2.13
│                                     │
│  → RuntimeResult                    │
│  → ExecutionTrace                   │
└─────────────────────────────────────┘
```

## Current Domains

| Domain | Oracle | Surrogate | Status |
|--------|--------|-----------|--------|
| Circuit (DC) | MNAOracleAdapter | CircuitGNN / ZeroBaseline | Active |
| Example (Ohm's Law) | ExampleOhmsLawOracle | ExampleZeroSurrogate | Demo |
| KiCad | TBD | TBD | Planned |
| FreeCAD | TBD | TBD | Planned |
