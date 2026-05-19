# CORE - Cognitive Operational Runtime Engine

## What CORE Is

CORE is not a simulator, nor a GNN framework, nor a physics solver.
It is a **deterministic hybrid runtime** that orchestrates oracle
computation, surrogate inference, constraint projection, memory
retrieval, and adaptive scheduling to execute verifiable cognitive
tasks.

CORE is the canonical runtime. CPT (Circuit Projection Tool) is the
first-domain validation lineage, originating from cpt_simulator_v5.
Circuits were the historical first domain that validated the entire
runtime architecture.

## Architecture

- **core_runtime/core/** -- domain-agnostic runtime (scheduling, routing, memory, tracing)
- **core_runtime/domains/circuits/** -- first validated domain (v2.15.0 lineage)
- **core_runtime/domains/linear_system/** -- proof-of-concept second domain (v0.2.0)
- **core_runtime/data/** -- migrated operational artifacts

## Key Facts

- Circuits were the first validated domain
- Deterministic runtime guarantees remain intact
- Same input always yields same execution trace
- Projection remains the final authority
- Future domains can plug into the same runtime via the Domain SDK

## Installation

```bash
# Core only (no circuit dependencies)
pip install core-runtime-engine

# With circuit domain
pip install core-runtime-engine[circuits]

# With linear system domain
pip install core-runtime-engine[linear-system]

# Development
pip install core-runtime-engine[dev]
```

## Version

- CORE: v3.1.0
- Circuits domain: v2.15.0 (frozen from v2.15-runtime-stable)
- Linear System domain: v0.2.0

## Test Status

- Core tests: 31+ passing
- Full regression: 742+ passing, 1 skipped
- v3.1 additions: oscillatory convergence frozen, linear_system E2E pipeline

## Documentation

- docs/CORE_ARCHITECTURE.md -- package structure and relationships
- docs/CORE_PRINCIPLES.md -- 10 operational guarantees + CPT lineage
- docs/CORE_DOMAIN_MODEL.md -- how domains plug in
- docs/CORE_RUNTIME_GUARANTEES.md -- deterministic guarantees
- docs/CORE_TRANSITION_PLAN.md -- migration from CPT to CORE
- docs/VERSIONING_POLICY.md -- semantic versioning rules
- docs/PAPER_POSITIONING.md -- academic contribution positioning
- docs/V215_ADAPTIVE_RUNTIME_SCHEDULING.md -- v2.15 operational guarantees
- docs/V215_STABILITY_GUARANTEES.md -- v2.15 frozen APIs

## What CORE Is NOT

- Not a learning system (no LoRA, replay, or continual training)
- Not a distributed runtime
- Not a physics engine
- Not a replacement for projection
