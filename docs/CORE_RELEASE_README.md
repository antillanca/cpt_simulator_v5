# CORE - Cognitive Operational Runtime Engine

## What CORE Is

CORE is a domain-agnostic runtime for physics-informed surrogate
projection. It schedules, routes, caches, and traces execution
across multiple computational domains.

## Architecture

- **core_runtime/core/** -- domain-agnostic runtime (scheduling, routing, memory, tracing)
- **core_runtime/domains/circuits/** -- first validated domain (v2.15.0 lineage)
- **core_runtime/domains/linear_system/** -- proof-of-concept second domain (v0.1.0)
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

- CORE: v3.0.0
- Circuits domain: v2.15.0 (frozen from v2.15-runtime-stable)
- Linear System domain: v0.1.0

## Test Status

- Core tests: 32 passing
- Full regression: 697 passing, 1 skipped

## Documentation

- docs/CORE_ARCHITECTURE.md -- package structure and relationships
- docs/CORE_PRINCIPLES.md -- 10 operational guarantees
- docs/CORE_DOMAIN_MODEL.md -- how domains plug in
- docs/CORE_RUNTIME_GUARANTEES.md -- deterministic guarantees
- docs/CORE_TRANSITION_PLAN.md -- migration from CPT to CORE
- docs/VERSIONING_POLICY.md -- semantic versioning rules
- docs/PAPER_POSITIONING.md -- academic contribution positioning

## What CORE Is NOT

- Not a learning system (no LoRA, replay, or continual training)
- Not a distributed runtime
- Not a physics engine
- Not a replacement for projection
