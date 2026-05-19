# CORE Versioning Policy

## Semantic Versioning

All CORE components follow strict semantic versioning (MAJOR.MINOR.PATCH).

### CORE Runtime

- **Current**: v3.0.0
- **MAJOR**: Breaking changes to runtime contracts, frozen API signatures,
  or deterministic guarantee semantics
- **MINOR**: New domain SDK methods, new scheduling policies, new tracing
  fields (with defaults)
- **PATCH**: Bug fixes, performance improvements, documentation

### Domain Packages

Each domain versions independently:

| Domain | Current | Notes |
|--------|---------|-------|
| circuits | v2.15.0 | Carries forward the v2.15 frozen lineage |
| linear_system | v0.1.0 | Proof-of-concept domain |

Domain version rules:
- **MAJOR**: Breaking changes to domain API or projection semantics
- **MINOR**: New domain features, new task types, new evaluator metrics
- **PATCH**: Bug fixes, surrogate model updates, projection tuning

### Frozen Specs

Frozen schemas from v2.15 (experience_dataset_schema.py) remain
readable across all v3.x releases. A schema format change requires
a MAJOR CORE version bump.

## Compatibility Matrix

| CORE | circuits | linear_system | Compatible |
|------|----------|---------------|------------|
| 3.0.x | 2.15.x | 0.1.x | Yes |
| 3.1.x | 2.15.x | 0.2.x | Yes |
| 4.0.x | 3.0.x | 1.0.x | Breaking |

## Backward Compatibility

- Old import paths (`backend.runtime.*`, `backend.core_runtime.*`)
  remain functional through compatibility shims until v3.1+
- Deprecation warnings are emitted when old paths are used
- Removal of shims requires MAJOR version bump
- v2.15 operational experience JSONL/CSV remain loadable in v3.x

## Release Tagging

- CORE releases: `core-v3.0.0`
- Domain releases: `circuits-v2.15.0`, `linear-system-v0.1.0`
- Combined releases: `core-v3.0.0-circuits-v2.15.0`
