# CPT v2.11 — Core Standardization Freeze

## Why Standardization Before Scaling

CPT is about to grow beyond circuit simulation into:
- **KiCad** plugin (graph import from schematic netlists)
- **FreeCAD** material simulation (physics projection for FEM)
- **mathematics / logic / programming / language** reasoning domains

Each domain needs the same guarantees: deterministic fingerprints, immutable contracts, reproducible experiments. Without a frozen core spec, every new domain invents its own ad-hoc format → fragmentation → technical debt.

**v2.11 freezes the foundation. No new AI, no new models. Just stable contracts.**

---

## Canonical Contracts

### 1. CanonicalCircuitGraph (`graph_spec.py`)

```python
@dataclass(frozen=True)
class CanonicalCircuitGraph:
    graph_id: str
    fingerprint: str          # SHA-256 over canonical JSON
    num_nodes: int
    num_edges: int
    node_features: Tensor     # [N, D]
    edge_index: Tensor        # [2, E]
    edge_features: Tensor     # [E, D_e]
    topology_family: TopologyFamily
    cycle_count: int
    connected_components: int
    source_nodes: list[int]
    ground_node: int
    metadata: dict[str, Any]
```

- **Deterministic fingerprint**: SHA-256 over sorted JSON (tensors → lists, floats → 6 decimal places)
- **Topology family enum**: `RADIAL | MESH | BRIDGE | CURRENT_SOURCE | MIXED | UNKNOWN`
- **Conversion helper**: `from_circuit_graph(CircuitGraph) → CanonicalCircuitGraph`
- **Validation**: `validate_graph()` catches degenerate structures

### 2. ProjectionResult (`projection_spec.py`)

```python
@dataclass(frozen=True)
class ProjectionResult:
    iterations: int
    initial_kcl_residual: float
    final_kcl_residual: float
    initial_kvl_residual: float
    final_kvl_residual: float
    initial_power_residual: float
    final_power_residual: float
    converged: bool
    used_virtual_node: bool
    projection_time_ms: float
```

- **Bridge**: `from_projection_effort(effort_dict) → ProjectionResult`
- **Deterministic fingerprint**, JSON roundtrip, validation

### 3. CPTModel Protocol + ModelMetadata (`model_spec.py`)

```python
class CPTModel(Protocol):
    def predict(self, graph): ...
    def fingerprint(self) -> str: ...
    def export(self, path): ...
    def load(self, path): ...
    def metadata(self) -> ModelMetadata: ...

@dataclass(frozen=True)
class ModelMetadata:
    model_name: str
    version: str
    parameter_count: int
    topology_specialization: str | None
    training_dataset_fingerprint: str
    projection_aware: bool
```

- **CircuitGNNAdapter**: wraps existing `CircuitGNN` to satisfy `CPTModel`
- Future models (LoRA experts, domain-specific) implement the same protocol

### 4. ExperimentSpec (`experiment_spec.py`)

```python
@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    seed: int
    dataset_fingerprint: str
    checkpoint_fingerprint: str | None
    target_mode: str           # oracle | blended_projection
    topology_curriculum: bool
    projection_enabled: bool
    projection_config: dict
    training_config: dict
    evaluation_config: dict
    metadata: dict
```

- **Deterministic fingerprint** — same config → same ID
- **Validation**: target_mode against known modes
- **YAML export**: `to_yaml_lines()` for human-readable config

### 5. EvaluationReport (`report_spec.py`)

```python
@dataclass(frozen=True)
class EvaluationReport:
    report_id: str
    model_fingerprint: str
    dataset_fingerprint: str
    iid_mae: float
    ood_mae: float
    iid_kcl_max: float
    ood_kcl_max: float
    iid_kvl_max: float
    ood_kvl_max: float
    projection_iterations_mean: float
    speedup_factor: float
    topology_metrics: dict
    failure_summary: dict[str, int]
```

- **Deterministic fingerprint** — insertion-order-independent
- **Validation**: MAE ≥ 0, fingerprint consistency

### 6. Failure Taxonomy (`failure_taxonomy.py`)

```
TOPOLOGY:  topology_collapse, disconnected_graph_confusion,
           symmetry_failure, node_aliasing

PHYSICS:   conservation_drift, cycle_drift_failure,
           dense_mesh_leakage, bridge_node_instability,
           extreme_resistance_instability

PROJECTION: projection_overshoot

OOD:       ood_extreme_resistance, ood_generalization_failure (legacy),
           ood_voltage_explosion
```

- **Frozen list**: `FAILURE_TYPES` — single source of truth
- **Category groups**: `FailureCategory` enum
- **Validation helpers**: `is_valid_failure_type()`, `category_of()`
- **Consistency check**: `validate_taxonomy_consistency()`
- **`failure_analysis.py` now imports from core_spec** (no duplicate definitions)

### 7. MemoryEntry (`memory_spec.py`)

```python
@dataclass(frozen=True)
class MemoryEntry:
    entry_id: str
    graph_fingerprint: str
    topology_family: str
    projection_iterations: int
    initial_residual: float
    final_residual: float
    dominant_failure: str | None
    oracle_time_ms: float
    projection_time_ms: float
    used_lora_expert: str | None
    metadata: dict
```

- **Schema ONLY** — no FAISS, no retrieval
- **Deterministic fingerprint**, JSON roundtrip, validation
- **dominant_failure** validated against `FAILURE_TYPES`

---

## Stable Interfaces

| Interface | Purpose | Extensibility |
|-----------|---------|---------------|
| `CPTModel` Protocol | All models satisfy `predict/fingerprint/export/load/metadata` | New domains add implementations |
| `CanonicalCircuitGraph` | Universal graph representation | KiCad netlists → same format |
| `ProjectionResult` | Physics projection outcomes | FreeCAD FEM → same contract |
| `ExperimentSpec` | Reproducible experiment configs | Any domain's experiments |
| `EvaluationReport` | Comparable evaluation results | Cross-domain benchmarking |
| `FAILURE_TYPES` | Frozen failure names | New types via PR to taxonomy |
| `MemoryEntry` | Replay learning schema | FAISS indexing when ready |

---

## Deterministic Reproducibility Guarantees

1. **All fingerprints are SHA-256 over sorted canonical JSON**
   - Dict key order does not affect fingerprint
   - Float rounding: 6 decimal places (graph), 12 (residuals), 3 (time_ms)

2. **All dataclasses are `frozen=True`** — no mutation after construction

3. **All contracts have `validate()` → list[str]**
   - Returns empty list if valid
   - Returns human-readable errors if invalid

4. **All contracts have `to_json_dict()` / `from_json_dict()`**
   - Full roundtrip preserves fingerprint

5. **Seed is part of ExperimentSpec fingerprint**
   - Same config + same seed → same experiment ID

---

## Future Compatibility

| Feature | How v2.11 Enables It |
|---------|---------------------|
| KiCad plugin | `CanonicalCircuitGraph` + `from_circuit_graph()` adapter |
| FreeCAD FEM | `ProjectionResult` contract applies to any physics domain |
| LoRA experts | `CPTModel` protocol + `ModelMetadata.topology_specialization` |
| FAISS memory | `MemoryEntry` schema ready for vector indexing |
| Replay learning | `MemoryEntry` captures projection effort per graph |
| Cross-domain eval | `EvaluationReport` is domain-agnostic |
| Artifact governance | All fingerprints compose into `ArtifactRecord` |

---

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `backend/core_spec/__init__.py` | 70 | Package exports |
| `backend/core_spec/graph_spec.py` | ~240 | CanonicalCircuitGraph + TopologyFamily |
| `backend/core_spec/projection_spec.py` | ~140 | ProjectionResult |
| `backend/core_spec/model_spec.py` | ~180 | CPTModel protocol + ModelMetadata + CircuitGNNAdapter |
| `backend/core_spec/experiment_spec.py` | ~120 | ExperimentSpec |
| `backend/core_spec/report_spec.py` | ~120 | EvaluationReport |
| `backend/core_spec/failure_taxonomy.py` | ~110 | FAILURE_TYPES + FailureCategory |
| `backend/core_spec/memory_spec.py` | ~120 | MemoryEntry |
| `tests/test_v211_core_standardization.py` | ~460 | 46 tests |

## Files Modified

| File | Change |
|------|--------|
| `backend/circuits/failure_analysis.py` | `FAILURE_TYPES` now imported from `core_spec/failure_taxonomy.py` |

---

## Test Coverage

46 tests covering:
- Deterministic fingerprints (same input → same hash)
- Fingerprint differentiation (different input → different hash)
- Serialization roundtrip (JSON → object → JSON → same fingerprint)
- Validation catches invalid inputs
- Frozen dataclasses (AttributeError on mutation)
- Taxonomy consistency (all types in exactly one category)
- Dict order independence (metadata key order doesn't affect fingerprint)
- Cross-contract conversion (`from_circuit_graph`, `from_projection_effort`)

**46/46 passed. No regression in v2.10 (69/69 combined).**
