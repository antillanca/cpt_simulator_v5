# CPT v2.7.5 — Tiny Distillation Experiments

## Overview

CPT v2.7.5 extends the v2.7 distillation layer into a first real experiment stack:

- large deterministic oracle generation
- resumable sharded exports
- deterministic tiny-model training loops
- oracle-vs-model arena comparison
- failure taxonomy
- invariant retention and replay metrics
- reproducible evaluation reports

Large models remain out of scope. The focus stays on tiny, deterministic experiments against oracle truth.

---

## Mission Scope

This layer answers one question:

Can a small neural model preserve symbolic invariants, trace structure, causal consistency, and compositional reasoning when trained on deterministic oracle datasets?

The implementation is intentionally narrow:

- do not expand curriculum semantics
- do not rewrite core truth or sandbox behavior
- do not require Kaggle or GPU access
- do not introduce uncontrolled randomness
- do not optimize for chat behavior

---

## Dataset Export Contract

### Schema Version: 2.7.0

Every exported JSONL sample follows this strict schema:

```json
{
  "sample_id": "str — deterministic hash of module+seed+index+params",
  "question": "str — natural language description",
  "structured_state": {
    "initial_state": "dict — sandbox input state",
    "parameters": "dict — parameter sweep values",
    "module": "str — module key",
    "module_version": "str — SHA-256 of module definition"
  },
  "reasoning_trace": "list[dict] — ordered trace steps from sandbox",
  "equations_used": "list[str] — Lua equations executed",
  "invariants_checked": "list[str] — invariant names verified",
  "final_answer": "dict — sandbox particle output",
  "verification_status": "dict — {passed: bool, violations: list}",
  "module_source": "str — provenance string",
  "curriculum_layer": "int — layer 0-34",
  "seed": "int — generation seed",
  "timestamp": "float — deterministic timestamp",
  "dataset_version": "str — '2.7.0'",
  "snapshot_hash": "str — snapshot fingerprint at generation time",
  "module_hash": "str — hash of modules.json"
}
```

### Key Properties

- **Deterministic ordering**: All JSON keys sorted alphabetically
- **Reproducible by seed**: Same seed + same modules = identical dataset
- **Versioned schema**: `dataset_version` field enables backward-compatible readers
- **Validation**: `validate_export_row()` checks all required fields and types
- **Fingerprinting**: `export_fingerprint()` hashes canonical fields only

### Module: `backend/datasets/export_contract.py`

- `EXPORT_SCHEMA_VERSION = "2.7.0"`
- `STRICT_EXPORT_FIELDS` — ordered list of required fields
- `validate_export_row(row)` → list of errors
- `normalize_export_row(row, dataset_version, snapshot_hash, module_hash)` → upgraded row
- `row_to_contract(row, ...)` → validated + fingerprinted row
- `export_fingerprint(row)` → SHA-256 hex digest

---

## Dataset Manifests

Every export generates a manifest with full provenance.

### Manifest Fields

| Field | Type | Description |
|-------|------|-------------|
| dataset_version | str | "2.7.0" |
| schema_version | str | "2.7.0" |
| snapshot_hash | str | System snapshot fingerprint |
| module_hash | str | SHA-256 of modules.json |
| curriculum_coverage | list[int] | Layers present in dataset |
| generation_seed | int | Seed used for generation |
| record_count | int | Number of samples |
| shard_list | list[str] | Shard filenames (if sharded) |
| benchmark_fingerprint | str | Benchmark suite fingerprint |
| modules_used | list[str] | Module keys used |
| module_versions | dict | Module key → version hash |
| parameter_sweeps | dict | Parameter sweep configuration |
| timestamp | str | ISO 8601 generation time |
| timestamp_unix | float | Unix timestamp |
| generator_version | str | "oracle-v2.7" |
| fingerprint | str | Deterministic manifest fingerprint |

### Module: `backend/datasets/manifest.py`

- `DatasetManifest` — dataclass with `compute_fingerprint()`, `to_dict()`, `save()`, `from_file()`
- `validate_manifest(data)` → list of errors (includes fingerprint verification)
- `DatasetManifest.from_oracle_result(...)` — build from generation output

---

## Sharding

Large datasets can be split into fixed-size shards.

### Module: `backend/datasets/sharding.py`

- `shard_dataset(input_path, output_dir, shard_size)` → shard manifest dict
- `save_shard_manifest(manifest, path)` / `load_shard_manifest(path)`
- `validate_shard_manifest(manifest)` — includes fingerprint check
- `iter_shard_records(shard_path)` → list of records
- `iter_dataset_from_shards(shard_dir, manifest)` → generator
- `reassemble_dataset(shard_dir, manifest, output_path)` → reassembled JSONL

### Shard Naming

`shard_0000.jsonl`, `shard_0001.jsonl`, etc.

### Shard Manifest

Each shard records: name, path, record count, content hash.

---

## Dataset Loader

### Module: `backend/datasets/loader.py`

- `load_jsonl(path, validate)` → list of dicts
- `iter_jsonl(path, validate)` → generator
- `load_with_manifest(dataset_path, manifest_path)` → (rows, manifest)
- `upgrade_v26_row(row, ...)` — auto-upgrade v2.6 rows to v2.7 contract
- `load_sharded_dataset(shard_dir, manifest_path)` → (rows, manifest)

### Backward Compatibility

v2.6 datasets can be loaded and auto-upgraded:
- Missing `dataset_version`, `snapshot_hash`, `module_hash` are filled with defaults
- The `upgrade_v26_row()` function handles the upgrade

---

## Training Scaffold

### Architecture Support

| Model Type | Class | Description |
|-----------|-------|-------------|
| transformer | `TinyTransformerModel` | Small transformer baseline |
| seq2seq | `TinySeq2SeqModel` | Minimal encoder-decoder baseline |
| gnn | `TinyGNNModel` | Placeholder graph-style baseline |
| pinn | `TinyPINNModel` | Placeholder physics-informed baseline |

All models implement `NeuralModel.predict(inputs) → dict`.

### Dataloader: `backend/neural/dataloaders/`

- `DatasetShardLoader` — deterministic train/eval split
  - Same seed → same split every time
  - Default 80/20 split
  - Supports both JSONL and sharded datasets

### Scripts

- `scripts/generate_large_dataset.py` — resumable large oracle export
  - `--output-dir`, `--modules`, `--seed`, `--num-samples`, `--shard-size`, `--resume`
- `scripts/train_tiny_model.py` — functional tiny-model training loop
  - `--dataset`, `--shard-dir`, `--manifest`, `--seed`, `--epochs`, `--batch-size`, `--lr`, `--model-type`, `--output-dir`
- `scripts/model_eval_runner.py` — oracle-vs-model evaluation runner
  - `--dataset`, `--checkpoint`, `--predictions`, `--model-type`, `--seed`, `--output`, `--layer`, `--module`
- `scripts/eval_tiny_model.py` — compatibility wrapper around the new eval runner

### Runtime Notes

- PyTorch is used for the tiny-model baselines.
- CPU is the default execution path.
- CUDA is optional and isolated behind the training config.
- Checkpoints are deterministic for the same seed, data, and hyperparameters.

---

## Evaluation Harness

### Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| exact_match_rate | Exact dict match of final_answer | 0-1 |
| token_or_struct_match_rate | Same keys and value types | 0-1 |
| invariant_violation_rate | Fraction violating invariants | 0-1 (lower=better) |
| trace_consistency | Matching trace step counts | 0-1 |
| answer_consistency | Key-value matches in final_answer | 0-1 |
| trajectory_deviation | Average step count deviation | 0+ (lower=better) |
| replay_consistency | Fraction passing replay check | 0-1 |

### Aggregate Reporting

All metrics can be broken down by:
- **Layer**: `result.by_layer[layer]`
- **Module**: `result.by_module[module_key]`
- **Category**: `result.by_category[category]`

### Module: `backend/validation/model_eval.py`

- `ModelEvaluator(model_type)` — main evaluator class
- `ModelEvaluator.evaluate(predictions, oracle_records)` → `ModelEvaluationResult`
- `ModelEvaluationResult.save(path)` — save JSON with fingerprint
- Individual metric functions available for custom pipelines

---

## Fingerprint Policy

Fingerprints detect **meaningful changes**. A fingerprint changes when:

| Component | Fingerprint changes if... |
|-----------|--------------------------|
| Dataset row | Any canonical field value changes |
| Manifest | Any provenance field changes (seed, modules, version, etc.) |
| Shard | Content hash of any shard changes |
| Evaluation | Any metric value, model type, or sample count changes |
| Snapshot | Git hash, module hash, config, or key file hash changes |

Same inputs → same fingerprint. Always.

### Fingerprint Algorithm

```python
payload = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

Compact JSON encoding (no whitespace) ensures deterministic hashing.

---

## Versioning Policy

- **Major version** (2.x): Breaking changes to export contract
- **Minor version** (2.7.x): New fields (backward-compatible readers must handle missing fields)
- **Schema version**: Tracked in `dataset_version` and `schema_version` fields

### Upgrade Path

v2.6 → v2.7: Add `dataset_version`, `snapshot_hash`, `module_hash` fields.
Use `upgrade_v26_row()` for automatic migration.

---

## Future GPU/Kaggle Path

### Module: `backend/neural/kaggle_hooks.py`

- `TrainingProfile` — config-driven training profile (JSON)
- `package_dataset_for_upload()` — prepare a directory for Kaggle upload
- `create_kaggle_metadata()` — generate Kaggle metadata.json

### Workflow

1. Generate oracle dataset → JSONL + manifest
2. Optionally shard for large datasets
3. Package with `package_dataset_for_upload()`
4. Upload to Kaggle as a dataset
5. Create notebook using `training_profile.json`
6. Train on GPU, download model
7. Evaluate locally with `model_eval_runner.py` or `eval_tiny_model.py`

### Not Required for Local Operation

Kaggle hooks are **optional and isolated**. All local workflows work without them.

---

## File Reference

### New in v2.7.5

```
backend/datasets/export_contract.py    — Export schema + validation
backend/datasets/manifest.py           — Dataset manifest + fingerprint
backend/datasets/sharding.py           — Dataset sharding
backend/datasets/loader.py             — Dataset loading + v2.6 upgrade
backend/neural/dataloaders/__init__.py
backend/neural/dataloaders/base.py     — DatasetShardLoader
backend/neural/kaggle_hooks.py         — Optional GPU/Kaggle hooks
backend/neural/tiny_experiments.py     — Tiny-model datasets, training, checkpoints
backend/validation/failure_taxonomy.py — Stable failure categories
backend/validation/model_eval.py       — Evaluation harness + metrics
backend/validation/oracle_arena.py     — Oracle/model comparison runtime
scripts/generate_large_dataset.py      — Large resumable oracle export
scripts/train_tiny_model.py            — Functional tiny-model training
scripts/model_eval_runner.py           — Oracle-vs-model evaluation runner
scripts/eval_tiny_model.py             — Compatibility wrapper
scripts/generate_eval_report.py        — Compact report generation
scripts/compare_eval_reports.py        — Evaluation diffing
scripts/validate_checkpoint.py         — Checkpoint validation
scripts/migrate_checkpoint.py          — Checkpoint migration
scripts/snapshot_generator.py          — (v2.6) Snapshot generation
```

### Unchanged from v2.6

All core truth, traces, validation, benchmarks, and existing neural modules
remain backward-compatible.

---

## v2.7.5 Practical Workflow

1. Generate or refresh oracle data with `scripts/generate_large_dataset.py`.
2. Train a tiny baseline with `scripts/train_tiny_model.py`.
3. Evaluate with `scripts/model_eval_runner.py`.
4. Inspect `evaluation`, `arena`, and `failure_breakdown` in the JSON report.
5. Compare `by_layer`, `by_module`, and `by_category` for retention patterns.

## v2.7.6 Governance Addendum

The 2.7.6 layer adds:

- formal checkpoint schema validation
- safe migration of legacy checkpoint payloads
- artifact registry tracking for governed outputs
- compact JSON and Markdown evaluation reports
- deterministic evaluation diffs

### Determinism Guarantees

- identical seed + identical inputs + identical config -> identical export fingerprints
- evaluation reports are reproducible across runs
- checkpoint load/save preserves model and tokenizer state
- resume generation continues from the saved state file
