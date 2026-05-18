# CPT V2.10 — Projection-Distilled Surrogate Retraining

## Overview

Version 2.10 introduces **Physics-Aware Surrogate Retraining**: training the GNN
surrogate on **blended targets** that lie on the physical manifold, so the model
learns representations closer to the solution space the projection layer converges
to. The expected outcome is a significant reduction in projection iterations at
inference time (from ~20 to <5).

## Core Idea

The v2.9F surrogate is trained on oracle solutions (exact MNA). At inference, the
surrogate's raw predictions are far from the physical manifold, requiring many
projection iterations to satisfy KCL/KVL/Power invariants.

**v2.10 hypothesis**: If we train the surrogate on targets that already live on
(or near) the manifold, it learns a mapping closer to the physically-consistent
solution space, and projection needs fewer iterations.

### Blended Target Formula

```
blended = alpha * oracle + (1 - alpha) * projected
```

Where:
- `oracle` = exact MNA solution
- `projected` = result of applying PhysicsProjection to a perturbed version of oracle
- `alpha` = blending coefficient (default 0.2)

The perturbation uses Gaussian noise (sigma=1.5V) over oracle voltages, simulating
the kind of errors the surrogate typically makes. The projection then brings these
perturbed voltages back to the manifold, producing physically-consistent but
slightly-imperfect targets.

**alpha=1.0** → pure oracle training (baseline, same as v2.9F)
**alpha=0.0** → pure projected training (aggressive manifold distillation)
**alpha=0.2** → mostly oracle with 20% projected character (recommended starting point)

## New Files

| File | Purpose |
|------|---------|
| `scripts/generate_projected_targets.py` | Generate v2.10 blended-targets dataset from oracle JSONL |
| `backend/circuits/projection_effort.py` | Projection effort metrics: iterations, decay rate, correction distance |
| `scripts/run_ablation_alpha.py` | Sweep alpha values and compare projection effort |
| `tests/test_v210_projection_retraining.py` | Integration tests for v2.10 pipeline |

## Modified Files

| File | Change |
|------|--------|
| `scripts/train_circuit_gnn.py` | Added `--target-mode blended_projection`, `load_blended_training_data()`, oracle MAE tracking |
| `scripts/run_circuit_arena.py` | Multi-checkpoint comparison (`--checkpoints`, `--checkpoint-labels`), projection effort metrics, `--save-traces`, `_load_model_from_checkpoint()` |

## Pipeline

### Step 1: Generate Blended Targets

```bash
python scripts/generate_projected_targets.py \
    --input workspace/datasets/circuits/train_10k.jsonl \
    --output workspace/datasets/circuits/projected_targets_v210.jsonl \
    --alpha 0.2 --sigma 1.5 --seed 42
```

This produces `projected_targets_v210.jsonl` where each record contains:
- `oracle_solution`: exact MNA node voltages
- `perturbed_solution`: oracle + Gaussian noise
- `projected_solution`: PhysicsProjection(perturbed) result
- `blended_solution`: alpha * oracle + (1-alpha) * projected
- `residual_history`: per-step projection residuals
- `fingerprint`: SHA-256 of the record

### Step 2: Train with Blended Targets

```bash
python scripts/train_circuit_gnn.py \
    --target-mode blended_projection \
    --epochs 100 --lr 1e-3 --seed 42
```

The `--target-mode blended_projection` flag loads the v2.10 dataset and trains
against blended targets. The evaluate function tracks `oracle_mae` (MAE vs
oracle solution) alongside the blended-target loss.

### Step 3: Compare Checkpoints

```bash
python scripts/run_circuit_arena.py \
    --checkpoints workspace/checkpoints/circuit_gnn_v29f.pt workspace/checkpoints/circuit_gnn_v210.pt \
    --checkpoint-labels v29f v210 \
    --save-traces
```

This runs the full arena on both checkpoints, computes projection effort metrics,
saves per-circuit traces, and prints a comparison table.

### Step 4: Alpha Ablation

```bash
python scripts/run_ablation_alpha.py \
    --alphas 0.0 0.1 0.2 0.5 1.0 \
    --epochs 30 --max-circuits 1000
```

Trains a model for each alpha and reports oracle MAE, projection iterations, and
correction distance in a summary table.

## Projection Effort Metrics

New first-class metrics defined in `backend/circuits/projection_effort.py`:

| Metric | Description | Goal |
|--------|-------------|------|
| `mean_iterations` | Average projection steps to converge | < 5 |
| `median_iterations` | Median projection steps | < 3 |
| `p90_iterations` | 90th percentile projection steps | < 10 |
| `mean_residual_after_1_step` | KCL residual after first projection step | < v2.9F |
| `mean_correction_distance` | RMS correction applied by projection | Smaller = closer to manifold |
| `residual_decay_rate` | Geometric mean of step-to-step residual ratio | < 0.5 |
| `mean_raw_kcl_violation` | KCL violation before projection | Lower = better starting point |
| `mean_raw_kvl_violation` | KVL violation before projection | Lower = better starting point |

## Acceptance Criteria (v2.10)

1. **Projection iterations**: mean < 5 (down from ~20 in v2.9F)
2. **Invariant violations**: max KCL/KVL < v2.9F baseline
3. **Residual after 1 step**: reduced vs v2.9F
4. **Oracle MAE**: < 2x v2.9F baseline (trading some raw accuracy for manifold proximity)
5. **All existing tests pass**: `pytest tests/test_v29f_*.py tests/test_v210_*.py -v`

## Key Design Decisions

1. **Alpha=0.2 default**: Conservative blend that preserves most oracle accuracy
   while injecting manifold structure. The ablation study validates this choice.

2. **Perturbation sigma=1.5V**: Chosen to match typical surrogate error magnitudes.
   Too small → projected targets ≈ oracle (no benefit). Too large → projected
   targets diverge from meaningful solutions.

3. **Seed=42**: Mandatory for reproducibility. The Gaussian perturbation must be
   deterministic across runs.

4. **Projection config fixed**: g_virtual=1.0, steps=50, tolerance=1e-9. Same
   parameters used for target generation and evaluation to ensure consistency.

5. **Global Virtual Node unchanged**: The VirtualNodeProjection class from v2.9F
   is not modified. Only the training targets change.

6. **Backward compatibility**: `--target-mode oracle` (default) reproduces v2.9F
   behavior exactly. The arena's single-checkpoint mode is unchanged.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Oracle MAE regression | Alpha ablation study; cap at 2x baseline |
| Projected targets too noisy | Sigma=1.5V chosen empirically; can tune |
| Multi-checkpoint arena slow | Data loaded once, shared across checkpoints |
| Determinism break | Seed=42 enforced in all scripts |

## Version History

- **v2.9F**: Global Virtual Node projection (stable baseline)
- **v2.10**: Projection-distilled surrogate retraining (this version)
