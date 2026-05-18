# CPT v2.9C First Surrogate Validation

## Experiment Goals

- Execute the first end-to-end surrogate validation run.
- Check determinism across repeated training and evaluation passes.
- Measure IID performance, OOD behavior, invariants, and latency.

## Dataset Details

- Dataset fingerprint: 3ff4dcb6e9d589e4e0d4cc7f4032a7426187bb56ddc23bf873cc7b783dffa9b9
- Train count: 408
- Eval count: 102
- OOD cases: 64

## Model

- Type: edge_aware
- Hidden dim: 64
- Parameters: 82113
- Artifact fingerprint: 05d11acd33873f94140d23ffb20db9547f3fba13328bd6be4ae979c971f19fea

## Training Metrics

| Epoch | Train Loss | Eval Loss | MAE V | RMSE V | Max Error V | LR |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1736.664311 | 1546.104899 | 8.153320 | 9.645974 | 519.158142 | 1.00e-06 |

## Determinism Verification
- Deterministic: True
- Metrics equal: True
- Run A fingerprint: b992ce9cc63cb630ce119e8e825ef346e0190cefa6312d4da12cc2738f2f2391
- Run B fingerprint: b992ce9cc63cb630ce119e8e825ef346e0190cefa6312d4da12cc2738f2f2391

## Baseline Comparisons
- GNN MAE: 8.153320 V
- Mean baseline MAE: 10.622278 V
- Linear baseline MAE: 12.103985 V
- Random stable baseline MAE: 14.200710 V
- GNN beats mean: True
- GNN beats linear: True

## IID Performance
- IID MAE: 8.153320 V
- IID RMSE: 9.645974 V
- IID max error: 519.158142 V
- IID KCL max violation: 1.01e-01
- IID KVL max violation: 2.00e+01

## OOD Performance
- OOD MAE: 158.752605 V
- OOD RMSE: 168.718830 V
- OOD max error: 6144.023926 V
- OOD KCL max violation: 4.89e+07
- OOD KVL max violation: 2.82e+03

## Failure Taxonomy Summary
- Dominant failure: conservation_drift
- OOD cases classified: 64
- Failure counts: {"conservation_drift": 35, "extreme_resistance_instability": 7, "node_aliasing": 6, "symmetry_failure": 16}

## Invariant Preservation
- IID KCL violation: 2.06e-01
- OOD KCL violation: 4.89e+07
- IID KVL violation: 2.00e+01
- OOD KVL violation: 2.38e+03
- IID power violation: 2.44e+00
- OOD power violation: 3.01e+09
- Replay max abs diff: 0.00e+00

## Speedup Metrics
- Oracle mean latency: 0.000064 s
- Oracle p95 latency: 0.000089 s
- GNN mean latency: 0.056731 s
- GNN p95 latency: 0.084863 s
- Speedup: 0.00x

## Known Weaknesses
- Dominant observed failure mode: conservation_drift.
- Baseline gaps should be interpreted conservatively on this dataset if the circuit topology is simple or low-variance.
- OOD behavior is only as strong as the sampled OOD generator; it does not prove general circuit validity.

## Recommended Next Steps
- Extend validation coverage to harder OOD topologies before claiming broad generalization.
- Compare the surrogate against more structured baselines if topology complexity increases.
- Keep retraining and rerun validation under identical fingerprints to guard against drift.

## Reproducibility
- Report fingerprint: 3beee0ba598c9960f71638c43f46ce39940e0d1968ba25c40728799696dd45f8
- Checkpoint fingerprint: 05d11acd33873f94140d23ffb20db9547f3fba13328bd6be4ae979c971f19fea
- Deterministic: True