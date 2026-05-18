# CPT v2.9B — Circuit Arena Results

## Model Info

- Type: edge_aware
- Parameters: 82817
- Epochs trained: 10
- Best epoch: 3
- Artifact fingerprint: 0df1c367e14d50c225f5b2fcc6fa17c60bceba44ba3ff8eea44b9d484958383c

## GNN

- In-distribution circuits: 453
- In-distribution MAE: 16.640638 V
- In-distribution RMSE: 17.990854 V
- In-distribution max error: 572.473145 V
- In-distribution KCL max violation: 3.02e-01
- In-distribution KVL max violation: 0.00e+00
- In-distribution replay consistency: 0.00e+00
- OOD circuits: 148
- OOD MAE: 226.442234 V
- OOD RMSE: 244.305053 V
- OOD max error: 6017.655762 V
- OOD KCL max violation: 4.82e+00
- OOD KVL max violation: 0.00e+00
- Deterministic rerun validation: True
- Replay consistency fingerprint match: True

## Per-Topology Family Breakdown

| Topology Family | In-Distribution Count | In-Distribution MAE (V) | In-Distribution KCL Max Violation (A) | OOD Count | OOD MAE (V) | OOD KCL Max Violation (A) |
|---|---|---|---|---|---|---|
| **Trivial** (Tree-like) | 0 | 0.0000 | 0.00e+00 | 0 | 0.0000 | 0.00e+00 |
| **Simple** (1 Cycle) | 131 | 23.5744 | 3.02e-01 | 32 | 827.0187 | 2.17e+00 |
| **Medium** (2-3 Cycles) | 257 | 16.7405 | 3.02e-01 | 87 | 59.1230 | 4.82e+00 |
| **Dense** (>3 Cycles) | 65 | 2.2718 | 6.60e-02 | 29 | 65.6947 | 1.03e+00 |

## Baselines

- Mean baseline MAE: 24.778842 V
- Linear baseline MAE: 25.240210 V
- Random stable baseline MAE: 21.916994 V
- GNN beats mean: True
- GNN beats linear: True
- Mean baseline replay consistency: 0.00e+00
- Linear baseline replay consistency: 0.00e+00
- Random baseline replay consistency: 0.00e+00

## Oracle

- In-distribution circuits: 453
- OOD circuits: 148
- Deterministic rerun validation: True

## Speed

- Oracle mean solve: 0.045 ms
- Surrogate mean inference: 3.407 ms
- Speedup: 0.01x
- OOD speedup: 0.04x

## Reproducibility

- Dataset fingerprint: 3c5b08dbb43a21542849527da2e55593a0103e7b673bdaf7a07334a790a44b31
- Config fingerprint: 93873e5d241513676d1237a86f7b643e097c897dadb608f663201683246de671
- Snapshot hash: 46b5a1891d785e074e703f36e43be709045c0d92b38a9f63dfb8f0fe601ba18b
- Snapshot fingerprint: 46b5a1891d785e074e703f36e43be709045c0d92b38a9f63dfb8f0fe601ba18b
- Evaluation fingerprint: ad89f0b2e3fa3fc3c6f1148989deca83e952c3f5726aaa6791dba1828fe10835
- Rerun validation: True
- Kaggle-ready: True

## Metadata

- Train count: 1809
- Eval count: 453
- OOD count: 148
- Dataset path: /home/john/www/cpt_simulator_v5/workspace/datasets/circuits/train_10k/circuits.jsonl
- OOD dataset path: /home/john/www/cpt_simulator_v5/workspace/datasets/circuits/ood_circuits.jsonl