# CPT v2.9B — Circuit Arena Results

## Model Info

- Type: edge_aware
- Parameters: 82113
- Epochs trained: 10
- Best epoch: 9
- Artifact fingerprint: 725a09491bb13c5caf0ca00161184b84ce6e529457e1329baa723465c09efe24

## GNN

- In-distribution circuits: 453
- In-distribution MAE: 15.440447 V
- In-distribution RMSE: 16.640898 V
- In-distribution max error: 579.522156 V
- In-distribution KCL max violation: 2.75e-01
- In-distribution KVL max violation: 0.00e+00
- In-distribution replay consistency: 0.00e+00
- OOD circuits: 148
- OOD MAE: 225.559766 V
- OOD RMSE: 243.344899 V
- OOD max error: 6035.514648 V
- OOD KCL max violation: 4.84e+00
- OOD KVL max violation: 0.00e+00
- Deterministic rerun validation: True
- Replay consistency fingerprint match: True

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

- Oracle mean solve: 0.044 ms
- Surrogate mean inference: 2.405 ms
- Speedup: 0.02x
- OOD speedup: 0.03x

## Reproducibility

- Dataset fingerprint: 3c5b08dbb43a21542849527da2e55593a0103e7b673bdaf7a07334a790a44b31
- Config fingerprint: 93873e5d241513676d1237a86f7b643e097c897dadb608f663201683246de671
- Snapshot hash: e5809e77a8eb93895f23abf4c2c677e43d1a54b551e5fae403619d801afc1510
- Snapshot fingerprint: e5809e77a8eb93895f23abf4c2c677e43d1a54b551e5fae403619d801afc1510
- Evaluation fingerprint: 5311f8a84f4bdde689732830a3fc75afe3517a279d48327a512feeeb888c4003
- Rerun validation: True
- Kaggle-ready: True

## Metadata

- Train count: 1809
- Eval count: 453
- OOD count: 148
- Dataset path: /home/john/www/cpt_simulator_v5/workspace/datasets/circuits/train_10k/circuits.jsonl
- OOD dataset path: /home/john/www/cpt_simulator_v5/workspace/datasets/circuits/ood_circuits.jsonl