# CPT v2.9B — Circuit Arena Results

## Model Info

- Type: edge_aware
- Parameters: 82305
- Epochs trained: 10
- Best epoch: 6
- Artifact fingerprint: c52215f18fc2086ec789021a777a861fa81415403e14398fc862ade315714e0b

## GNN

- In-distribution circuits: 453
- In-distribution MAE: 15.829487 V
- In-distribution RMSE: 17.145140 V
- In-distribution max error: 582.245789 V
- In-distribution KCL max violation: 3.98e-01
- In-distribution KVL max violation: 0.00e+00
- In-distribution replay consistency: 0.00e+00
- OOD circuits: 148
- OOD MAE: 225.712223 V
- OOD RMSE: 243.726417 V
- OOD max error: 6028.875977 V
- OOD KCL max violation: 7.69e+00
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

- Oracle mean solve: 0.047 ms
- Surrogate mean inference: 4.402 ms
- Speedup: 0.01x
- OOD speedup: 0.04x

## Reproducibility

- Dataset fingerprint: 3c5b08dbb43a21542849527da2e55593a0103e7b673bdaf7a07334a790a44b31
- Config fingerprint: 93873e5d241513676d1237a86f7b643e097c897dadb608f663201683246de671
- Snapshot hash: 0923381b65a84fd9bec0f61bd5c1300c7feba1499da021d4f39f397a8032eb34
- Snapshot fingerprint: 0923381b65a84fd9bec0f61bd5c1300c7feba1499da021d4f39f397a8032eb34
- Evaluation fingerprint: 3fb0cc4fd169e753fab217bc12a8a4dce46e95edc76a5606ac595543e819eebb
- Rerun validation: True
- Kaggle-ready: True

## Metadata

- Train count: 1809
- Eval count: 453
- OOD count: 148
- Dataset path: /home/john/www/cpt_simulator_v5/workspace/datasets/circuits/train_10k/circuits.jsonl
- OOD dataset path: /home/john/www/cpt_simulator_v5/workspace/datasets/circuits/ood_circuits.jsonl