# CPT v2.9B — Circuit Arena Results

## Model Info

- Type: edge_aware
- Parameters: 83009
- Epochs trained: 10
- Best epoch: 9
- Artifact fingerprint: c9a2e76258e93678e37732af6a7cfd58adb626910b9770f4f18540d83736b260

## GNN

- In-distribution circuits: 453
- In-distribution MAE: 14.156767 V
- In-distribution RMSE: 15.348807 V
- In-distribution max error: 591.257629 V
- In-distribution KCL max violation: 1.63e-01
- In-distribution KVL max violation: 0.00e+00
- In-distribution replay consistency: 0.00e+00
- OOD circuits: 148
- OOD MAE: 225.900280 V
- OOD RMSE: 243.564474 V
- OOD max error: 6038.804199 V
- OOD KCL max violation: 1.10e+00
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

- Oracle mean solve: 0.040 ms
- Surrogate mean inference: 1.922 ms
- Speedup: 0.02x
- OOD speedup: 0.02x

## Reproducibility

- Dataset fingerprint: 3c5b08dbb43a21542849527da2e55593a0103e7b673bdaf7a07334a790a44b31
- Config fingerprint: 93873e5d241513676d1237a86f7b643e097c897dadb608f663201683246de671
- Snapshot hash: 4853a8aee995b58f656c7f20f16c9514e755f195bb4de52fffb1134352ab6475
- Snapshot fingerprint: 4853a8aee995b58f656c7f20f16c9514e755f195bb4de52fffb1134352ab6475
- Evaluation fingerprint: 4715f602278e3342a7a0568625aa88b1679a2450455c65a4382c9fb629e4a48e
- Rerun validation: True
- Kaggle-ready: True

## Metadata

- Train count: 1809
- Eval count: 453
- OOD count: 148
- Dataset path: /home/john/www/cpt_simulator_v5/workspace/datasets/circuits/train_10k/circuits.jsonl
- OOD dataset path: /home/john/www/cpt_simulator_v5/workspace/datasets/circuits/ood_circuits.jsonl