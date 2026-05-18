# CPT v2.9B — Circuit Arena Results

## Model Info

- Type: edge_aware
- Parameters: 82113
- Epochs trained: 50
- Best epoch: 38
- Artifact fingerprint: 4b5b439133425d55d70f943a8221bec3069aacb18b95abf5e1e45b2993abbe45

## GNN

- In-distribution circuits: 1449
- In-distribution MAE: 5.018451 V
- In-distribution RMSE: 5.695050 V
- In-distribution max error: 480.952179 V
- In-distribution KCL max violation: 4.52e-01
- In-distribution KVL max violation: 1.05e+01
- In-distribution replay consistency: 0.00e+00
- OOD circuits: 1000
- OOD MAE: 107.310963 V
- OOD RMSE: 129.570154 V
- OOD max error: 11081.733398 V
- OOD KCL max violation: 2.85e+07
- OOD KVL max violation: 2.46e+03
- Deterministic rerun validation: True
- Replay consistency fingerprint match: True

## Baselines

- Mean baseline MAE: 9.429575 V
- Linear baseline MAE: 10.555431 V
- Random stable baseline MAE: 12.302528 V
- GNN beats mean: True
- GNN beats linear: True
- Mean baseline replay consistency: 0.00e+00
- Linear baseline replay consistency: 0.00e+00
- Random baseline replay consistency: 0.00e+00

## Oracle

- In-distribution circuits: 1449
- OOD circuits: 1000
- Deterministic rerun validation: True

## Speed

- Oracle mean solve: 0.043 ms
- Surrogate mean inference: 2.373 ms
- Speedup: 0.02x
- OOD speedup: 0.02x

## Reproducibility

- Dataset fingerprint: 3c5b08dbb43a21542849527da2e55593a0103e7b673bdaf7a07334a790a44b31
- Config fingerprint: df3c018ce72e6069db5aedf09595ca2f08f795031c48569111696331df25dd7c
- Snapshot hash: f13e8918cf17ef6772c223e696592515df170a8cf8a7a7c7c676ad2d375adcf0
- Snapshot fingerprint: f13e8918cf17ef6772c223e696592515df170a8cf8a7a7c7c676ad2d375adcf0
- Evaluation fingerprint: aa13f3f7834e494b084854ce807d43ca3c4ac5df6cd184333b991e13d4902d5e
- Rerun validation: True
- Kaggle-ready: True

## Metadata

- Train count: 5796
- Eval count: 1449
- OOD count: 1000
- Dataset path: /home/john/www/cpt_simulator_v5/workspace/datasets/circuits/train_10k.jsonl
- OOD dataset path: /home/john/www/cpt_simulator_v5/workspace/datasets/circuits/ood_circuits.jsonl