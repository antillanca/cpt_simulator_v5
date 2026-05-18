#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"
python scripts/train_circuit_gnn.py --config configs/training/kaggle_v29d.yaml
python scripts/run_circuit_arena.py --checkpoint workspace/checkpoints/circuit_gnn_v29d_full.pt --output-dir workspace/arena_results
python scripts/generate_v29d_report.py --config configs/training/kaggle_v29d.yaml
