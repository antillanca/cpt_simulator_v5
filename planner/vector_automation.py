"""
Vector Automation (Capa 8) — CPT Cognitive Engine v2
"""
import json, csv, math, random
from pathlib import Path

DATA_PATH = Path("datasets/vector_success_log.csv")
NOTEBOOK_PATH = Path("scripts/train_vector_v1.ipynb")

def collect_data(samples=1000):
    DATA_PATH.parent.mkdir(exist_ok=True)
    with open(DATA_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vx", "vy", "mag", "angle", "tdx", "tdy", "success"])
        for _ in range(samples):
            mag = random.uniform(1, 10)
            angle = random.uniform(0, 360)
            vx = mag * math.cos(math.radians(angle))
            vy = mag * math.sin(math.radians(angle))
            tdx, tdy = random.uniform(-100, 100), random.uniform(-100, 100)
            # success if vx/vy points towards tdx/tdy (approx)
            dot = vx * tdx + vy * tdy
            success = 1 if dot > 0 else 0
            w.writerow([vx, vy, mag, angle, tdx, tdy, success])

def generate_notebook():
    with open(DATA_PATH, "r") as f: csv_data = f.read()
    notebook = {
     "cells": [
      {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
       "source": [
        "import torch, torch.nn as nn, pandas as pd, io\n",
        "csv_data = \"\"\"" + csv_data + "\"\"\"\n",
        "df = pd.read_csv(io.StringIO(csv_data))\n",
        "X = torch.tensor(df.drop('success', axis=1).values, dtype=torch.float32)\n",
        "y = torch.tensor(df['success'].values, dtype=torch.float32).view(-1, 1)\n",
        "class TabularNet(nn.Module):\n",
        "    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())\n",
        "    def forward(self, x): return self.net(x)\n",
        "model = TabularNet(); opt = torch.optim.Adam(model.parameters(), lr=0.01)\n",
        "for e in range(200): opt.zero_grad(); torch.nn.BCELoss()(model(X), y).backward(); opt.step()\n",
        "torch.save(model.state_dict(), 'vector_tabular_filter.pt')\n"
       ]},
      {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
       "source": ["print('Training complete. Model saved.')\n"]}
     ],
     "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 4
    }
    with open(NOTEBOOK_PATH, "w") as f: json.dump(notebook, f, indent=1)
    from scripts.kaggle_trainer import KaggleTrainer
    KaggleTrainer().train("vector", str(NOTEBOOK_PATH))

if __name__ == "__main__":
    collect_data()
    generate_notebook()
