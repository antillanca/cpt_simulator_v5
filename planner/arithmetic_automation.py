"""
Arithmetic Automation (Capa 2) — CPT Cognitive Engine v2
"""
import json, csv, random
from pathlib import Path

DATA_PATH = Path("datasets/arithmetic_success_log.csv")
NOTEBOOK_PATH = Path("scripts/train_arithmetic_v1.ipynb")

def collect_data(samples=1000):
    DATA_PATH.parent.mkdir(exist_ok=True)
    with open(DATA_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "op", "result", "success"])
        for _ in range(samples):
            a, b = random.randint(0, 50), random.randint(1, 50)
            op = random.choice([0, 1, 2, 3]) # +, -, *, /
            if op == 0: res = a + b
            elif op == 1: res = a - b
            elif op == 2: res = a * b
            else: res = a / b
            success = 1 if res >= -100 and res <= 2500 else 0
            w.writerow([a, b, op, res, success])

def generate_notebook():
    with open(DATA_PATH, "r") as f: csv_data = f.read()
    notebook = {
     "cells": [
      {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
       "source": [
        "import torch, torch.nn as nn, pandas as pd, io\n",
        "csv_data = \"\"\"" + csv_data + "\"\"\"\n",
        "df = pd.read_csv(io.StringIO(csv_data))\n",
        "X = torch.tensor(df[['a', 'b', 'op', 'result']].values, dtype=torch.float32)\n",
        "y = torch.tensor(df['success'].values, dtype=torch.float32).view(-1, 1)\n",
        "class TabularNet(nn.Module):\n",
        "    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())\n",
        "    def forward(self, x): return self.net(x)\n",
        "model = TabularNet(); opt = torch.optim.Adam(model.parameters(), lr=0.01)\n",
        "for e in range(200): opt.zero_grad(); torch.nn.BCELoss()(model(X), y).backward(); opt.step()\n",
        "torch.save(model.state_dict(), 'arithmetic_tabular_filter.pt')\n"
       ]},
      {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
       "source": ["print('Training complete. Model saved.')\n"]}
     ],
     "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 4
    }
    with open(NOTEBOOK_PATH, "w") as f: json.dump(notebook, f, indent=1)
    from scripts.kaggle_trainer import KaggleTrainer
    KaggleTrainer().train("arithmetic", str(NOTEBOOK_PATH))

if __name__ == "__main__":
    collect_data()
    generate_notebook()
