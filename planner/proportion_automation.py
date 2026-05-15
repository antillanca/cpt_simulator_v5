"""
Proportion Automation (Capa 4) — CPT Cognitive Engine v2
"""
import json, csv, random
from pathlib import Path

DATA_PATH = Path("datasets/proportion_success_log.csv")
NOTEBOOK_PATH = Path("scripts/train_proportion_v1.ipynb")

def collect_data(samples=1000):
    DATA_PATH.parent.mkdir(exist_ok=True)
    with open(DATA_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "ratio", "scaled", "success"])
        for _ in range(samples):
            a, b = random.uniform(1, 100), random.uniform(1, 100)
            ratio = a / b
            scaled = a * 1.5
            success = 1 if ratio > 0 and scaled > a else 0
            w.writerow([a, b, ratio, scaled, success])

def generate_notebook():
    with open(DATA_PATH, "r") as f: csv_data = f.read()
    notebook = {
     "cells": [
      {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
       "source": [
        "import torch, torch.nn as nn, pandas as pd, io\n",
        "csv_data = \"\"\"" + csv_data + "\"\"\"\n",
        "df = pd.read_csv(io.StringIO(csv_data))\n",
        "X = torch.tensor(df[['a', 'b', 'ratio', 'scaled']].values, dtype=torch.float32)\n",
        "y = torch.tensor(df['success'].values, dtype=torch.float32).view(-1, 1)\n",
        "class TabularNet(nn.Module):\n",
        "    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())\n",
        "    def forward(self, x): return self.net(x)\n",
        "model = TabularNet(); opt = torch.optim.Adam(model.parameters(), lr=0.01)\n",
        "for e in range(200): opt.zero_grad(); torch.nn.BCELoss()(model(X), y).backward(); opt.step()\n",
        "torch.save(model.state_dict(), 'proportion_tabular_filter.pt')\n"
       ]},
      {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
       "source": ["print('Training complete. Model saved.')\n"]}
     ],
     "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 4
    }
    with open(NOTEBOOK_PATH, "w") as f: json.dump(notebook, f, indent=1)
    from scripts.kaggle_trainer import KaggleTrainer
    KaggleTrainer().train("proportion", str(NOTEBOOK_PATH))

if __name__ == "__main__":
    collect_data()
    generate_notebook()
