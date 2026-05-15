"""
Trig Automation (Capa 9) — CPT Cognitive Engine v2
"""
import json, csv, math, random
from pathlib import Path

DATA_PATH = Path("datasets/trig_success_log.csv")
NOTEBOOK_PATH = Path("scripts/train_trig_v1.ipynb")

def collect_data(samples=1000):
    DATA_PATH.parent.mkdir(exist_ok=True)
    with open(DATA_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["angle", "sin", "cos", "valid", "success"])
        for _ in range(samples):
            angle = random.uniform(0, 360)
            s = math.sin(math.radians(angle))
            c = math.cos(math.radians(angle))
            valid = 1 if (s*s + c*c) > 0.99 else 0
            success = valid
            w.writerow([angle, s, c, valid, success])

def generate_notebook():
    with open(DATA_PATH, "r") as f: csv_data = f.read()
    notebook = {
     "cells": [
      {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
       "source": [
        "import torch, torch.nn as nn, pandas as pd, io\n",
        "csv_data = \"\"\"" + csv_data + "\"\"\"\n",
        "df = pd.read_csv(io.StringIO(csv_data))\n",
        "X = torch.tensor(df[['angle', 'sin', 'cos', 'valid']].values, dtype=torch.float32)\n",
        "y = torch.tensor(df['success'].values, dtype=torch.float32).view(-1, 1)\n",
        "class TabularNet(nn.Module):\n",
        "    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())\n",
        "    def forward(self, x): return self.net(x)\n",
        "model = TabularNet(); opt = torch.optim.Adam(model.parameters(), lr=0.01)\n",
        "for e in range(200): opt.zero_grad(); torch.nn.BCELoss()(model(X), y).backward(); opt.step()\n",
        "torch.save(model.state_dict(), 'trig_tabular_filter.pt')\n"
       ]},
      {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
       "source": ["print('Training complete. Model saved.')\n"]}
     ],
     "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 4
    }
    with open(NOTEBOOK_PATH, "w") as f: json.dump(notebook, f, indent=1)
    from scripts.kaggle_trainer import KaggleTrainer
    KaggleTrainer().train("trig", str(NOTEBOOK_PATH))

if __name__ == "__main__":
    collect_data()
    generate_notebook()
