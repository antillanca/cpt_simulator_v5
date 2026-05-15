"""
Counting Automation (Capa 1) — CPT Cognitive Engine v2
Generador de datos y entrenamiento para el Filtro de Conteo.
"""
import json
import csv
import random
from pathlib import Path

# Configuración
DATA_PATH = Path("datasets/counting_success_log.csv")
NOTEBOOK_PATH = Path("scripts/train_counting_v1.ipynb")
TARGET_SAMPLES = 1000

def collect_data(target_samples=1000):
    print(f"🔢 Recolectando {target_samples} muestras de Conteo Discreto...")
    
    DATA_PATH.parent.mkdir(exist_ok=True)
    with open(DATA_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["value", "delta", "success"])
        
        for _ in range(target_samples):
            v = random.randint(0, 100)
            d = random.choice([-1, 1])
            expected = v + d
            
            # Éxito si el incremento/decremento es correcto (lógica interna)
            # Para el filtro tabular, entrenamos al agente a reconocer si una operación de conteo fue exitosa.
            success = 1 if expected >= 0 else 0
            w.writerow([v, d, success])

def generate_notebook():
    with open(DATA_PATH, "r") as f:
        csv_data = f.read()
        
    notebook = {
     "cells": [
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": ["# 🔢 CPT Counting Trainer (v1)\n", "Entrenamiento de intuición de conteo."]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import pandas as pd\n",
        "import io\n",
        "\n",
        "csv_data = \"\"\"" + csv_data + "\"\"\"\n",
        "df = pd.read_csv(io.StringIO(csv_data))\n",
        "X = torch.tensor(df[['value', 'delta']].values, dtype=torch.float32)\n",
        "y = torch.tensor(df['success'].values, dtype=torch.float32).view(-1, 1)\n",
        "\n",
        "class TabularNet(nn.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())\n",
        "    def forward(self, x): return self.net(x)\n",
        "\n",
        "model = TabularNet()\n",
        "opt = torch.optim.Adam(model.parameters(), lr=0.01)\n",
        "for e in range(200):\n",
        "    opt.zero_grad(); loss = torch.nn.BCELoss()(model(X), y); loss.backward(); opt.step()\n",
        "\n",
        "torch.save(model.state_dict(), 'counting_tabular_filter.pt')\n"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "import shutil, os\n",
        "if os.path.exists('/kaggle/working'): shutil.copy('counting_tabular_filter.pt', '/kaggle/working/counting_tabular_filter.pt')\n"
       ]
      }
     ],
     "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}},
     "nbformat": 4, "nbformat_minor": 4
    }
    
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(notebook, f, indent=1)
    
    from scripts.kaggle_trainer import KaggleTrainer
    trainer = KaggleTrainer()
    trainer.train("counting", str(NOTEBOOK_PATH))

if __name__ == "__main__":
    collect_data(TARGET_SAMPLES)
    generate_notebook()
