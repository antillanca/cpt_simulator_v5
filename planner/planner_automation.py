"""
Planner Automation (Etapa 6) — CPT Cognitive Engine v2
Generador de Notebooks DPO Factory.
"""
import json
import csv
from pathlib import Path
from simulation.physics_engine_wrapper import step
from world_model.state_encoder import encode, encode_action
try:
    from tabular_filter import tabular_filter
except ImportError:  # pragma: no cover - compatibility when imported as package
    from planner.tabular_filter import tabular_filter

# Configuración
DATA_PATH = Path("datasets/action_success_log.csv")
NOTEBOOK_PATH = Path("scripts/train_tabular_v2.ipynb")
TARGET_SAMPLES = 500

def collect_data(target_samples=500):
    current_samples = 0
    if DATA_PATH.exists():
        with open(DATA_PATH, "r") as f:
            current_samples = sum(1 for _ in f) - 1
            
    if current_samples >= target_samples:
        print(f"✅ Ya existen {current_samples} muestras en {DATA_PATH}")
        return

    print(f"🚀 Iniciando recolección de {target_samples} muestras (Exploración)")
    
    # Acciones base
    ACTIONS = [
        {"vx": 5, "vy": 0}, {"vx": -5, "vy": 0},
        {"vx": 0, "vy": 5}, {"vx": 0, "vy": -5}
    ]
    
    target_pos = [500, 300]
    initial_state = {"position": [400, 300], "velocity": [0, 0]}
    state = initial_state.copy()
    
    while current_samples < target_samples:
        for action in ACTIONS:
            # Simulamos 10 frames de una vez para ver trayectoria clara
            next_state = step(state, action, frames=10)
            
            # Evaluar si nos acercamos al objetivo (Causalidad)
            dist_before = ((state["position"][0]-target_pos[0])**2 + (state["position"][1]-target_pos[1])**2)**0.5
            dist_after = ((next_state["position"][0]-target_pos[0])**2 + (next_state["position"][1]-target_pos[1])**2)**0.5
            
            # Éxito si la distancia se reduce significativamente
            success = 1 if (dist_before - dist_after) > 0.5 else 0
            
            # Registrar
            features = encode(state) + encode_action(action)
            tabular_filter.record(features, bool(success), subject="action")
            
            state = next_state
            current_samples += 1
            
            if current_samples % 100 == 0:
                print(f"📊 Recolectadas {current_samples}/{target_samples} muestras...")
                
            if current_samples >= target_samples:
                break
            
            # Random reset si nos alejamos mucho o cada 50 pasos
            if dist_after > 800 or current_samples % 50 == 0:
                import random
                state = {"position": [random.randint(100,700), random.randint(100,500)], "velocity": [0, 0]}

def generate_notebook():
    with open(DATA_PATH, "r") as f:
        csv_data = f.read()
        
    notebook = {
     "cells": [
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": ["# 🧠 CPT Navigation Trainer (v2)\n", "Entrenamiento de intuición de movimiento."]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "import pandas as pd\n",
        "import io\n",
        "\n",
        "# 1. Datos Incrustados\n",
        "csv_data = \"\"\"" + csv_data + "\"\"\"\n",
        "df = pd.read_csv(io.StringIO(csv_data))\n",
        "print(f'Cargadas {len(df)} muestras')"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# 2. Definir Red Estándar CPT\n",
        "class TabularNet(nn.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "        self.net = nn.Sequential(\n",
        "            nn.Linear(8, 32),\n",
        "            nn.ReLU(),\n",
        "            nn.Linear(32, 16),\n",
        "            nn.ReLU(),\n",
        "            nn.Linear(16, 1),\n",
        "            nn.Sigmoid()\n",
        "        )\n",
        "    def forward(self, x):\n",
        "        return self.net(x)\n",
        "\n",
        "model = TabularNet()\n",
        "opt = optim.Adam(model.parameters(), lr=0.005)\n",
        "loss_fn = nn.BCELoss()"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# 3. Entrenamiento\n",
        "X = torch.tensor(df.drop('success', axis=1).values, dtype=torch.float32)\n",
        "y = torch.tensor(df['success'].values, dtype=torch.float32).view(-1, 1)\n",
        "\n",
        "for epoch in range(200):\n",
        "    opt.zero_grad()\n",
        "    outputs = model(X)\n",
        "    loss = loss_fn(outputs, y)\n",
        "    loss.backward()\n",
        "    opt.step()\n",
        "    if epoch % 50 == 0: print(f'Epoch {epoch}, Loss: {loss.item():.4f}')\n",
        "\n",
        "torch.save(model.state_dict(), 'action_tabular_filter.pt')\n",
        "print('✅ Modelo de Navegación Guardado')"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# 4. Descargar / Guardar\n",
        "import shutil, os\n",
        "output_file = 'action_tabular_filter.pt'\n",
        "if os.path.exists('/kaggle/working'):\n",
        "    print('Model deployed.')\n",
        "    print('✅ Modelo guardado en Kaggle Output')\n",
        "else:\n",
        "    try:\n",
        "        from google.colab import files\n",
        "        files.download(output_file)\n",
        "        print('✅ Modelo descargado desde Colab')\n",
        "    except ImportError:\n",
        "        print('✅ Modelo guardado localmente')\n"
       ]
      }
     ],
     "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}},
     "nbformat": 4, "nbformat_minor": 4
    }
    
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"📝 Notebook generado en: {NOTEBOOK_PATH}")
    
    # 5. Entrenar automáticamente en Kaggle
    from scripts.kaggle_trainer import KaggleTrainer
    trainer = KaggleTrainer()
    print("🚀 Iniciando entrenamiento automático en Kaggle...")
    success = trainer.train("action", str(NOTEBOOK_PATH))
    
    if success:
        from backend.notifier import notifier
        notifier.send("🧠 Filtro de Navegación (Acción) entrenado y desplegado exitosamente vía Kaggle.")
    else:
        from backend.notifier import notifier
        notifier.send("⚠️ Error en el entrenamiento de Kaggle para Navegación.")

if __name__ == "__main__":
    collect_data(TARGET_SAMPLES)
    generate_notebook()
