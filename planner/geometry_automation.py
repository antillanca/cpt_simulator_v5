"""
Geometry Automation (Etapa 7.G) — CPT Cognitive Engine v2
Generador de datos y entrenamiento para el Filtro Geométrico (Obstáculos).
"""
import json
import csv
import math
import random
from pathlib import Path
from simulation.physics_engine_wrapper import step
from backend.notifier import notifier

# Configuración
DATA_PATH = Path("datasets/geometry_success_log.csv")
NOTEBOOK_PATH = Path("scripts/train_geometry_v1.ipynb")
TARGET_SAMPLES = 1000

# Obstáculo estático de entrenamiento
OBSTACLE = {"x": 400, "y": 300, "radius": 50}

def collect_data(target_samples=1000):
    print(f"📐 Recolectando {target_samples} muestras de Geometría Espacial...")
    
    current_samples = 0
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        with open(DATA_PATH, "r") as f:
            current_samples = max(0, sum(1 for _ in f) - 1)
            
    if current_samples >= target_samples:
        print(f"✅ Ya existen {current_samples} muestras en {DATA_PATH}")
        return

    ACTIONS = [
        {"vx": 5, "vy": 0}, {"vx": -5, "vy": 0},
        {"vx": 0, "vy": 5}, {"vx": 0, "vy": -5},
        {"vx": 5, "vy": 5}, {"vx": -5, "vy": -5},
        {"vx": 5, "vy": -5}, {"vx": -5, "vy": 5}
    ]
    
    with open(DATA_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if current_samples == 0:
            w.writerow(["dx", "dy", "vx", "vy", "success"])
            
        while current_samples < target_samples:
            # Posición inicial aleatoria cerca del obstáculo
            state = {
                "position": [
                    random.uniform(OBSTACLE["x"] - 100, OBSTACLE["x"] + 100),
                    random.uniform(OBSTACLE["y"] - 100, OBSTACLE["y"] + 100)
                ],
                "velocity": [0, 0]
            }
            
            # Solo muestras que no empiecen dentro del obstáculo
            dist_start = math.hypot(state["position"][0] - OBSTACLE["x"], state["position"][1] - OBSTACLE["y"])
            if dist_start <= OBSTACLE["radius"]:
                continue
                
            action = random.choice(ACTIONS)
            
            # Simular 10 frames (1 zancada A*)
            next_state = step(state, action, frames=10)
            
            # Verificar colisión
            dist_end = math.hypot(next_state["position"][0] - OBSTACLE["x"], next_state["position"][1] - OBSTACLE["y"])
            
            # Éxito si no entra en el radio del obstáculo
            success = 1 if dist_end > OBSTACLE["radius"] else 0
            
            # Features relativas al obstáculo
            dx = state["position"][0] - OBSTACLE["x"]
            dy = state["position"][1] - OBSTACLE["y"]
            vx = action.get("vx", 0)
            vy = action.get("vy", 0)
            
            w.writerow([f"{dx:.2f}", f"{dy:.2f}", f"{vx:.2f}", f"{vy:.2f}", success])
            
            current_samples += 1
            if current_samples % 100 == 0:
                print(f"📊 Geometría: {current_samples}/{target_samples} muestras...")

def generate_notebook():
    with open(DATA_PATH, "r") as f:
        csv_data = f.read()
        
    notebook = {
     "cells": [
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": ["# 📐 CPT Geometry Trainer (v1)\n", "Entrenamiento de Percepción Espacial (Evitar obstáculos)."]
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
        "            nn.Linear(4, 32),\n",
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
        "torch.save(model.state_dict(), 'geometry_tabular_filter.pt')\n",
        "print('✅ Modelo Geométrico Guardado')"
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
        "output_file = 'geometry_tabular_filter.pt'\n",
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
    success = trainer.train("geometry", str(NOTEBOOK_PATH))
    
    if success:
        msg = (
            f"📐 <b>CPT v2: Etapa 7 (Geometría) Completada</b>\n\n"
            f"Filtro geométrico entrenado y desplegado exitosamente vía Kaggle."
        )
        notifier.send(msg)
    else:
        notifier.send("⚠️ Error en el entrenamiento de Kaggle para Geometría.")
    print("🔔 Notificación enviada por Telegram.")

if __name__ == "__main__":
    collect_data(TARGET_SAMPLES)
    generate_notebook()
