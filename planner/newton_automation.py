"""
Newton Automation (Etapa 8.N) — CPT Cognitive Engine v2
Generador de datos y entrenamiento para el Filtro Newtoniano (Colisiones).
"""
import json
import csv
import math
import random
from pathlib import Path
from simulation.physics_engine_wrapper import step
from backend.notifier import notifier

# Configuración
DATA_PATH = Path("datasets/newton_success_log.csv")
NOTEBOOK_PATH = Path("scripts/train_newton_v1.ipynb")
TARGET_SAMPLES = 1000

def collect_data(target_samples=1000):
    print(f"🍎 Recolectando {target_samples} muestras de Mecánica Newtoniana...")
    
    current_samples = 0
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        with open(DATA_PATH, "r") as f:
            current_samples = max(0, sum(1 for _ in f) - 1)
            
    if current_samples >= target_samples:
        print(f"✅ Ya existen {current_samples} muestras en {DATA_PATH}")
        return

    # Meta para la partícula B
    TARGET_B = {"x": 600, "y": 400}

    with open(DATA_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if current_samples == 0:
            w.writerow(["dx_ab", "dy_ab", "vx_a", "vy_a", "dx_bt", "dy_bt", "vx_b", "vy_b", "success"])
            
        while current_samples < target_samples:
            # Posición inicial de B aleatoria
            pB = {"x": random.uniform(200, 600), "y": random.uniform(150, 450), "vx": 0, "vy": 0, "radius": 15}
            
            # Posición inicial de A aleatoria cerca de B
            pA = {
                "x": pB["x"] + random.uniform(-50, 50),
                "y": pB["y"] + random.uniform(-50, 50),
                "vx": 0, "vy": 0, "radius": 10
            }
            
            # Evitar solapamiento inicial
            dist_ab = math.hypot(pA["x"] - pB["x"], pA["y"] - pB["y"])
            if dist_ab < (pA["radius"] + pB["radius"]):
                continue
                
            state = {"A": pA, "B": pB}
            
            # Acción para A
            action = {"vx": random.choice([-10, -5, 0, 5, 10]), "vy": random.choice([-10, -5, 0, 5, 10])}
            
            # Simular 10 frames para ver el impacto
            dist_b_target_before = math.hypot(pB["x"] - TARGET_B["x"], pB["y"] - TARGET_B["y"])
            
            next_state = step(state, action, frames=10)
            pB_new = next_state["B"]
            
            dist_b_target_after = math.hypot(pB_new["x"] - TARGET_B["x"], pB_new["y"] - TARGET_B["y"])
            
            # Éxito si B se acerca a su meta tras la colisión
            success = 1 if (dist_b_target_before - dist_b_target_after) > 2.0 else 0
            
            # Features
            features = [
                pA["x"] - pB["x"], pA["y"] - pB["y"],
                action["vx"], action["vy"],
                pB["x"] - TARGET_B["x"], pB["y"] - TARGET_B["y"],
                pB["vx"], pB["vy"]
            ]
            
            w.writerow([f"{v:.2f}" for v in features] + [success])
            
            current_samples += 1
            if current_samples % 100 == 0:
                print(f"📊 Newton: {current_samples}/{target_samples} muestras...")

def generate_notebook():
    with open(DATA_PATH, "r") as f:
        csv_data = f.read()
        
    notebook = {
     "cells": [
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": ["# 🍎 CPT Newton Trainer (v1)\n", "Entrenamiento de Interacción y Colisiones (Transferencia de Momento)."]
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
        "torch.save(model.state_dict(), 'newton_tabular_filter.pt')\n",
        "print('✅ Modelo Newtoniano Guardado')"
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
        "output_file = 'newton_tabular_filter.pt'\n",
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
    success = trainer.train("newton", str(NOTEBOOK_PATH))
    
    if success:
        msg = (
            f"🍎 <b>CPT v2: Etapa 8 (Newton) Completada</b>\n\n"
            f"Filtro Newtoniano entrenado y desplegado exitosamente vía Kaggle."
        )
        notifier.send(msg)
    else:
        notifier.send("⚠️ Error en el entrenamiento de Kaggle para Newton.")
    print("🔔 Notificación enviada por Telegram.")

if __name__ == "__main__":
    collect_data(TARGET_SAMPLES)
    generate_notebook()
