"""
Logic Automation (Etapa 5.L) — CPT Cognitive Engine v2
Generador de datos y entrenamiento para el Filtro Lógico (Leyes de Aristóteles).
"""
import json
from pathlib import Path
from simulation.physics_engine_wrapper import step
from planner.tabular_filter import tabular_filter

# Configuración
DATA_PATH = Path("datasets/logic_success_log.csv")
LOGIC_MODEL_PATH = Path("models/logic_tabular_filter.pt")

def collect_logic_data(samples=500):
    print(f"🏛️ Recolectando {samples} muestras de Lógica Aristotélica...")
    
    with open(DATA_PATH, "w") as f:
        f.write("f0,f1,success\n")
        f.flush()
        
        # 1. Escenario de No Contradicción
        for i in range(samples // 4):
            state = {"is_hot": 1, "is_cold": 1}
            from backend.sandbox.sandbox_manager import sandbox_manager
            lua_rule = "particle.is_hot = 1; particle.is_cold = 0"
            result = sandbox_manager.run_rule(lua_rule, state)
            p = result.get("particle", {})
            success = 1 if p.get("is_hot") == 1 and p.get("is_cold") == 0 else 0
            f.write(f"1,1,{success}\n")
            if i % 10 == 0: 
                f.flush()
                print(f"📊 Lógica (1/2): {i}/{samples//4} muestras...")
            
        # 2. Escenario de Causa Eficiente
        for i in range(samples // 4):
            state = {"force_applied": 0, "x": 100, "has_moved": 0}
            lua_rule = "if particle.force_applied == 1 then particle.x = particle.x + 10; particle.has_moved = 1 end"
            result = sandbox_manager.run_rule(lua_rule, state)
            p = result.get("particle", {})
            success = 1 if p.get("x") == 100 and p.get("has_moved") == 0 else 0
            f.write(f"0,100,{success}\n")
            if i % 10 == 0: 
                f.flush()
                print(f"📊 Lógica (2/2): {i}/{samples//4} muestras...")

    print(f"📊 Dataset lógico guardado en {DATA_PATH}")
    generate_logic_notebook()

def generate_logic_notebook():
    notebook_path = Path("scripts/train_logic_v1.ipynb")
    
    # Leer datos para incrustarlos
    with open(DATA_PATH, "r") as f:
        csv_data = f.read()

    notebook = {
     "cells": [
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": ["# 🏛️ CPT Logic Trainer (Aristotelian v1)\n", "Entrenamiento del Filtro Lógico."]
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
        "# 1. Datos Incrustados\n",
        "csv_data = \"\"\"" + csv_data + "\"\"\"\n",
        "df = pd.read_csv(io.StringIO(csv_data))\n",
        "print(f'Cargadas {len(df)} muestras lógicas')"
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
        "            nn.Linear(2, 32),\n",
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
        "optimizer = torch.optim.Adam(model.parameters(), lr=0.01)\n",
        "criterion = nn.BCELoss()"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# 3. Entrenar\n",
        "X = torch.tensor(df[['f0', 'f1']].values, dtype=torch.float32)\n",
        "y = torch.tensor(df['success'].values, dtype=torch.float32).view(-1, 1)\n",
        "\n",
        "for epoch in range(200):\n",
        "    optimizer.zero_grad()\n",
        "    outputs = model(X)\n",
        "    loss = criterion(outputs, y)\n",
        "    loss.backward()\n",
        "    optimizer.step()\n",
        "    if epoch % 50 == 0: print(f'Epoch {epoch}, Loss: {loss.item():.4f}')\n",
        "\n",
        "torch.save(model.state_dict(), 'logic_tabular_filter.pt')\n",
        "print('✅ Modelo Lógico Guardado')"
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
        "output_file = 'logic_tabular_filter.pt'\n",
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
    
    with open(notebook_path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"📝 Notebook lógico generado en {notebook_path}")
    
    # 5. Entrenar automáticamente en Kaggle
    from scripts.kaggle_trainer import KaggleTrainer
    trainer = KaggleTrainer()
    print("🚀 Iniciando entrenamiento automático en Kaggle...")
    success = trainer.train("logic", str(notebook_path))
    
    if success:
        from backend.notifier import notifier
        notifier.send("🧠 Filtro Lógico (Aristóteles) entrenado y desplegado exitosamente vía Kaggle.")
    else:
        from backend.notifier import notifier
        notifier.send("⚠️ Error en el entrenamiento de Kaggle para Lógica.")

if __name__ == "__main__":
    collect_logic_data()
