"""
Módulo Predictivo Tabular (Etapa 5.M) — Multimodular
Permite cargar múltiples filtros neuronales (Navegación, Lógica, etc.)
"""
import csv
from pathlib import Path

# Intentar importar torch, pero no fallar si no está
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "datasets"

if TORCH_AVAILABLE:
    class TabularNet(nn.Module):
        def __init__(self, input_dim=8):
            super().__init__()
            # Arquitectura idéntica a la del notebook
            self.net = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            return self.net(x)

class TabularFilter:
    def __init__(self):
        self.models = {} # dict: subject -> model
        self.mode = "exploration"
        
        # Dimensiones esperadas por materia (según el currículo y encoders)
        self.INPUT_DIMS = {
            "logic": 2,
            "counting": 2,
            "arithmetic": 4,
            "numeric": 4,
            "proportion": 4,
            "algebra": 4,
            "function": 4,
            "geometry": 4,
            "vector": 6,
            "time": 2,
            "action": 8,    # Planificador / Kinematics
            "newton": 8     # Dynamics
        }
        
        self.scan_models()

    def scan_models(self):
        """Escanea la carpeta models/ y carga todos los filtros disponibles."""
        if not MODELS_DIR.exists():
            return
            
        for pt_file in MODELS_DIR.glob("*_tabular_filter.pt"):
            subject = pt_file.name.replace("_tabular_filter.pt", "")
            dim = self.INPUT_DIMS.get(subject, 8) # Default 8 si es desconocido
            self.load_model(subject, pt_file, input_dim=dim)

    def load_model(self, subject: str, model_path: Path, input_dim: int):
        if TORCH_AVAILABLE and model_path.exists():
            try:
                model = TabularNet(input_dim=input_dim)
                model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                model.eval()
                self.models[subject] = model
                self.mode = "neural"
                print(f"DEBUG: Modelo '{subject}' cargado desde {model_path.name}")
            except Exception as e:
                print(f"DEBUG: Error al cargar modelo '{subject}': {e}")
        else:
            if not TORCH_AVAILABLE:
                print(f"DEBUG: Torch no disponible para cargar '{subject}'")

    def predict(self, features: list, subject: str = "action", threshold=0.60) -> tuple:
        if self.mode == "exploration" or subject not in self.models:
            return True, 1.0
        
        if TORCH_AVAILABLE:
            with torch.no_grad():
                prob = self.models[subject](torch.FloatTensor([features])).item()
            return prob >= threshold, prob
        return True, 1.0

    def record(self, features: list, success: bool, subject: str = "action"):
        data_path = DATA_DIR / f"{subject}_success_log.csv"
        exists = data_path.exists()
        input_dim = len(features)
        with open(data_path, "a", newline="") as f:
            w = csv.writer(f)
            if not exists: w.writerow([f"f{i}" for i in range(input_dim)] + ["success"])
            w.writerow([f"{v:.4f}" for v in features] + [int(success)])

# Singleton para uso global
tabular_filter = TabularFilter()
