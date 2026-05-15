import json
from pathlib import Path

def reset_curriculum():
    path = Path("backend/core_truth/modules.json")
    with open(path, "r") as f:
        data = json.load(f)
        
    modules = data.get("modules", {})
    for key, mod in modules.items():
        level = mod.get("level", 0)
        # Resetear niveles del 1 al 11 para disparar las nuevas fábricas DPO
        if 1 <= level <= 11:
            mod["status"] = "pending"
            
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print("🔄 Currículo (Niveles 1-11) reseteado a 'pending'. Listos para entrenamiento autónomo.")

if __name__ == "__main__":
    reset_curriculum()
