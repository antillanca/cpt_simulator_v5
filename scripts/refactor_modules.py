import json
from pathlib import Path

def refactor_modules():
    path = Path("backend/core_truth/modules.json")
    if not path.exists():
        print("Error: modules.json not found")
        return

    with open(path, "r") as f:
        data = json.load(f)

    modules = data.get("modules", {})
    for key, mod in modules.items():
        level = mod.get("level", 0)
        
        # Clasificar según el plan maestro
        if level <= 11:
            mod["engine_type"] = "tabular"
            # Mapear el nombre del filtro esperado
            subject = mod.get("subject", "unknown")
            mod["filter_file"] = f"{subject}_tabular_filter.pt"
        else:
            mod["engine_type"] = "lua"
            mod["filter_file"] = None

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print("✅ Refactorización de modules.json completada.")

if __name__ == "__main__":
    refactor_modules()
