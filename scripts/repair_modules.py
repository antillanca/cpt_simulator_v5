import json
from pathlib import Path

def repair_modules():
    path = Path("backend/core_truth/modules.json")
    with open(path, "r") as f:
        data = json.load(f)
        
    modules = data.get("modules", {})
    repaired_count = 0
    
    for key, mod in modules.items():
        status = mod.get("status")
        lua_code = mod.get("lua_code", "")
        rejections = mod.get("rejection_count", 0)
        
        if status == "confirmed":
            if not lua_code.strip():
                print(f"Reparando {key} (Nivel {mod.get('level')}): Código Lua vacío.")
                mod["status"] = "pending"
                repaired_count += 1
            elif rejections > 0:
                print(f"Reparando {key} (Nivel {mod.get('level')}): {rejections} rechazos.")
                mod["status"] = "pending"
                repaired_count += 1
                
    if repaired_count > 0:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n✅ Reparados {repaired_count} módulos. Se han devuelto a estado 'pending'.")
    else:
        print("\n✅ Ningún módulo necesitaba reparación.")

if __name__ == "__main__":
    repair_modules()
