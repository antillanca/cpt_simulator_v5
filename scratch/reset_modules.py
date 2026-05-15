import json
from pathlib import Path

MODULES_FILE = Path("backend/core_truth/modules.json")

def reset_modules():
    if not MODULES_FILE.exists():
        print("Error: modules.json not found")
        return

    with open(MODULES_FILE, "r") as f:
        data = json.load(f)

    targets = [
        "energy_potential",
        "energy_conservation",
        "waves_oscillation",
        "waves_frequency_amplitude"
    ]

    for name in targets:
        if name in data["modules"]:
            print(f"Resetting {name}...")
            mod = data["modules"][name]
            mod["status"] = "pending"
            mod["lua_code"] = None
            mod["rejection_count"] = 0
            mod["rejection_reason"] = ""

    with open(MODULES_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    reset_modules()
