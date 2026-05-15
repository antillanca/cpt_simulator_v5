import json
import os

modules_path = "backend/core_truth/modules.json"

with open(modules_path, "r") as f:
    data = json.load(f)

for mod_key, mod in data["modules"].items():
    if "invariants" not in mod:
        mod["invariants"] = []

with open(modules_path, "w") as f:
    json.dump(data, f, indent=2)

print("Successfully added 'invariants': [] to all modules.")
