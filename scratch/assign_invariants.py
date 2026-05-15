import json

modules_path = "backend/core_truth/modules.json"

with open(modules_path, "r") as f:
    data = json.load(f)

if "energy_conservation" in data["modules"]:
    data["modules"]["energy_conservation"]["invariants"] = ["energy_conservation"]
    print("Added 'energy_conservation' invariant to 'energy_conservation' module.")
else:
    print("Module 'energy_conservation' not found.")

with open(modules_path, "w") as f:
    json.dump(data, f, indent=2)
