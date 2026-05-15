import json

file_path = "backend/core_truth/modules.json"

with open(file_path, "r") as f:
    data = json.load(f)

# 1. Identity (A = A)
data["modules"]["layer_00_existence"] = {
    "level": 0,
    "subject": "logic",
    "status": "pending",
    "description": "Principle of Identity (A = A). Define the inherent identity of the particle. Set particle.essence to 1 and particle.is_self to 1.",
    "lua_code": "function layer_00_existence(args)\n    return 1\nend",
    "uses": [],
    "target_state": {"essence": 1, "is_self": 1},
    "tolerance": 0.1,
    "simulation_frames": 1,
    "order": 0
}

# 2. Non-contradiction
data["modules"]["logic_02_non_contradiction"] = {
    "level": 0,
    "subject": "logic",
    "status": "pending",
    "description": "Principle of Non-Contradiction. A thing cannot be and not be at the same time. Define a state where particle.is_hot is 1, and forcefully set particle.is_cold to 0.",
    "lua_code": "",
    "uses": ["layer_00_existence"],
    "target_state": {"is_hot": 1, "is_cold": 0},
    "tolerance": 0.1,
    "simulation_frames": 1,
    "order": 1
}

# 3. Excluded Middle
data["modules"]["logic_03_excluded_middle"] = {
    "level": 0,
    "subject": "logic",
    "status": "pending",
    "description": "Principle of Excluded Middle. Eliminate ambiguity. Set particle.state_defined to 1 and ensure particle.ambiguity is set to 0.",
    "lua_code": "",
    "uses": ["logic_02_non_contradiction"],
    "target_state": {"state_defined": 1, "ambiguity": 0},
    "tolerance": 0.1,
    "simulation_frames": 1,
    "order": 2
}

# 4. Potency to Act
data["modules"]["logic_04_potency_to_act"] = {
    "level": 0,
    "subject": "logic",
    "status": "pending",
    "description": "Actualize potentiality. Assume the particle has potential (set particle.potential_v = 5), actualize it by setting particle.vx to particle.potential_v, and then reduce particle.potential_v to 0.",
    "lua_code": "",
    "uses": ["logic_03_excluded_middle"],
    "target_state": {"vx": 5, "potential_v": 0},
    "tolerance": 0.1,
    "simulation_frames": 1,
    "order": 3
}

# 5. Efficient Cause
data["modules"]["logic_05_efficient_cause"] = {
    "level": 0,
    "subject": "logic",
    "status": "pending",
    "description": "Demonstrate causality. Assume a force is applied (set particle.force_applied = 1). As a direct effect, increase particle.x by 10 and set particle.has_moved to 1.",
    "lua_code": "",
    "uses": ["logic_04_potency_to_act"],
    "target_state": {"force_applied": 1, "has_moved": 1, "x": 310},
    "tolerance": 0.1,
    "simulation_frames": 1,
    "order": 4
}

with open(file_path, "w") as f:
    json.dump(data, f, indent=2)

print("✅ Módulos de Lógica Aristotélica inyectados exitosamente.")
