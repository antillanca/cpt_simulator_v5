import json
import re

with open('curriculum_capas_logicas_fisica_matematica_v_1.md', 'r') as f:
    lines = f.readlines()

out_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if line.strip() == "# FORMATO OFICIAL":
        # Skip the old official format section entirely
        while i < len(lines) and not lines[i].strip().startswith("# CAPA 0"):
            i += 1
        # Insert the new official format
        out_lines.append("# FORMATO OFICIAL (V2 Unificado)\n\n")
        out_lines.append("```json\n[\n  {\n")
        out_lines.append('    "id": "kinematics_constant_velocity",\n')
        out_lines.append('    "title": "Constant Velocity Motion",\n')
        out_lines.append('    "layer": 10,\n')
        out_lines.append('    "prerequisites": ["math_vectors_addition"],\n')
        out_lines.append('    "objective": "Move the particle to the right at a constant horizontal speed of 5.0.",\n')
        out_lines.append('    "target_state": {"vx": 5.0, "vy": 0.0},\n')
        out_lines.append('    "tolerance": 0.5,\n')
        out_lines.append('    "simulation_frames": 1,\n')
        out_lines.append('    "order": 100\n')
        out_lines.append("  }\n]\n```\n\n---\n\n")
        continue

    if line.strip() == "## Nodo":
        # Skip the node section
        while i < len(lines) and not lines[i].strip().startswith("---") and not lines[i].strip().startswith("## Ejercicio") and not lines[i].strip().startswith("# CAPA"):
            i += 1
        continue
        
    if line.strip() == "## Ejercicio":
        out_lines.append("## Ejercicio (Formato Unificado V2)\n\n")
        i += 1
        while lines[i].strip() != "```json":
            i += 1
        out_lines.append("```json\n")
        i += 1
        json_str = ""
        while lines[i].strip() != "```":
            json_str += lines[i]
            i += 1
        
        try:
            data = json.loads(json_str)
            new_data = {
                "id": data.get("id", "ex_001").lower(),
                "title": data.get("title", "Title in English"),
                "layer": data.get("layer", 0),
                "prerequisites": data.get("prerequisites", []),
                "objective": data.get("objective", "Objective in English"),
                "target_state": data.get("validation", {}).get("target_state", {}),
                "tolerance": data.get("validation", {}).get("tolerance", 0.5),
                "simulation_frames": data.get("validation", {}).get("max_steps", 60),
                "order": data.get("layer", 0) * 10
            }
            out_lines.append(json.dumps(new_data, indent=2) + "\n")
        except:
            out_lines.append(json_str)
            
        out_lines.append("```\n")
        i += 1
        continue

    out_lines.append(line)
    i += 1

with open('curriculum_capas_logicas_fisica_matematica_v_2.md', 'w') as f:
    f.writelines(out_lines)

print("Created V2 file.")
