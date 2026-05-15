"""Sync curriculum/*.json → backend/core_truth/modules.json

Rules:
- Scans all JSON files in the curriculum/ directory.
- Already-confirmed modules in modules.json are PRESERVED.
- New nodes are added as 'pending'.
- Modules that are no longer in any tree file are NOT removed.
"""
import json
import os
import glob
from pathlib import Path

BASE_DIR = Path(__file__).parent
CURRICULUM_DIR = BASE_DIR / "curriculum"
MODULES_FILE = BASE_DIR / "backend" / "ai" / "modules.json"


def sync():
    if not CURRICULUM_DIR.exists() or not CURRICULUM_DIR.is_dir():
        print(f"Error: Curriculum directory not found at {CURRICULUM_DIR}")
        return

    # 1. Load all nodes from all JSON files in curriculum/
    all_nodes = []
    json_files = sorted(glob.glob(str(CURRICULUM_DIR / "*.json")))
    
    if not json_files:
        print(f"Warning: No JSON files found in {CURRICULUM_DIR}")
        return

    print(f"Found {len(json_files)} curriculum files:")
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                nodes = content if isinstance(content, list) else content.get("nodes", [])
                all_nodes.extend(nodes)
                print(f"  - {os.path.basename(file_path)}: {len(nodes)} nodes")
        except Exception as e:
            print(f"  - Error reading {os.path.basename(file_path)}: {e}")

    # 2. Load existing modules to preserve confirmed status
    existing_modules = {}
    if MODULES_FILE.exists():
        try:
            data = json.loads(MODULES_FILE.read_text())
            existing_modules = data.get("modules", {})
        except Exception as e:
            print(f"Error reading existing modules.json: {e}")

    # 3. Process nodes
    added = 0
    updated = 0
    preserved = 0

    # Sort all nodes by layer and order to maintain logical consistency
    all_nodes.sort(key=lambda x: (x.get("layer", 0), x.get("order", 999)))

    for node in all_nodes:
        node_id = node.get("id")
        if not node_id:
            continue

        if node_id in existing_modules and existing_modules[node_id].get("status") == "confirmed":
            # Preserve confirmed module — do not overwrite learning progress
            preserved += 1
            continue

        # Add or update as pending (or refresh description/prerequisites)
        is_new = node_id not in existing_modules
        # Infer subject from ID if not provided (e.g. "math_operations" -> "math")
        inferred_subject = node_id.split('_')[0] if node_id else "general"
        subject = node.get("category", node.get("subject", inferred_subject))
        
        existing_modules[node_id] = {
            "level": node.get("layer", 0),
            "subject": subject,
            "status": "pending",
            "description": node.get("objective", node.get("description", "")),
            "lua_code": existing_modules.get(node_id, {}).get("lua_code", ""),
            "uses": node.get("prerequisites", []),
            "target_state": node.get("target_state", {}),
            "tolerance": node.get("tolerance", 0.5),
            "simulation_frames": node.get("simulation_frames", 1),
            "order": node.get("order", 999),
        }
        
        if is_new:
            added += 1
        else:
            updated += 1

    # 4. Save results
    MODULES_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODULES_FILE.write_text(json.dumps({"modules": existing_modules}, indent=2))

    print(f"\nSync complete:")
    print(f"  {preserved} modules preserved (confirmed status kept)")
    print(f"  {added} new modules added")
    print(f"  {updated} pending modules updated")
    print(f"  {len(existing_modules)} total modules in registry")


if __name__ == "__main__":
    sync()
