import os
import re

planner_dir = "/home/john/www/cpt_simulator_v5/planner"
files = [f for f in os.listdir(planner_dir) if f.endswith(".py")]

# Pattern for the broken state (missing closing quote and bracket on the same line)
broken_pattern = re.compile(r'"source": \["print\(\'Training complete\. Model saved\.\'\)\n"\]\}')

for filename in files:
    path = os.path.join(planner_dir, filename)
    with open(path, "r") as f:
        content = f.read()
    
    # First, fix the broken ones
    if '"source": ["print(\'Training complete. Model saved.\')\n"]}' in content:
        print(f"Fixing broken {filename}...")
        new_content = content.replace('"source": ["print(\'Training complete. Model saved.\')\n"]}', '"source": ["print(\'Training complete. Model saved.\')\\n"]}')
        with open(path, "w") as f:
            f.write(new_content)
    
    # Also handle the multi-line ones that might have broken
    if '"    print(\'Model deployed.\')\n",' in content:
        print(f"Fixing broken multi-line in {filename}...")
        new_content = content.replace('"    print(\'Model deployed.\')\n",', '"    print(\'Model deployed.\')\\n",')
        with open(path, "w") as f:
            f.write(new_content)
