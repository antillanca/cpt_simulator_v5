import os
import re

planner_dir = "/home/john/www/cpt_simulator_v5/planner"
files = [f for f in os.listdir(planner_dir) if f.endswith(".py")]

pattern = re.compile(r'"import shutil, os\\nif os\.path\.exists\(\'/kaggle/working\'\): shutil\.copy\(\'.*?\.pt\', \'/kaggle/working/.*?\.pt\'\)\\n"')
replacement = '"print(\'Training complete. Model saved.\')\\n"'

for filename in files:
    path = os.path.join(planner_dir, filename)
    with open(path, "r") as f:
        content = f.read()
    
    if pattern.search(content):
        print(f"Fixing {filename}...")
        new_content = pattern.sub(replacement, content)
        with open(path, "w") as f:
            f.write(new_content)
    else:
        # Also check for the multi-line version
        multi_pattern = re.compile(r'\"    shutil\.copy\(output_file, \'/kaggle/working/\' \+ output_file\)\\n\",')
        if multi_pattern.search(content):
            print(f"Fixing multi-line in {filename}...")
            new_content = multi_pattern.sub('"    print(\'Model deployed.\')\\n",', content)
            with open(path, "w") as f:
                f.write(new_content)
