import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.circuits.graph_dataset import dataset_to_graphs
from backend.circuits.topology_curriculum import determine_level, CurriculumLevel

def main():
    dataset_path = PROJECT_ROOT / "workspace" / "datasets" / "circuits" / "train_10k" / "circuits.jsonl"
    print(f"Loading {dataset_path}...")
    graphs = dataset_to_graphs(dataset_path)
    print(f"Loaded {len(graphs)} graphs.")
    
    counts = {level: 0 for level in CurriculumLevel}
    for g in graphs:
        lvl = determine_level(g)
        counts[lvl] += 1
        
    for lvl, count in counts.items():
        print(f"{lvl}: {count} ({count/len(graphs)*100:.2f}%)")

if __name__ == "__main__":
    main()
