import json
import os
from pathlib import Path

# Config
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR
OUTPUT_FILE = BASE_DIR / "dpo_dataset_moe_final.jsonl"

EXPERTS = ["math", "physics", "advanced", "logic", "general"]

def main():
    print("🚀 Consolidando datasets para arquitectura MoE...")
    
    final_count = 0
    stats = {e: 0 for e in EXPERTS}
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for expert in EXPERTS:
            filename = f"dpo_dataset_{expert}.jsonl"
            filepath = DATASETS_DIR / filename
            
            if not filepath.exists():
                print(f"⚠️ Dataset no encontrado: {filename}")
                continue
                
            count = 0
            with open(filepath, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        
                        # Inject routing tag into the prompt
                        # This helps the model learn which internal expert should activate
                        routing_tag = f"[EXPERT: {expert.upper()}]"
                        if routing_tag not in data["prompt"]:
                            data["prompt"] = f"{routing_tag} {data['prompt']}"
                        
                        out_f.write(json.dumps(data) + "\n")
                        count += 1
                        stats[expert] += 1
                    except Exception as e:
                        print(f"❌ Error procesando línea en {filename}: {e}")
            
            print(f"✅ Procesado {filename}: {count} pares")
            final_count += count

    print("\n--- RESUMEN DE CONSOLIDACIÓN ---")
    for expert, count in stats.items():
        if count > 0:
            print(f"📊 Expert {expert.upper()}: {count} pares")
    print(f"✨ Total final: {final_count} pares en {OUTPUT_FILE.name}")
    print("--------------------------------")

if __name__ == "__main__":
    main()
