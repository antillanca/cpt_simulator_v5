import asyncio
import json
import logging
import os
from pathlib import Path
from backend.ai.dpo_pipeline import dpo_pipeline
from backend.ai.student_engine import student, EXPERT_MAPPING
from backend.config import DEFAULT_STATE
from backend.notifier import notifier

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FAILURES_FILE = Path(__file__).parent.parent / "backend" / "ai" / "failed_modules.json"
SUCCESS_FILE = Path(__file__).parent.parent / "backend" / "ai" / "success_counts.json"

def load_failures():
    if FAILURES_FILE.exists():
        try: return json.loads(FAILURES_FILE.read_text())
        except: return {}
    return {}

def save_failures(data):
    FAILURES_FILE.write_text(json.dumps(data, indent=2))

def load_successes():
    if SUCCESS_FILE.exists():
        try: return json.loads(SUCCESS_FILE.read_text())
        except: return {}
    return {}

def save_successes(data):
    SUCCESS_FILE.write_text(json.dumps(data, indent=2))

HINTS_CATALOG = {
    "physics": [
        "Apply air friction: particle.vx = particle.vx * 0.98",
        "Always clamp coordinates to [0, 800] and [0, 600]",
        "Avoid velocities > 100 to prevent sandbox boundary violations",
        "CRITICAL: The sandbox does NOT have a global 'time' variable. Track it manually inside particle (e.g., particle.time = (particle.time or 0) + 1)",
        "CRITICAL: If asked to calculate based on an initial state (like y=100), explicitly set particle.y = 100 in your code first.",
        "For oscillations, use 'function update_particle(particle)' and math.sin((particle.time or 0) * 0.1)",
        "Lorentz Force: particle.vy = particle.vy + (q * particle.vx * B). If B=2 and q=1, just use particle.vx * 2"
    ],
    "math": [
        "CRITICAL: Use math.rad(angle) OR angle * math.pi / 180 to convert degrees to radians for math.sin/cos.",
        "To calculate percentages of X: result = X * (percentage / 100)",
        "For vectors: particle.vx = magnitude * math.cos(math.rad(angle)), particle.vy = magnitude * math.sin(math.rad(angle))",
        "Always set values directly on particle (particle.vx = ..., particle.vy = ...)",
        "Keep results simple and direct"
    ],
    "advanced": [
        "Use damping to stabilize energy-heavy equations",
        "Check bounds explicitly: if particle.x < 0 then particle.x = 0 end",
        "Apply a strong friction factor (0.90) to ensure the particle stays on screen",
        "For probability: if math.random(1,100) <= 50 then ... end",
        "Expansion: v = H0 * d. If H0=70 and d=particle.x, then particle.vx = 70 * particle.x"
    ]
}

async def main():
    logger.info("Starting DPO Synthetic Data Factory (MoE Edition)...")
    notifier.send("🚀 <b>CPT Simulator</b>: Iniciando DPO Factory (Ciclo Persistente)")
    
    while True:
        # Reload modules and failures each cycle
        from backend.ai.student_engine import load_modules
        modules_data = load_modules()
        modules = modules_data.get("modules", {})
        failures = load_failures()
        successes = load_successes()
        
        target_modules = []
        for name, mod in modules.items():
            # Filtro de Optimización: Solo procesar lógica
            if mod.get("subject") != "logic":
                continue
                
            status = mod.get("status")
            if status == "pending":
                target_modules.append((name, mod))
            elif status == "confirmed":
                # Limitar a máximo 2 pares para los módulos que ya dominamos
                if successes.get(name, 0) < 2:
                    target_modules.append((name, mod))
                    
        target_modules = sorted(target_modules, key=lambda x: x[1].get("level", 999))
        
        if not target_modules:
            logger.info("No modules found to process. Waiting 30m...")
            await asyncio.sleep(1800)
            continue

        logger.info(f"Cycle Start: {len(target_modules)} modules to process (including confirmed).")
        
        for module_name, module in target_modules:
            objective = module.get("description", "")
            target_state = module.get("target_state", {})
            level = module.get("level", 0)
            subject = module.get("subject", "general")
            
            expert_cat = EXPERT_MAPPING.get(subject, "general")
            dataset_path = f"dpo_dataset_{expert_cat}.jsonl"
            
            # Hint injection logic
            f_count = failures.get(module_name, 0)
            hints = []
            if f_count >= 2:
                hints = HINTS_CATALOG.get(expert_cat, HINTS_CATALOG["physics"])
                logger.info(f"Applying stability hints for {module_name} (Failures: {f_count})")

            logger.info(f"Processing Layer {level} [{subject}]: {module_name}")
            
            try:
                # 1. Onda Phase
                variations, full_prompt = await dpo_pipeline.generate_superposition(
                    module_name, objective, DEFAULT_STATE, num_variations=3, hints=hints
                )
                
                if not variations: 
                    failures[module_name] = f_count + 1
                    save_failures(failures)
                    continue
                    
                # 2. Collider Phase
                threshold = module.get("tolerance", 0.5)
                results = await dpo_pipeline.run_collider(
                    variations, target_state, DEFAULT_STATE, threshold=threshold
                )
                
                # 3. Collapse Phase
                old_path = dpo_pipeline.dataset_path
                dpo_pipeline.dataset_path = dataset_path
                entry = dpo_pipeline.collapse(results, full_prompt)
                dpo_pipeline.dataset_path = old_path
                
                if entry:
                    logger.info(f"✅ Success [{expert_cat}]: {module_name}")
                    notifier.send(f"✅ <b>Expert [{expert_cat}]</b>: Nuevo par DPO para <i>{module_name}</i>")
                    
                    # Track successes
                    successes[module_name] = successes.get(module_name, 0) + 1
                    save_successes(successes)
                    
                    # Clear failures on success
                    if module_name in failures: 
                        del failures[module_name]
                        save_failures(failures)
                else:
                    failures[module_name] = f_count + 1
                    save_failures(failures)
                    
            except Exception as e:
                logger.error(f"Error in DPO cycle for {module_name}: {e}")
                failures[module_name] = f_count + 1
                save_failures(failures)

        logger.info("Cycle complete. Cooling down for 5m...")
        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(main())
