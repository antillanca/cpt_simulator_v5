import asyncio
import os
import sys
from backend.tooling.hermes import hermes_assistant

async def test_repair():
    mod_key = "energy_potential"
    prompt = (
        f"El módulo {mod_key} falló tras 5 intentos de aprendizaje autónomo.\n"
        "Por favor, analiza los logs recientes en training_orchestrator.log y "
        f"repara el archivo backend/ai/{mod_key}.py para que sea físicamente consistente."
    )
    
    # Simular la aprobación
    os.environ["CPT_HERMES_HUMAN_APPROVAL"] = "1"
    
    print(f"Invocando Hermes para {mod_key}...")
    result = await hermes_assistant.suggest_patch(prompt, target_path=f"backend/ai/{mod_key}.py")
    
    print(f"Allowed: {result.allowed}")
    print(f"Approved: {result.approved}")
    print(f"Message: {result.message}")
    print(f"STDOUT: {result.stdout[:500]}...")
    print(f"STDERR: {result.stderr[:500]}...")

if __name__ == "__main__":
    asyncio.run(test_repair())
