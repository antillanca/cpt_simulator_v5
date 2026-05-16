import asyncio
import logging
import os
from backend.ai.student_engine import student
import backend.ai.student_engine as student_mod
import pytest

logging.basicConfig(level=logging.INFO)

pytestmark = pytest.mark.skipif(os.getenv("RUN_INTEGRATION_TESTS", "0") != "1", reason="integration probe requires explicit opt-in")

async def test():
    # Force Advanced Motor to use the cascaded LLMs (NVIDIA/OpenRouter)
    student_mod.USE_ADVANCED_MOTOR = True
    
    concept = "general_relativity_geodesic"
    objective = "Simulate the geodesic motion of a particle near a Schwarzschild black hole. Calculate the Christoffel symbols for r and phi, and update vx, vy using the geodesic equation."
    
    print(f"--- Testing Student with ADVANCED MOTOR: {concept} ---")
    lua_code = await student.generate_lua(concept, objective)
    
    if lua_code:
        print("\n--- Generated Lua Code (by Professor Engine) ---")
        print(lua_code)
    else:
        print("\nFailed to generate code.")

if __name__ == "__main__":
    asyncio.run(test())
