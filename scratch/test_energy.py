import asyncio
from backend.ai.student_engine import load_modules
from backend.sandbox.sandbox_manager import sandbox_manager

def test_sandbox():
    rule = """
    particle.y = 10
    local m = 2.0
    local g = 9.81
    particle.x = m * g * particle.y
    """
    res = sandbox_manager.run_rule(rule, {"x":0,"y":0,"vx":0,"vy":0}, frames=1)
    print("Energy Potential Rule:", res)

    rule2 = """
    particle.time = (particle.time or 0) + 1
    particle.x = 50 * math.sin(particle.time)
    """
    res2 = sandbox_manager.run_rule(rule2, {"x":0,"y":0,"vx":0,"vy":0}, frames=15)
    print("Waves Oscillation Rule:", res2)

test_sandbox()
