from backend.sandbox.sandbox_manager import sandbox_manager

particle = {"x": 400, "y": 300, "vx": 0, "vy": 0}
rule = "particle.vx = 5"
result = sandbox_manager.run_rule(rule, particle, frames=1)
print(result)
