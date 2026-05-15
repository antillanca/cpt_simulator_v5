from planner.planner import plan
import json

# Estado inicial simulado
start_state = {
    "position": [100, 100],
    "velocity": [0, 0]
}

# Objetivo
target = {"x": 500, "y": 250}

print("🧪 Probando motores de lenguaje directamente...")
from backend.ai.symbol_engine import symbol_engine
from backend.ai.syntax_engine import syntax_engine

# Simular una trayectoria de acercamiento
p_start = [100, 100]
p_end = [500, 250]
symbols = symbol_engine.parse_path([p_start, p_end], target_pos=p_end)
intent = syntax_engine.compose(symbols)
print(f"🗨️  Símbolos: {symbols}")
print(f"🗨️  Intención (EN): \"{intent}\"")

# Simular una trayectoria con obstáculo
obs = [{"x": 300, "y": 175, "radius": 50}]
symbols_avoid = symbol_engine.parse_path([p_start, [300, 175], p_end], target_pos=p_end, obstacles=obs)
intent_avoid = syntax_engine.compose(symbols_avoid)
print(f"🗨️  Símbolos (Evitación): {symbols_avoid}")
print(f"🗨️  Intención (Evitación): \"{intent_avoid}\"")
