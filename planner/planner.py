"""
Planner (Etapa 6.1) — CPT Cognitive Engine v2
Implementación de A* con Poda Neural.
"""
import math
import heapq
from simulation.physics_engine_wrapper import step
from world_model.state_encoder import encode, encode_action
try:
    from tabular_filter import tabular_filter
except ImportError:  # pragma: no cover - compatibility when imported as package
    from planner.tabular_filter import tabular_filter

ACTION_SPACE = [
    {"vx":  5, "vy":  0, "force_applied": 1}, {"vx": -5, "vy":  0, "force_applied": 1},
    {"vx":  0, "vy":  5, "force_applied": 1}, {"vx":  0, "vy": -5, "force_applied": 1},
    {"vx": 10, "vy":  0, "force_applied": 1}, {"vx":-10, "vy":  0, "force_applied": 1},
    {"vx":  0, "vy": 10, "force_applied": 1}, {"vx":  0, "vy":-10, "force_applied": 1},
    {"vx":  5, "vy":  5, "force_applied": 1}, {"vx": -5, "vy": -5, "force_applied": 1},
    {"vx":  0, "vy":  0, "force_applied": 0}, # Inercia
]

# --- Capas Cognitivas Superiores (Inyección) ---
try:
    from backend.ai.symbol_engine import symbol_engine
    from backend.ai.syntax_engine import syntax_engine
    COGNITION_AVAILABLE = True
except ImportError:
    COGNITION_AVAILABLE = False

OBSTACLES = [{"x": 400, "y": 300, "radius": 50}]


def _state_hash(state: dict, resolution=10, subject="A") -> tuple:
    # Si es v1, usamos position. Si es v2, usamos el subject (A o B)
    if subject in state:
        p = state[subject]
        pos = [p.get("x", 0), p.get("y", 0)]
        vel = [p.get("vx", 0), p.get("vy", 0)]
    else:
        pos = state.get("position", [0, 0])
        vel = state.get("velocity", [0, 0])
    return (round(pos[0]/resolution), round(pos[1]/resolution), round(vel[0]/resolution), round(vel[1]/resolution))

def _distance(state: dict, target: dict, subject="A") -> float:
    if subject in state:
        p = state[subject]
        pos = [p.get("x", 0), p.get("y", 0)]
    else:
        pos = state.get("position", [0, 0])
    return math.sqrt((pos[0]-target["x"])**2 + (pos[1]-target["y"])**2)

def plan(start_state: dict, target: dict, max_nodes=5000, target_subject="A"):
    """
    Planificador A* con Poda Neural Multimodular.
    target_subject: "A" para navegación propia, "B" para mover un objeto.
    """
    queue = []
    # (prioridad, contador, estado, g, camino)
    heapq.heappush(queue, (0, 0, start_state, 0, []))
    
    visited = {}
    nodes_visited = 0
    nodes_pruned = 0
    node_counter = 1

    dist_initial = _distance(start_state, target, subject=target_subject)
    print(f"🚀 Iniciando A* (Distancia: {dist_initial:.2f}, Modo: neural, Objetivo: {target_subject})")

    while queue:
        priority, _, state, g, path = heapq.heappop(queue)
        
        nodes_visited += 1
        dist = _distance(state, target, subject=target_subject)
        
        if dist < 20: # Margen de éxito
            print(f"✅ ¡Objetivo alcanzado! Pasos: {len(path)}")
            
            if COGNITION_AVAILABLE:
                # Extraer posiciones clave para el análisis simbólico
                p_start = start_state.get("position") or [start_state.get("A", {}).get("x", 0), start_state.get("A", {}).get("y", 0)]
                p_end = [target["x"], target["y"]]
                
                symbols = symbol_engine.parse_path([p_start, p_end], 
                                                 target_pos=p_end, 
                                                 obstacles=OBSTACLES if target_subject == "A" else None)
                intent = syntax_engine.compose(symbols)
                print(f"🗨️ Agent intent (EN): \"{intent}\"")
                
                # Integración con traducción conceptual (Opcional)
                try:
                    from backend.i18n import t
                    # Si el sistema i18n soportara traducción de oraciones dinámicas, se usaría aquí.
                    # Por ahora, el inglés es la Verdad Sintáctica.
                except ImportError:
                    pass

            return [a for _, a in path]

        if nodes_visited > max_nodes:
            break

        h_val = _state_hash(state, subject=target_subject)
        if h_val in visited and visited[h_val] <= g:
            continue
        visited[h_val] = g

        for action in ACTION_SPACE:
            # 1. Poda Lógica (Leyes de Aristóteles)
            logic_features = [action.get("force_applied", 0), state.get("x", 0)]
            is_logical, _ = tabular_filter.predict(logic_features, subject="logic", threshold=0.5)
            if not is_logical:
                nodes_pruned += 1
                continue

            # 2. Poda Geométrica (Espacio Físico)
            geom_pruned = False
            for obs in OBSTACLES:
                dx = state.get("position", [0,0])[0] - obs["x"]
                dy = state.get("position", [0,0])[1] - obs["y"]
                geom_features = [dx, dy, action.get("vx", 0), action.get("vy", 0)]
                is_safe, _ = tabular_filter.predict(geom_features, subject="geometry", threshold=0.5)
                if not is_safe:
                    geom_pruned = True
                    break
            
            if geom_pruned:
                nodes_pruned += 1
                continue

            # 4. Poda Newtoniana (Interacción con B)
            if target_subject == "B" and "A" in state and "B" in state:
                dx_ab = state["A"]["x"] - state["B"]["x"]
                dy_ab = state["A"]["y"] - state["B"]["y"]
                dx_bt = state["B"]["x"] - target["x"]
                dy_bt = state["B"]["y"] - target["y"]
                
                newton_features = [
                    dx_ab, dy_ab, action.get("vx", 0), action.get("vy", 0),
                    dx_bt, dy_bt, state["B"].get("vx", 0), state["B"].get("vy", 0)
                ]
                is_hit_useful, _ = tabular_filter.predict(newton_features, subject="newton", threshold=0.4)
                if not is_hit_useful:
                    nodes_pruned += 1
                    continue

            # 3. Poda de Acción (Navegación)
            # Solo si el objetivo es A. Si es B, la acción es experimental.
            if target_subject == "A":
                features = encode(state) + encode_action(action)
                is_promising, prob = tabular_filter.predict(features, subject="action")
                if not is_promising:
                    nodes_pruned += 1
                    continue

            next_state = step(state, action, frames=10)
            priority = g + 1 + (dist / 10.0)
            node_counter += 1
            heapq.heappush(queue, (priority, node_counter, next_state, g + 1, path + [(priority, action)]))

    print(f"⚠️ Búsqueda terminada sin éxito.")
    return None
