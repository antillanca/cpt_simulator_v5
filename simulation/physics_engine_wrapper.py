"""
Motor de Simulación Física (Etapa 1) — CPT Cognitive Engine v2
Wrapper que reutiliza el Sandbox Docker de Lua existente.
"""
from backend.sandbox.sandbox_manager import sandbox_manager

BOUNDS_X = (0, 800)
BOUNDS_Y = (0, 600)

def step(state: dict, action: dict, frames: int = 1) -> dict:
    """Aplica una acción en un sistema multi-partícula usando el Sandbox Lua."""
    
    # 1. Preparar el estado inicial (Soporta A y B)
    # Si el estado es simple (v1), lo convertimos a multi (v2)
    pA = state.get("A", {
        "x": state.get("position", [0, 0])[0],
        "y": state.get("position", [0, 0])[1],
        "vx": state.get("velocity", [0, 0])[0],
        "vy": state.get("velocity", [0, 0])[1],
        "radius": 10
    })
    
    # Añadimos la acción a la partícula A (el Agente)
    pA["vx"] = pA.get("vx", 0) + action.get("vx", 0)
    pA["vy"] = pA.get("vy", 0) + action.get("vy", 0)
    
    # Partícula B (El Objeto) - Si no existe, la creamos en una posición estática por defecto
    pB = state.get("B", {
        "x": 300, "y": 200, "vx": 0, "vy": 0, "radius": 15
    })
    
    particles = {"A": pA, "B": pB}
    
    # 2. El código a ejecutar (Física Newtoniana: Colisiones Elásticas)
    lua_code = """
    -- 1. Actualizar posiciones
    for id, p in pairs(particles) do
        p.x = p.x + p.vx
        p.y = p.y + p.vy
    end
    
    -- 2. Detectar Colisión A <-> B
    local a = particles.A
    local b = particles.B
    local dx = a.x - b.x
    local dy = a.y - b.y
    local dist = math.sqrt(dx*dx + dy*dy)
    local minDist = a.radius + b.radius
    
    if dist < minDist then
        -- Colisión Elástica Simple (Intercambio de momento asumiendo masa=1)
        -- Normalizar vector de colisión
        local nx = dx / dist
        local ny = dy / dist
        
        -- Velocidad relativa
        local rvx = a.vx - b.vx
        local rvy = a.vy - b.vy
        
        -- Producto escalar (velocidad a lo largo de la normal)
        local velAlongNormal = rvx * nx + rvy * ny
        
        -- Solo resolver si se están acercando
        if velAlongNormal < 0 then
            local j = -2 * velAlongNormal
            local impulseX = j * nx
            local impulseY = j * ny
            
            a.vx = a.vx + impulseX
            a.vy = a.vy + impulseY
            b.vx = b.vx - impulseX
            b.vy = b.vy - impulseY
            
            a.collision = 1
            b.collision = 1
        end
    end
    
    -- 3. Obstáculo Estático (Límite Geométrico persistente)
    local obs_x, obs_y, obs_r = 400, 300, 50
    for id, p in pairs(particles) do
        local d_obs_x = p.x - obs_x
        local d_obs_y = p.y - obs_y
        if math.sqrt(d_obs_x*d_obs_x + d_obs_y*d_obs_y) <= obs_r then
            p.vx = 0
            p.vy = 0
        end
    end
    """
    
    # 3. Ejecutar en Docker
    result = sandbox_manager.run_rule(lua_code, particles, frames=frames)
    
    # 4. Formatear respuesta
    # Mantenemos compatibilidad con el planner v1 devolviendo la posición de A
    # Pero también incluimos el estado completo para el planner v2
    res_particles = result.get("particles", particles)
    pA_new = res_particles.get("A", pA)
    pB_new = res_particles.get("B", pB)
    
    return {
        "position": [pA_new.get("x", 0), pA_new.get("y", 0)],
        "velocity": [pA_new.get("vx", 0), pA_new.get("vy", 0)],
        "acceleration": [0, 0],
        "A": pA_new,
        "B": pB_new
    }
