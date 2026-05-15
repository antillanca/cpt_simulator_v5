--- physics_rules.lua - Compiled physics rules for CPT Simulator v5
--- Estos son bloques de fisica verificados que el sandbox ejecuta directamente
--- Cada regla es una funcion anonima: function(particle) ... end

local physics_rules = {}

-- ============================================================================
-- REGLA BASE: Seek (mover hacia target con velocidad constante)
-- Parametros: target_x, target_y, speed
-- No usa dt, el sandbox hace Euler despues (p = p + v)
-- ============================================================================
function physics_rules.seek(target_x, target_y, speed)
    local code = string.format([[
        local tx, ty, spd = %d, %d, %.2f
        local dx = tx - particle.x
        local dy = ty - particle.y
        local dist = math.sqrt(dx*dx + dy*dy)
        if dist > 1 then
            particle.vx = (dx / dist) * spd
            particle.vy = (dy / dist) * spd
        else
            particle.vx = 0
            particle.vy = 0
        end
    ]], target_x or 500, target_y or 250, speed or 2.0)
    return code
end

-- ============================================================================
-- REGLA: Gravedad basica
-- ============================================================================
function physics_rules.gravity(g)
    local code = string.format([[
        particle.vy = particle.vy + %.3f
    ]], g or 0.5)
    return code
end

-- ============================================================================
-- REGLA: Rebote en bordes
-- ============================================================================
function physics_rules.bounce(bounce_factor)
    local b = bounce_factor or 0.9
    return string.format([[
        if particle.x < 0 then particle.x = 0; particle.vx = math.abs(particle.vx) * %.2f end
        if particle.x > 800 then particle.x = 800; particle.vx = -math.abs(particle.vx) * %.2f end
        if particle.y < 0 then particle.y = 0; particle.vy = math.abs(particle.vy) * %.2f end
        if particle.y > 600 then particle.y = 600; particle.vy = -math.abs(particle.vy) * %.2f end
    ]], b, b, b, b)
end

-- ============================================================================
-- REGLA: Friccion (decaimiento de velocidad)
-- ============================================================================
function physics_rules.friction(factor)
    local f = factor or 0.95
    return string.format([[
        particle.vx = particle.vx * %.3f
        particle.vy = particle.vy * %.3f
    ]], f, f)
end

-- ============================================================================
-- REGLA COMPUESTA: Seek + Gravedad + Rebote
-- La regla "inteligente" que deberia alcanzar cualquier target
-- ============================================================================
function physics_rules.seek_with_physics(target_x, target_y, speed, gravity, bounce_f)
    local tx = target_x or 500
    local ty = target_y or 250
    local spd = speed or 2.0
    local g = gravity or 0.0
    local b = bounce_f or 0.9
    
    return string.format([[
        -- Seek behavior
        local dx = %d - particle.x
        local dy = %d - particle.y
        local dist = math.sqrt(dx*dx + dy*dy)
        if dist > 1 then
            particle.vx = (dx / dist) * %.2f
            particle.vy = (dy / dist) * %.2f
        else
            particle.vx = 0
            particle.vy = 0
        end
        
        -- Gravity
        particle.vy = particle.vy + %.3f
        
        -- Bounce
        if particle.x < 0 then particle.x = 0; particle.vx = math.abs(particle.vx) * %.2f end
        if particle.x > 800 then particle.x = 800; particle.vx = -math.abs(particle.vx) * %.2f end
        if particle.y < 0 then particle.y = 0; particle.vy = math.abs(particle.vy) * %.2f end
        if particle.y > 600 then particle.y = 600; particle.vy = -math.abs(particle.vy) * %.2f end
    ]], tx, ty, spd, spd, g, b, b, b)
end

-- ============================================================================
-- REGLA MINIMAL: Solo velocidad constante en X
-- La mas simple posible. Si esto no falla, la mecanica base funciona.
-- ============================================================================
function physics_rules.move_right(speed)
    return string.format("particle.vx = %.2f", speed or 2.0)
end

function physics_rules.move_left(speed)
    return string.format("particle.vx = -%.2f", speed or 2.0)
end

function physics_rules.move_down(speed)
    return string.format("particle.vy = %.2f", speed or 2.0)
end

function physics_rules.move_up(speed)
    return string.format("particle.vy = -%.2f", speed or 2.0)
end

-- ============================================================================
-- REGLA: Stop (frenar)
-- ============================================================================
function physics_rules.stop()
    return "particle.vx = 0; particle.vy = 0"
end

return physics_rules
