-- CPT Simulator v5 - Hardened Lua Sandbox Runner
-- Reads JSON from stdin: {particle: {...}, rule: "Lua code", frames: N, collect_trace: bool}
-- Executes the rule for N frames inside a restricted environment.

local input = io.read("*a")
local cjson = assert(require("cjson"), "cjson not available")
local data = assert(cjson.decode(input), "invalid JSON input")
local source_particle = assert(data.particle, "missing particle")
local rule_code = assert(data.rule, "missing rule")
local frames = tonumber(data.frames) or 1
local collect_trace = data.collect_trace and true or false

local function is_multi_particle_state(particle)
    return type(particle) == "table" and (particle.A ~= nil or particle.B ~= nil)
end

local function shallow_copy(src)
    local dst = {}
    if type(src) ~= "table" then
        return dst
    end
    for k, v in pairs(src) do
        dst[k] = v
    end
    return dst
end

local function normalized_particle(particle)
    local p = shallow_copy(particle)
    p.x = tonumber(p.x) or 0
    p.y = tonumber(p.y) or 0
    p.vx = tonumber(p.vx) or 0
    p.vy = tonumber(p.vy) or 0
    return p
end

local function normalized_particles(particles)
    local out = {}
    for name, particle in pairs(particles or {}) do
        if type(particle) == "table" then
            out[name] = normalized_particle(particle)
        else
            out[name] = particle
        end
    end
    return out
end

local sandbox_env = {
    particle = nil,
    particles = nil,
    math = {
        abs = math.abs, ceil = math.ceil, floor = math.floor,
        max = math.max, min = math.min, sqrt = math.sqrt,
        sin = math.sin, cos = math.cos, tan = math.tan,
        pi = math.pi, fmod = math.fmod, exp = math.exp, log = math.log,
        pow = math.pow, huge = math.huge,
        rad = math.rad, deg = math.deg,
        asin = math.asin, acos = math.acos, atan = math.atan,
    },
    string = {
        byte = string.byte,
        char = string.char,
        find = string.find,
        format = string.format,
        gmatch = string.gmatch,
        gsub = string.gsub,
        len = string.len,
        lower = string.lower,
        match = string.match,
        sub = string.sub,
        upper = string.upper,
    },
    table = {
        insert = table.insert,
        remove = table.remove,
        concat = table.concat,
        sort = table.sort,
        unpack = table.unpack or unpack,
    },
    tostring = tostring,
    tonumber = tonumber,
    type = type,
    pairs = pairs,
    ipairs = ipairs,
    next = next,
    select = select,
    assert = assert,
    error = error,
    pcall = pcall,
    xpcall = xpcall,
    rawequal = rawequal,
    rawget = rawget,
    rawset = rawset,
    print = function() end,
}

if is_multi_particle_state(source_particle) then
    sandbox_env.particles = normalized_particles(source_particle)
    sandbox_env.particle = sandbox_env.particles.A or normalized_particle({})
else
    sandbox_env.particle = normalized_particle(source_particle)
    sandbox_env.particles = { particle = sandbox_env.particle }
end

local function snapshot_state()
    if sandbox_env.particles then
        return normalized_particles(sandbox_env.particles)
    end
    return shallow_copy(sandbox_env.particle)
end

local function apply_integration()
    if sandbox_env.particles then
        for _, p in pairs(sandbox_env.particles) do
            if type(p) == "table" then
                p.x = (tonumber(p.x) or 0) + (tonumber(p.vx) or 0)
                p.y = (tonumber(p.y) or 0) + (tonumber(p.vy) or 0)
            end
        end
    else
        local p = sandbox_env.particle
        p.x = (tonumber(p.x) or 0) + (tonumber(p.vx) or 0)
        p.y = (tonumber(p.y) or 0) + (tonumber(p.vy) or 0)
    end
end

local ok, err = pcall(function()
    local code = rule_code
    if code:match("^%s*function%s*%(") then
        code = "_fn = " .. code
    end

    local chunk, compile_err = load(code, "user_rule", "t", sandbox_env)
    if not chunk then
        error("Failed to compile rule: " .. tostring(compile_err))
    end
    chunk()

    local fn = nil
    local candidates = {
        "_fn",
        "update_particle", "update", "step", "tick",
        "moveParticle", "updateParticle", "move_particle",
        "move",
    }
    for _, name in ipairs(candidates) do
        if type(sandbox_env[name]) == "function" then
            fn = sandbox_env[name]
            break
        end
    end

    local trace = collect_trace and {
        initial_state = snapshot_state(),
        steps = {},
    } or nil

    for i = 1, frames do
        local before = snapshot_state()
        if fn then
            local fn_ok, fn_err = pcall(fn, sandbox_env.particles or sandbox_env.particle)
            if not fn_ok then
                error("Rule runtime error: " .. tostring(fn_err))
            end
        end

        apply_integration()

        if trace then
            trace.steps[#trace.steps + 1] = {
                frame = i,
                before = before,
                after = snapshot_state(),
            }
        end
    end

    sandbox_env.trace = trace
end)

local result
if ok then
    result = {
        status = "ok",
        particle = sandbox_env.particle,
        particles = sandbox_env.particles,
    }
    if collect_trace then
        result.trace = sandbox_env.trace
    end
else
    result = {
        status = "error",
        message = tostring(err),
    }
end

print(cjson.encode(result))
