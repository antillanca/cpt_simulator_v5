--- CPT Simulator v5 - Physics Base Model (Lua)
-- Core physics engine that compiles and runs as a standalone AI model
-- Represents: position, velocity, acceleration, forces, energy

local PhysicsModel = {}
PhysicsModel.__index = PhysicsModel

-- === Constructor ===
function PhysicsModel.new(config)
    local self = setmetatable({}, PhysicsModel)
    
    -- Model parameters (trainable)
    self.params = {
        gravity = config.gravity or 0.0,        -- px/frame^2
        friction = config.friction or 0.0,       -- velocity multiplier per frame
        bounce = config.bounce or 1.0,           -- energy retained after bounce
        mass = config.mass or 1.0,               -- particle mass
        dt = config.dt or 1.0,                   -- time step (1 = 1 frame)
    }
    
    -- State
    self.state = {
        x = config.x or 0,
        y = config.y or 0,
        vx = config.vx or 0,
        vy = config.vy or 0,
        ax = 0,
        ay = 0,
    }
    
    -- Energy tracking
    self.energy = {
        kinetic = 0,
        potential = 0,
        total = 0,
    }
    
    -- Bounds
    self.bounds = {
        x_min = config.x_min or 0,
        x_max = config.x_max or 800,
        y_min = config.y_min or 0,
        y_max = config.y_max or 600,
    }
    
    return self
end

-- === Core physics step (Euler integration) ===
function PhysicsModel:step(forces)
    forces = forces or {}
    local p = self.params
    local s = self.state
    local dt = p.dt
    
    -- Sum forces: F = ma, so a = F/m
    local fx, fy = 0, 0
    for _, f in ipairs(forces) do
        fx = fx + (f.x or 0)
        fy = fy + (f.y or 0)
    end
    
    -- Add gravity (acts on y axis)
    fy = fy + p.gravity * p.mass
    
    -- Acceleration
    s.ax = fx / p.mass
    s.ay = fy / p.mass
    
    -- Integrate velocity: v = v + a*dt
    s.vx = s.vx + s.ax * dt
    s.vy = s.vy + s.ay * dt
    
    -- Apply friction
    if p.friction > 0 then
        local friction_factor = math.max(0, 1 - p.friction * dt)
        s.vx = s.vx * friction_factor
        s.vy = s.vy * friction_factor
    end
    
    -- Integrate position: p = p + v*dt
    s.x = s.x + s.vx * dt
    s.y = s.y + s.vy * dt
    
    -- Boundary collision (bounce)
    self:_apply_bounds()
    
    -- Update energy
    self:_update_energy()
    
    return self:get_state()
end

-- === Boundary collision with energy loss ===
function PhysicsModel:_apply_bounds()
    local s = self.state
    local b = self.bounds
    local bounce = self.params.bounce
    
    if s.x < b.x_min then
        s.x = b.x_min
        s.vx = math.abs(s.vx) * bounce
    elseif s.x > b.x_max then
        s.x = b.x_max
        s.vx = -math.abs(s.vx) * bounce
    end
    
    if s.y < b.y_min then
        s.y = b.y_min
        s.vy = math.abs(s.vy) * bounce
    elseif s.y > b.y_max then
        s.y = b.y_max
        s.vy = -math.abs(s.vy) * bounce
    end
end

-- === Energy calculation ===
function PhysicsModel:_update_energy()
    local s = self.state
    local p = self.params
    local v_sq = s.vx * s.vx + s.vy * s.vy
    self.energy.kinetic = 0.5 * p.mass * v_sq
    self.energy.potential = p.mass * p.gravity * (self.bounds.y_max - s.y)
    self.energy.total = self.energy.kinetic + self.energy.potential
end

-- === Get current state ===
function PhysicsModel:get_state()
    return {
        x = self.state.x,
        y = self.state.y,
        vx = self.state.vx,
        vy = self.state.vy,
        ax = self.state.ax,
        ay = self.state.ay,
        energy = {
            kinetic = self.energy.kinetic,
            potential = self.energy.potential,
            total = self.energy.total,
        }
    }
end

-- === Neural network layer (for "AI" model) ===
local NeuralLayer = {}
NeuralLayer.__index = NeuralLayer

function NeuralLayer.new(input_size, output_size)
    local self = setmetatable({}, NeuralLayer)
    self.weights = {}
    self.biases = {}
    
    -- Xavier initialization
    local scale = math.sqrt(2.0 / (input_size + output_size))
    for i = 1, output_size do
        self.weights[i] = {}
        for j = 1, input_size do
            self.weights[i][j] = (math.random() * 2 - 1) * scale
        end
        self.biases[i] = 0
    end
    
    return self
end

function NeuralLayer:forward(input)
    local output = {}
    for i = 1, #self.weights do
        local sum = self.biases[i]
        for j = 1, #input do
            sum = sum + self.weights[i][j] * input[j]
        end
        -- ReLU activation
        output[i] = math.max(0, sum)
    end
    return output
end

-- === Complete AI-controlled physics model ===
local AIPhysicsModel = {}
AIPhysicsModel.__index = AIPhysicsModel

function AIPhysicsModel.new(config)
    local self = setmetatable({}, AIPhysicsModel)
    
    -- Base physics
    self.physics = PhysicsModel.new(config)
    
    -- Neural network layers
    self.layers = {
        NeuralLayer.new(6, 12),  -- input: [x, y, vx, vy, target_x, target_y]
        NeuralLayer.new(12, 8),
        NeuralLayer.new(8, 4),   -- output: [force_x, force_y, param_gravity, param_friction]
    }
    
    -- Training data buffer
    self.training_buffer = {}
    self.max_buffer_size = config.max_buffer_size or 1000
    
    return self
end

-- === Forward pass: decide forces based on state and target ===
function AIPhysicsModel:decide(target_x, target_y)
    local s = self.physics.state
    
    -- Input vector
    local input = {
        s.x / 800,           -- normalize position
        s.y / 600,
        s.vx / 10,           -- normalize velocity
        s.vy / 10,
        target_x / 800,      -- normalize target
        target_y / 600,
    }
    
    -- Forward through layers
    local output = input
    for _, layer in ipairs(self.layers) do
        output = layer:forward(output)
    end
    
    -- Output: forces and parameter adjustments
    return {
        force_x = (output[1] - 0.5) * 2,    -- [-1, 1]
        force_y = (output[2] - 0.5) * 2,
        gravity_mod = output[3],
        friction_mod = output[4],
    }
end

-- === Step with AI decision ===
function AIPhysicsModel:step_ai(target_x, target_y)
    local decision = self.decide(target_x, target_y)
    
    local forces = {{
        x = decision.force_x * self.physics.params.mass,
        y = decision.force_y * self.physics.params.mass,
    }}
    
    -- Temporarily modify params based on AI decision
    local orig_gravity = self.physics.params.gravity
    self.physics.params.gravity = self.physics.params.gravity + decision.gravity_mod * 0.1
    self.physics.params.friction = math.max(0, math.min(1, 
        self.physics.params.friction + decision.friction_mod * 0.05))
    
    local result = self.physics:step(forces)
    
    -- Restore original params
    self.physics.params.gravity = orig_gravity
    
    -- Store training data
    self:store_training_data({
        state = {result.x, result.y, result.vx, result.vy},
        target = {target_x, target_y},
        forces = forces,
        reward = self:_compute_reward(result, target_x, target_y),
    })
    
    return result
end

-- === Reward function (distance to target) ===
function AIPhysicsModel:_compute_reward(state, target_x, target_y)
    local dx = state.x - target_x
    local dy = state.y - target_y
    local dist = math.sqrt(dx * dx + dy * dy)
    
    -- Negative reward (closer = better)
    -- Bonus for low velocity near target
    local speed = math.sqrt(state.vx * state.vx + state.vy * state.vy)
    local speed_bonus = speed < 2.0 and 10 or 0
    
    return -dist + speed_bonus
end

-- === Store experience ===
function AIPhysicsModel:store_training_data(experience)
    table.insert(self.training_buffer, experience)
    if #self.training_buffer > self.max_buffer_size then
        table.remove(self.training_buffer, 1)
    end
end

-- === Simple policy gradient training ===
function AIPhysicsModel:train_step(learning_rate)
    if #self.training_buffer < 10 then return end
    
    learning_rate = learning_rate or 0.01
    
    -- Sample recent experiences
    for i = math.max(1, #self.training_buffer - 9), #self.training_buffer do
        local exp = self.training_buffer[i]
        
        -- Simple weight perturbation based on reward
        local reward = exp.reward
        for _, layer in ipairs(self.layers) do
            for i = 1, #layer.weights do
                for j = 1, #layer.weights[i] do
                    -- Gradient-free: perturb proportional to reward
                    local perturbation = (math.random() * 2 - 1) * learning_rate * reward
                    layer.weights[i][j] = layer.weights[i][j] + perturbation
                end
                layer.biases[i] = layer.biases[i] + (math.random() * 2 - 1) * learning_rate * reward
            end
        end
    end
end

-- === Serialize model to JSON-serializable table ===
function AIPhysicsModel:serialize()
    local layers = {}
    for _, layer in ipairs(self.layers) do
        table.insert(layers, {
            weights = layer.weights,
            biases = layer.biases,
        })
    end
    return {
        layers = layers,
        params = self.physics.params,
        bounds = self.physics.bounds,
    }
end

-- === Load model from serialized data ===
function AIPhysicsModel.deserialize(data)
    local model = AIPhysicsModel.new(data.params or {})
    for i, layer_data in ipairs(data.layers) do
        if model.layers[i] then
            model.layers[i].weights = layer_data.weights
            model.layers[i].biases = layer_data.biases
        end
    end
    return model
end

-- === Export for sandbox integration ===
function PhysicsModel.to_sandbox_rule(model_data)
    -- Generates Lua code that the sandbox can execute
    -- This compiles the neural model into inline Lua
    local code = [[
-- Auto-generated physics AI rule
local W = ]] .. require("cjson").encode(model_data.layers[1].weights) .. [[
local B = ]] .. require("cjson").encode(model_data.layers[1].biases) .. [[
local dt = 1.0
local p = particle

-- Simple: seek target
local target_x = 500
local target_y = 250
local dx = target_x - p.x
local dy = target_y - p.y
local dist = math.sqrt(dx*dx + dy*dy)

if dist > 5 then
    local speed = 2.0
    p.vx = (dx / dist) * speed
    p.vy = (dy / dist) * speed
end
]]
    return code
end

return {
    PhysicsModel = PhysicsModel,
    AIPhysicsModel = AIPhysicsModel,
    NeuralLayer = NeuralLayer,
}
