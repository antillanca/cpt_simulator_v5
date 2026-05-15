"""
CPT Simulator v5 — Módulo: energy_potential
Nivel 12, Subject: energy
Motor: lua

Energía Potencial Gravitacional: Ep = m * g * h
Dado: m=2.0, g=9.81, h=particle.y=10
Almacena el resultado en particle.x.

Código Lua correcto, verificado en sandbox Docker:
  particle.x = 196.2 (target = 196.2 ± 1.0)
"""

ENERGY_POTENTIAL_LUA = """\
-- Energy Potential: Ep = m * g * h
-- Given: m=2.0, g=9.81, h=particle.y=10
-- Store result in particle.x
local mass = 2.0
local gravity = 9.81
local height = particle.y
local potential_energy = mass * gravity * height
particle.x = potential_energy"""

# Metadatos para inyección directa en modules.json
MODULE_META = {
    "module_name": "energy_potential",
    "level": 12,
    "subject": "energy",
    "description": "Compute gravitational potential energy. Given mass=2.0, gravity=9.81, and height=particle.y=10, store m*g*h in particle.x.",
    "lua_code": ENERGY_POTENTIAL_LUA,
    "target_state": {"x": 196.2},
    "tolerance": 1.0,
    "simulation_frames": 1,
}


def get_lua_code() -> str:
    """Return the verified Lua code for energy_potential."""
    return ENERGY_POTENTIAL_LUA
