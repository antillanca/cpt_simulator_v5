"""
State Encoder (Etapa 2) — Convierte estado físico a vector normalizado.
Sin embeddings lingüísticos.
"""
from simulation.physics_engine_wrapper import BOUNDS_X, BOUNDS_Y

STATE_DIM = 6
ACTION_DIM = 2
FULL_DIM = STATE_DIM + ACTION_DIM  # 8

def encode(state: dict) -> list:
    """state dict → [x, y, vx, vy, ax, ay] normalizado."""
    pos = state.get("position", [0, 0])
    vel = state.get("velocity", [0, 0])
    acc = state.get("acceleration", [0, 0])
    return [
        pos[0] / BOUNDS_X[1], pos[1] / BOUNDS_Y[1],
        vel[0] / 100.0, vel[1] / 100.0,
        acc[0] / 10.0, acc[1] / 10.0,
    ]

def encode_action(action: dict) -> list:
    return [action.get("vx", 0) / 100.0, action.get("vy", 0) / 100.0]
