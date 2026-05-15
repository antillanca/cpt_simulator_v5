"""LoRA Adapter - Physical Environment Tuner (NOT neural fine-tuning).

IMPORTANT: This module does NOT perform neural network fine-tuning.
It adjusts PHYSICAL CONSTANTS of the simulation environment (speed, gravity, friction)
based on how well the student's rules perform.

Think of it as a "physics dial tuner":
- If the student's rules consistently overshoot targets → increase friction
- If the student's rules are too slow → increase speed multiplier
- If the student struggles with falling → adjust gravity

The "Low-Rank Adaptation" math (W = W0 + A×B) is applied to the ENVIRONMENT PARAMETERS,
not to the LLM weights. The real learning happens in the Student Engine (Qwen3-0.6B)
and the confirmed Lua modules.

This is a "Physical Environment Tuner" — it makes the sandbox easier/harder
based on student performance, similar to how a teacher adjusts difficulty.
"""
import numpy as np

from backend.config import LORA_RANK, LORA_BASE_WEIGHTS, LORA_LEARNING_RATE_FACTOR


class LoraAdapter:
    """Low-Rank Adaptation simplified for parameter evolution.

    Instead of full LLM weights, we adapt the 'behavioral vector'
    of rules based on experimental feedback (speed, gravity, friction).
    W = W0 + A@B, where A and B are low-rank matrices.
    """

    def __init__(self, rank: int = None, base_weights: list = None):
        self.rank = rank or LORA_RANK
        self.base_weights = np.array(base_weights or LORA_BASE_WEIGHTS, dtype=np.float64)
        dim = len(self.base_weights)
        # LoRA decomposition: W = W0 + AB
        # A: (dim, rank), B: (rank, dim)
        self.A = np.random.randn(dim, self.rank) * 0.01
        self.B = np.zeros((self.rank, dim))
        self.adaptation_history = []

    def adapt(self, feedback_score: float, delta_vector: np.ndarray):
        """Adapt LoRA matrices based on experimental feedback.

        feedback_score: positive = good, negative = bad
        delta_vector: observed parameter change (same dim as base_weights)
        """
        learning_rate = LORA_LEARNING_RATE_FACTOR * feedback_score

        # Forward: prediction = A @ (B @ base_weights)
        z = self.B @ self.base_weights          # (rank,)
        prediction = self.A @ z                 # (dim,)
        error = delta_vector - prediction

        # Backward (manual gradients for W = A @ B, h = W @ x):
        # dL/dA = outer(error, z)               -> (dim, rank)
        dA = np.outer(error, z)
        # dL/dz = A.T @ error                    -> (rank,)
        dz = self.A.T @ error
        # dL/dB = outer(dz, base_weights)        -> (rank, dim)
        dB = np.outer(dz, self.base_weights)

        self.A += learning_rate * dA
        self.B += learning_rate * dB

        self.adaptation_history.append({
            "feedback": feedback_score,
            "error_norm": float(np.linalg.norm(error)),
        })

        # Keep only last 100 entries
        if len(self.adaptation_history) > 100:
            self.adaptation_history = self.adaptation_history[-100:]

    def get_adapted_params(self) -> np.ndarray:
        """Return adapted parameter multipliers (base + adaptation)."""
        adaptation = self.A @ self.B @ self.base_weights
        return self.base_weights + adaptation

    def get_suggestions(self) -> dict:
        """Get current adapted parameter suggestions for rule generation."""
        params = self.get_adapted_params()
        return {
            "speed_multiplier": float(params[0]),
            "gravity_value": float(params[1]),
            "friction_coefficient": float(params[2]),
        }

    def reset(self):
        """Reset adaptation to initial state."""
        dim = len(self.base_weights)
        self.A = np.random.randn(dim, self.rank) * 0.01
        self.B = np.zeros((self.rank, dim))
        self.adaptation_history = []


lora_adapter = LoraAdapter()