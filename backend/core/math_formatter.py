"""MathFormatter - Converts simulation state to mathematical vector notation.

All output in English (internal language). Translation happens at API boundary.
Bounds come from config, not hardcoded.
"""
from backend.config import CANVAS_WIDTH, CANVAS_HEIGHT


class MathFormatter:
    def __init__(self):
        self.canvas_width = CANVAS_WIDTH
        self.canvas_height = CANVAS_HEIGHT

    def format_state(self, state: dict) -> dict:
        """Convert raw state dict to mathematical expressions."""
        x = state.get("x", 0)
        y = state.get("y", 0)
        vx = state.get("vx", 0)
        vy = state.get("vy", 0)

        speed = (vx**2 + vy**2) ** 0.5

        return {
            "position_vector": f"P(t) = [{x:.2f}, {y:.2f}]",
            "velocity_vector": f"v(t) = [{vx:.2f}, {vy:.2f}]",
            "magnitude_v": f"|v| = {speed:.2f}",
            "algebraic": f"X:{x:.2f} + Y:{y:.2f}i",
            "constraints": {
                "in_bounds_x": 0 <= x <= self.canvas_width,
                "in_bounds_y": 0 <= y <= self.canvas_height,
            }
        }


math_formatter = MathFormatter()
