import logging

logger = logging.getLogger(__name__)

import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

BASE_DIR = Path(__file__).parent.parent.parent

# Teacher uses the main Hermes model (owl-alpha via OpenRouter)
# Called through the existing tutor_engine infrastructure

TEACHER_MODEL = "openrouter/owl-alpha"


# =============================================================================
# Luenberger State Observer
# Estimates student knowledge state from observable outputs
# =============================================================================

class LuenbergerObserver:
    """Observer that estimates internal knowledge state from observable outputs.
    
    State vector x_hat estimates:
      [0] syntax    - understands Lua syntax
      [1] semantics - understands what code does
      [2] integration - can compose modules
      [3] debugging - can fix errors
      [4] creativity - can solve novel problems
    
    Observable outputs y:
      [0] compile_rate - % of code that compiles
      [1] test_pass_rate - % of tests passed
      [2] reuse_rate - % of correct module reuse
      [3] error_recovery - % of errors fixed on retry
      [4] novelty_score - % of novel solutions
    
    Update rule: x_hat_new = x_hat + L * (y_observed - y_predicted)
    """

    STATE_NAMES = ["syntax", "semantics", "integration", "debugging", "creativity"]
    OUTPUT_NAMES = ["compile_rate", "test_pass_rate", "reuse_rate", "error_recovery", "novelty_score"]

    def __init__(self, learning_rate: float = 0.3):
        self.n_states = len(self.STATE_NAMES)
        self.n_outputs = len(self.OUTPUT_NAMES)
        # Estimated knowledge state [0, 1] for each dimension
        self.x_hat = [0.1] * self.n_states  # Start with minimal knowledge
        # Observer gain - how fast to correct based on observations
        self.L = learning_rate
        # History for tracking progress
        self.history: List[dict] = []

    def predict_output(self) -> List[float]:
        """Predict observable outputs from current state estimate.
        Simple linear model: y_pred[i] = sum(C[i][j] * x_hat[j])
        """
        # Output matrix C (how each state contributes to each output)
        C = [
            [0.9, 0.1, 0.0, 0.0, 0.0],  # compile_rate ← syntax
            [0.1, 0.7, 0.1, 0.1, 0.0],  # test_pass_rate ← semantics
            [0.0, 0.2, 0.6, 0.1, 0.1],  # reuse_rate ← integration
            [0.0, 0.1, 0.1, 0.7, 0.1],  # error_recovery ← debugging
            [0.0, 0.1, 0.2, 0.1, 0.6],  # novelty_score ← creativity
        ]
        y_pred = []
        for i in range(self.n_outputs):
            val = sum(C[i][j] * self.x_hat[j] for j in range(self.n_states))
            y_pred.append(min(1.0, max(0.0, val)))
        return y_pred

    def update(self, y_observed: List[float]) -> List[float]:
        """Update state estimate based on observed outputs.
        x_hat_new = x_hat + L * (y_observed - y_predicted)
        """
        y_predicted = self.predict_output()

        for i in range(self.n_states):
            # Innovation: difference between observed and predicted
            innovation = 0
            for j in range(self.n_outputs):
                # Weight by how much this state affects this output
                weight = 1.0 / self.n_states
                innovation += weight * (y_observed[j] - y_predicted[j])

            # Update state estimate
            self.x_hat[i] += self.L * innovation
            # Clamp to [0, 1]
            self.x_hat[i] = min(1.0, max(0.0, self.x_hat[i]))

        # Record history
        self.history.append({
            "timestamp": time.time(),
            "observed": y_observed[:],
            "predicted": y_predicted[:],
            "state": self.x_hat[:],
        })

        return self.x_hat[:]

    def get_state(self) -> Dict[str, float]:
        """Get current state estimate as named dict."""
        return {name: self.x_hat[i] for i, name in enumerate(self.STATE_NAMES)}

    def get_weakest_area(self) -> Tuple[str, float]:
        """Find the weakest knowledge area."""
        min_idx = 0
        min_val = self.x_hat[0]
        for i in range(1, self.n_states):
            if self.x_hat[i] < min_val:
                min_val = self.x_hat[i]
                min_idx = i
        return self.STATE_NAMES[min_idx], min_val

    def get_strongest_area(self) -> Tuple[str, float]:
        """Find the strongest knowledge area."""
        max_idx = 0
        max_val = self.x_hat[0]
        for i in range(1, self.n_states):
            if self.x_hat[i] > max_val:
                max_val = self.x_hat[i]
                max_idx = i
        return self.STATE_NAMES[max_idx], max_val

    def ready_for_next_level(self, threshold: float = 0.6) -> Tuple[bool, str]:
        """Check if student is ready to advance to next level."""
        weakest, val = self.get_weakest_area()
        if val >= threshold:
            return True, "ready"
        return False, f"weak in {weakest} ({val:.2f})"

    def should_intervene(self, threshold: float = 0.3) -> Tuple[bool, str]:
        """Check if teacher should intervene (student stuck)."""
        weakest, val = self.get_weakest_area()
        if val < threshold:
            return True, f"stuck in {weakest} ({val:.2f})"
        return False, ""

    def get_recommendation(self) -> str:
        """Generate teaching recommendation based on state."""
        state = self.get_state()
        weakest, wval = self.get_weakest_area()
        strongest, sval = self.get_strongest_area()

        if wval < 0.3:
            return f"Focus on {weakest} — student is struggling ({wval:.2f})"
        elif wval < 0.6:
            return f"Strengthen {weakest} before advancing ({wval:.2f})"
        elif sval > 0.8:
            return f"Student excels at {strongest} — can mentor others"
        else:
            return "Balanced progress — continue current pace"

    def get_progress_report(self) -> dict:
        """Full progress report."""
        state = self.get_state()
        weakest, wval = self.get_weakest_area()
        strongest, sval = self.get_strongest_area()
        ready, reason = self.ready_for_next_level()
        intervene, int_reason = self.should_intervene()

        return {
            "state": state,
            "weakest": {"area": weakest, "value": wval},
            "strongest": {"area": strongest, "value": sval},
            "ready_for_next": ready,
            "reason": reason,
            "needs_intervention": intervene,
            "intervention_reason": int_reason,
            "recommendation": self.get_recommendation(),
            "history_length": len(self.history),
        }


class TeacherEngine:
    """Teacher that evaluates student work and guides learning.
    
    Responsibilities:
    1. Generate exercises based on curriculum
    2. Evaluate student's Lua code
    3. Confirm or reject modules
    4. Provide explanations when student fails
    5. Decide when to intervene vs let student struggle
    """

    def __init__(self):
        self.evaluation_count = 0
        self.confirmation_count = 0
        # Luenberger state observer for tracking student knowledge
        self.observer = LuenbergerObserver(learning_rate=0.3)
        # Track module-specific stats for observer input
        self.module_stats: Dict[str, dict] = {}

    def _compute_observer_outputs(self, module_name: str, passed: bool,
                                   score: float, error: str,
                                   uses_modules: List[str]) -> List[float]:
        """Compute observable outputs for the Luenberger observer.
        
        Returns: [compile_rate, test_pass_rate, reuse_rate, error_recovery, novelty_score]
        """
        stats = self.module_stats.get(module_name, {
            "attempts": 0, "compiles": 0, "passes": 0,
            "reuses": 0, "recoveries": 0, "novel": 0,
        })
        stats["attempts"] = stats.get("attempts", 0) + 1
        if passed:
            stats["compiles"] = stats.get("compiles", 0) + 1
            stats["passes"] = stats.get("passes", 0) + 1
        if not error or "compile" not in error.lower():
            stats["compiles"] = stats.get("compiles", 0) + 1
        if uses_modules and len(uses_modules) > 0:
            stats["reuses"] = stats.get("reuses", 0) + 1
        self.module_stats[module_name] = stats

        n = max(stats["attempts"], 1)
        return [
            min(1.0, stats.get("compiles", 0) / n),   # compile_rate
            score,                                      # test_pass_rate
            min(1.0, stats.get("reuses", 0) / n),      # reuse_rate
            1.0 if passed and n > 1 else 0.0,          # error_recovery
            0.5 if passed else 0.0,                    # novelty_score (simplified)
        ]

    async def generate_exercise(self, module_name: str, module_desc: str, level: int) -> dict:
        """Generate an exercise for a specific module using LLM (dynamic)."""
        # Try LLM first for dynamic exercise generation
        try:
            prompt = f"""You are a physics/math teacher. Create an exercise for:
Module: {module_name}
Description: {module_desc}
Level: {level}

Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{{
  "question": "A clear question",
  "expected_answer": <number or null>,
  "test_cases": [
    [<arg1>, <arg2>, ..., <expected_result>],
    ...
  ],
  "hints": ["hint1", "hint2"]
}}

For math modules (count, add, subtract, multiply, divide, fraction): test_cases are tuples like (a, b, expected).
For physics modules (move_to, velocity, acceleration, force, energy): test_cases can be dicts with "from", "to", "speed", "expected_vx", "expected_vy" or tuples like (displacement, time, expected_velocity).
Include 3-4 test cases per exercise."""

            from backend.ai.student_engine import ollama_generate
            response = await ollama_generate(prompt, timeout=30)
            if response:
                # Parse JSON from response
                import re, json as json_mod
                # Find JSON object in response
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    exercise = json_mod.loads(match.group())
                    if "question" in exercise and "test_cases" in exercise:
                        return exercise
        except Exception as e:
            logger.warning(f"[Teacher] LLM exercise generation failed: {e}")

        # Fallback to static exercises
        return self._static_exercise(module_name, module_desc, level)

    def _static_exercise(self, module_name: str, module_desc: str, level: int) -> dict:
        """Fallback static exercise generation."""
        if "multiply" in module_desc.lower() or "multiplication" in module_desc.lower():
            return {"question": "What is 4 * 3?", "expected_answer": 12, "test_cases": [(2, 3, 6), (4, 3, 12), (5, 5, 25), (0, 10, 0)], "hints": ["Multiplication is repeated addition"]}
        elif "count" in module_desc.lower():
            return {"question": "Count from 1 to 5", "expected_answer": 5, "test_cases": [(1, 1), (3, 3), (5, 5), (10, 10)], "hints": ["Use a loop"]}
        elif "add" in module_desc.lower():
            return {"question": "What is 3 + 5?", "expected_answer": 8, "test_cases": [(1, 2, 3), (3, 5, 8), (10, 20, 30), (0, 0, 0)], "hints": ["Addition combines two numbers"]}
        elif "move_to" in module_desc.lower() or "move" in module_name.lower():
            return {"question": "Move particle from (300,50) to (500,250) at speed 2", "expected_answer": {"vx": 1.41, "vy": 1.41}, "test_cases": [{"from": (300, 50), "to": (500, 250), "speed": 2, "expected_vx": 1.41, "expected_vy": 1.41}], "hints": ["Calculate direction vector", "Normalize it"]}
        elif "velocity" in module_desc.lower():
            return {"question": "A particle moves 100 pixels in 5 frames. What is its velocity?", "expected_answer": 20.0, "test_cases": [(100, 5, 20.0), (200, 10, 20.0)], "hints": ["velocity = displacement / time"]}
        else:
            return {"question": f"Implement: {module_desc}", "expected_answer": None, "test_cases": [], "hints": ["Think step by step"]}

    def evaluate_lua(self, lua_code: str, module_name: str, test_cases: list) -> tuple:
        """Evaluate student's Lua code.
        
        Returns: (passed, score, feedback)
        """
        self.evaluation_count += 1

        # Basic syntax check
        if not lua_code or len(lua_code.strip()) < 5:
            return False, 0.0, "Code is too short or empty"

        # Check for prohibited tokens
        prohibited = ["os.", "io.", "require", "loadfile", "dofile", "debug."]
        for token in prohibited:
            if token in lua_code.lower():
                return False, 0.0, f"Prohibited token: {token}"

        # Check for basic Lua structure
        has_function = "function" in lua_code.lower()
        has_local = "local" in lua_code.lower()
        if not has_function and not has_local:
            return False, 0.0, "Code should define a function or local variables"

        # Try to compile the Lua code in sandbox
        compile_ok, compile_err = self._check_lua_compiles(lua_code)
        if not compile_ok:
            return False, 0.0, f"Lua compile error: {compile_err}"

        # If no test cases, accept based on compile + structure
        if not test_cases:
            score = 0.8 if has_function else 0.6
            return True, score, "Code compiles and has correct structure"

        # Run test cases
        passed_count = 0
        errors = []
        for tc in test_cases:
            try:
                result = self._run_lua_test(lua_code, tc)
                if result:
                    passed_count += 1
                else:
                    errors.append("Test case failed")
            except Exception as e:
                errors.append(f"Runtime error: {e}")

        score = passed_count / max(len(test_cases), 1)

        # Update observer
        # If code compiles and has function, give partial credit even if tests fail
        base_score = 0.3 if (has_function and compile_ok) else 0.0
        final_score = max(score, base_score)
        
        passed = final_score >= 0.3  # Lower threshold: compile OK = pass
        
        error_msg = errors[0] if errors else ""
        y_observed = self._compute_observer_outputs(
            module_name, passed, final_score, error_msg, uses_modules=[]
        )
        self.observer.update(y_observed)

        if passed:
            return True, final_score, "Good work!"
        else:
            return False, final_score, f"Needs work: {error_msg}"

    def _check_lua_compiles(self, lua_code: str) -> tuple:
        """Check if Lua code compiles by running it in the sandbox."""
        from backend.sandbox.sandbox_manager import sandbox_manager

        payload = {"particle": {"x": 0, "y": 0, "vx": 0, "vy": 0}, "rule": lua_code, "frames": 1}
        try:
            result = sandbox_manager.run_rule(lua_code, payload["particle"], frames=1)
            if result.get("status") == "ok":
                return True, ""
            return False, result.get("message", "Unknown error")[:200]
        except Exception as e:
            return False, str(e)[:200]

    def get_observer_report(self) -> dict:
        """Get full observer progress report."""
        return self.observer.get_progress_report()

    def _run_lua_test(self, lua_code: str, test_case) -> bool:
        """Run a single Lua test case.
        
        For math test cases (tuples): call the function and check particle.x.
        For physics test cases (dicts): run in sandbox and check particle state.
        """
        import subprocess, json

        # --- Math test case: tuple (a, b, expected) ---
        if isinstance(test_case, tuple):
            if len(test_case) < 2:
                return False

            expected = test_case[-1]
            args = list(test_case[:-1])
            func_name = self._detect_function_name(lua_code)

            if func_name:
                args_str = ", ".join(str(a) for a in args)
                wrapper = lua_code + "\n" + f"""
local ok, result = pcall({func_name}, {args_str})
if ok then
    particle.x = result or 0
else
    particle.x = -999
end
"""
            else:
                wrapper = lua_code

            from backend.sandbox.sandbox_manager import sandbox_manager
            try:
                result = sandbox_manager.run_rule(wrapper, {"x": 0, "y": 0, "vx": 0, "vy": 0}, frames=1)
                if result.get("status") == "ok":
                    actual = result.get("particle", {}).get("x", -999)
                    if isinstance(expected, (int, float)):
                        return abs(actual - expected) < 0.5
                    return str(actual) == str(expected)
                return False
            except Exception:
                return False

        # --- Physics test case: dict with 'from' and 'to' ---
        if isinstance(test_case, dict) and "from" in test_case:
            fx, fy = test_case["from"]
            particle = {"x": fx, "y": fy, "vx": 0, "vy": 0}
            from backend.sandbox.sandbox_manager import sandbox_manager
            result = sandbox_manager.run_rule(lua_code, particle)

            if result.get("status") == "ok":
                final = result.get("particle", {})
                expected_vx = test_case.get("expected_vx", 0)
                expected_vy = test_case.get("expected_vy", 0)
                vx_ok = abs(final.get("vx", 0) - expected_vx) < 1.0
                vy_ok = abs(final.get("vy", 0) - expected_vy) < 1.0
                return vx_ok and vy_ok
            return False

        return False

    def _detect_function_name(self, lua_code: str) -> str:
        """Detect the main function name from Lua code."""
        import re
        patterns = [
            r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            r'local\s+function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        ]
        for pattern in patterns:
            m = re.search(pattern, lua_code)
            if m:
                return m.group(1)
        return None

    def should_intervene(self, module_name: str, failure_count: int, level: int) -> bool:
        """Decide if teacher should intervene or let student struggle more.
        
        Philosophy:
        - Early levels (1-5): intervene after 2 failures (build confidence)
        - Mid levels (6-10): intervene after 3 failures (build resilience)
        - Advanced (11+): intervene after 4 failures (deep learning)
        """
        if level <= 5:
            return failure_count >= 2
        elif level <= 10:
            return failure_count >= 3
        else:
            return failure_count >= 4

    def provide_hint(self, module_name: str, module_desc: str, error: str) -> str:
        """Provide a hint without giving away the answer."""
        hints = {
            "count": "Try using a for loop: for i=1,n do ... end",
            "add": "The simplest way: return a + b",
            "multiply": "Use a loop that adds 'a' exactly 'b' times",
            "velocity": "velocity = distance / time. Think about units.",
            "move_to": "First find the direction (dx, dy), then normalize, then multiply by speed",
            "force": "Newton's second law: F = m * a. Force equals mass times acceleration",
        }

        for key, hint in hints.items():
            if key in module_name.lower():
                return hint

        return f"Think about: {module_desc}. Error was: {error}"

    def generate_curriculum_explanation(self, level: int, subject: str) -> str:
        """Generate a brief explanation of why this level matters historically."""
        explanations = {
            1: "Counting was the first mathematical skill humans developed. Before writing, people used fingers, stones, and marks to track quantities.",
            2: "Addition emerged when people needed to combine quantities — merging herds, counting trade goods, building with multiple stones.",
            3: "Subtraction is the inverse of addition. It answers: 'How much is left?' or 'What is the difference?'",
            4: "Multiplication is repeated addition. Instead of adding 3+3+3+3, we write 4×3. Ancient Egyptians used doubling.",
            5: "Division is repeated subtraction. It answers: 'How many times does b fit into a?'",
            6: "Fractions represent parts of a whole. They emerged when people needed to divide food, land, and time.",
            7: "Perimeter is the distance around a shape. Ancient Egyptians used it to measure fields after Nile floods.",
            8: "Area measures the surface inside a shape. It's essential for construction, agriculture, and trade.",
            9: "Distance in geometry uses the Pythagorean theorem — one of the oldest and most useful formulas.",
            10: "Velocity describes how fast something moves. Galileo was the first to measure it precisely.",
            11: "Acceleration describes how velocity changes. It's the key to understanding forces.",
            12: "Moving to a target requires understanding direction and speed — the foundation of kinematics.",
            13: "Force causes acceleration. Newton's F=ma is the most important equation in classical physics.",
            14: "Kinetic energy is the energy of motion. It depends on mass and velocity squared.",
            15: "Potential energy is stored energy. A ball at height has energy that converts to motion when it falls.",
        }

        return explanations.get(level, f"Level {level} of {subject} builds on all previous knowledge.")


# === Singleton ===
teacher = TeacherEngine()
