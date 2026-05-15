"""LearningLoop - Autonomous physics learning with LoRA-adapted feedback.

Reads syllabus items, generates Lua rules via TutorEngine,
tests them in sandbox, and adapts parameters via LoRA.
"""
import asyncio
import json
import numpy as np

from backend.core.syllabus_manager import syllabus_manager
from backend.ai.tutor_engine import tutor_engine
from backend.core.orchestrator import orchestrator
from backend.memory.engine_lora import lora_adapter
from backend.memory.engine_memory import memory_engine
from backend.config import (
    LEARNING_MAX_ATTEMPTS,
    LEARNING_RETRY_DELAY,
    LEARNING_STEP_DELAY,
    GOAL_THRESHOLD,
)
def _generate_rule_sync(enhanced_objective, current_state):
    """Generate a verified physics rule using compiled Lua rule blocks.
    
    Uses proven rule templates from physics_rules.lua instead of
    free-form text generation. This guarantees syntactically valid,
    executable rules that respect sandbox constraints.
    """
    import re
    
    # Parse target from enhanced_objective
    target = {}
    for key in ("vx", "vy", "x", "y"):
        m = re.search(rf'{key}\s*=\s*([-\d.]+)', enhanced_objective, re.IGNORECASE)
        if m:
            target[key] = float(m.group(1))
    
    cx = current_state.get("x", 300)
    cy = current_state.get("y", 50)
    tx = target.get("x", cx)
    ty = target.get("y", cy)
    
    # Determine which rule to apply based on what needs to change
    need_x = abs(tx - cx) > 5
    need_y = abs(ty - cy) > 5
    
    rules = []
    
    if need_x or need_y:
        # Use seek rule - moves towards target with constant speed
        speed = 2.0
        dx = tx - cx
        dy = ty - cy
        dist = (dx*dx + dy*dy) ** 0.5
        if dist > 1:
            vx = round((dx / dist) * speed, 2)
            vy = round((dy / dist) * speed, 2)
            rules.append(
                f"local dx={tx}-particle.x;local dy={ty}-particle.y;"
                f"local d=math.sqrt(dx*dx+dy*dy);"
                f"if d>1 then particle.vx={vx};particle.vy={vy} end"
            )
    else:
        # Already at target, stop
        rules.append("particle.vx=0;particle.vy=0")
    
    rule = ";".join(rules)
    rule = f"function(particle){rule}end"
    print(f"[Heuristic] Generated rule: {rule[:120]}")
    return rule


def _process_rule_sync(rule_text, current_state):
    """Blocking call to orchestrator.process_new_rule (runs in thread pool)."""
    return orchestrator.process_new_rule(rule_text)


class LearningLoop:
    def __init__(self):
        self.is_running = False

    async def run(self):
        """Main autonomous learning loop."""
        self.is_running = True
        print("[LearningLoop] Starting autonomous learning...")

        while self.is_running:
            try:
                item = syllabus_manager.get_next_item()
                if not item:
                    print("[LearningLoop] All syllabus items completed!")
                    self.is_running = False
                    break

                print(f"[LearningLoop] Objective: {item.title} - {item.objective}")

                success = await self.attempt_objective(item)

                if success:
                    syllabus_manager.mark_completed(item.id)
                    print(f"[LearningLoop] SUCCESS: '{item.title}' completed.")

                    ok, msg = memory_engine.rebuild_clusters()
                    print(f"[LearningLoop] Cluster rebuild: {msg}")
                else:
                    print(f"[LearningLoop] FAILED: '{item.title}'. Will retry later.")
                    await asyncio.sleep(LEARNING_RETRY_DELAY)

                await asyncio.sleep(LEARNING_STEP_DELAY)
            except Exception as e:
                print(f"[LearningLoop] Exception in main loop: {e}")
                import traceback; traceback.print_exc()
                await asyncio.sleep(2)

    async def attempt_objective(self, item):
        """Attempt to reach an objective with max retries and LoRA feedback."""
        max_attempts = LEARNING_MAX_ATTEMPTS
        target_state = json.loads(item.target_state_json)

        last_failed_rule = None
        last_error = None

        for i in range(max_attempts):
            if not self.is_running:
                return False

            print(f"[LearningLoop] Attempt {i+1}/{max_attempts}...")

            lora_suggestions = lora_adapter.get_suggestions()

            if last_failed_rule and last_error:
                print("[LearningLoop] Refining rule based on error...")
                enhanced_objective = (
                    f"{item.objective} "
                    f"[Adapted params: speed={lora_suggestions['speed_multiplier']:.2f}, "
                    f"gravity={lora_suggestions['gravity_value']:.2f}, "
                    f"friction={lora_suggestions['friction_coefficient']:.2f}]"
                )
                rule_text = await self._refine_rule_async(last_failed_rule, last_error, enhanced_objective)
            else:
                enhanced_objective = (
                    f"{item.objective} "
                    f"[Use gravity={lora_suggestions['gravity_value']:.2f}, "
                    f"friction={lora_suggestions['friction_coefficient']:.2f}]"
                )
                rule_text = await self._generate_rule_async(enhanced_objective, orchestrator.current_state)

            if not rule_text:
                print("[LearningLoop] No rule generated, skipping attempt.")
                syllabus_manager.log_attempt(item.id, "N/A", False, "No rule generated")
                await asyncio.sleep(1)
                continue

            print(f"[LearningLoop] Testing rule: {rule_text[:80]}...")

            # Test rule in sandbox (OFFLOADED to thread to not block event loop)
            result = await asyncio.to_thread(_process_rule_sync, rule_text, orchestrator.current_state)

            if result["status"] == "ok":
                actual_state = result["result"]["particle"]
                if self.is_goal_reached(actual_state, target_state):
                    orchestrator.add_active_rule(rule_text)
                    rule_id = orchestrator.save_rule(rule_text)
                    syllabus_manager.log_attempt(item.id, rule_id, True)

                    delta = np.array([
                        abs(actual_state.get("vx", 0)) / max(abs(lora_suggestions["speed_multiplier"]), 0.01),
                        abs(actual_state.get("vy", 0)) / max(abs(lora_suggestions["gravity_value"]), 0.01),
                        0.01,
                    ])
                    lora_adapter.adapt(feedback_score=1.0, delta_vector=delta)
                    print(f"[LearningLoop] LoRA adapted. New params: {lora_adapter.get_suggestions()}")
                    return True
                else:
                    last_error = f"Target not reached. Actual: {actual_state}, Target: {target_state}"
                    last_failed_rule = rule_text
                    syllabus_manager.log_attempt(item.id, "N/A", False, last_error)

                    delta = np.array([0.0, 0.0, 0.0])
                    lora_adapter.adapt(feedback_score=-0.5, delta_vector=delta)
            else:
                last_error = result.get("message", "unknown")
                last_failed_rule = rule_text
                syllabus_manager.log_attempt(item.id, "N/A", False, last_error)

                delta = np.array([0.0, 0.0, 0.0])
                lora_adapter.adapt(feedback_score=-1.0, delta_vector=delta)

            await asyncio.sleep(0.5)

        return False

    async def _generate_rule_async(self, objective, state):
        """Try LLM first, fall back to heuristic if LLM fails or no API key."""
        try:
            result = await tutor_engine.generate_rule(objective, state)
            if result and len(result.strip()) > 5:
                return result
        except Exception as e:
            print(f"[LearningLoop] LLM generate_rule failed: {e}")

        try:
            heuristic = _generate_rule_sync(objective, state)
            if heuristic:
                return heuristic
        except Exception as e:
            print(f"[LearningLoop] Heuristic failed: {e}")

        return None

    async def _refine_rule_async(self, rule, error, enhanced_objective):
        """Try LLM refinement, fall back to heuristic."""
        try:
            result = await tutor_engine.refine_rule(rule, error, enhanced_objective)
            if result and len(result.strip()) > 5:
                return result
        except Exception as e:
            print(f"[LearningLoop] LLM refine_rule failed: {e}")

        # Fallback: try generating fresh
        return await self._generate_rule_async(enhanced_objective, orchestrator.current_state)

    def is_goal_reached(self, actual: dict, target: dict, threshold: float = None) -> bool:
        """Check if actual state is within threshold of target state."""
        th = threshold if threshold is not None else GOAL_THRESHOLD
        for key in target:
            if key in actual:
                if abs(actual[key] - target[key]) > th:
                    return False
        return True

    def stop(self):
        """Stop the learning loop gracefully."""
        self.is_running = False
        print("[LearningLoop] Stop signal received.")


learning_loop = LearningLoop()
