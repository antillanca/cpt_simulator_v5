#!/usr/bin/env python3
"""Diagnóstico directo del learning loop sin pasar por API."""
import sys, os, json, time, asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import DEFAULT_STATE, LEARNING_MAX_ATTEMPTS, GOAL_THRESHOLD
from backend.core.orchestrator import orchestrator
from backend.core.syllabus_manager import syllabus_manager
from backend.ai.tutor_engine import tutor_engine
from backend.memory.engine_lora import lora_adapter
from backend.persistence.database import SessionLocal, SyllabusItem

async def attempt_objective_debug(item, attempt_limit=3):
    """Reimplementation of learning_loop.attempt_objective with debug output."""
    target_state = json.loads(item.target_state_json)
    last_failed_rule = None
    last_error = None

    lora_suggestions = lora_adapter.get_suggestions()

    for i in range(attempt_limit):
        print(f"\n  Attempt {i+1}/{attempt_limit}")

        if last_failed_rule and last_error:
            enhanced = f"{item.objective} [Adapted params]"
            rule_text = tutor_engine.refine_rule(last_failed_rule, last_error, enhanced)
        else:
            enhanced = f"{item.objective} [Use gravity={lora_suggestions['gravity_value']:.2f}, friction={lora_suggestions['friction_coefficient']:.2f}]"
            rule_text = tutor_engine.generate_rule(enhanced, orchestrator.current_state)

        if not rule_text:
            print(f"  ⚠️ No rule generated, skipping")
            continue

        print(f"  Rule: {rule_text[:120]}...")

        # Test in sandbox
        result = orchestrator.process_new_rule(rule_text)
        print(f"  Sandbox result: status={result['status']}")

        if result["status"] == "ok":
            actual = result["result"]["particle"]
            print(f"  Particle: {actual}")

            # Check goal
            is_goal = True
            for key in target_state:
                if key in actual:
                    if abs(actual[key] - target_state[key]) > GOAL_THRESHOLD:
                        is_goal = False
                        print(f"  📏 {key}: {actual[key]:.2f} vs target {target_state[key]} (diff > {GOAL_THRESHOLD})")

            if is_goal:
                print(f"  ✅ GOAL REACHED!")
                return True
            else:
                last_error = f"Target not reached. Actual: {actual}, Target: {target_state}"
                last_failed_rule = rule_text
        else:
            last_error = result.get("message", "unknown")
            last_failed_rule = rule_text
            print(f"  ❌ Sandbox error: {last_error[:100]}")

        await asyncio.sleep(0.5)

    return False

async def main():
    print("=" * 60)
    print("LEARNING LOOP DIRECT DIAGNOSIS")
    print("=" * 60)

    # Reset
    orchestrator.current_state = dict(DEFAULT_STATE)
    orchestrator.active_rules = []
    lora_adapter.reset()

    # Count syllabus items
    db = SessionLocal()
    count = db.query(SyllabusItem).count()
    comp = db.query(SyllabusItem).filter(SyllabusItem.is_completed == True).count()
    print(f"Syllabus: {count} total, {comp} completed")
    db.close()

    # Get next item
    item = syllabus_manager.get_next_item()
    if not item:
        print("No syllabus items!")
        return

    print(f"\nTarget: {item.title}")
    print(f"Objective: {item.objective}")
    print(f"Target state: {item.target_state_json}")
    print(f"LOOP_MAX_ATTEMPTS: {LEARNING_MAX_ATTEMPTS}")

    try:
        success = await attempt_objective_debug(item, attempt_limit=2)
        print(f"\n{'='*60}")
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")
        print(f"State: {orchestrator.current_state}")
        print(f"Active rules: {len(orchestrator.active_rules)}")
    except Exception as e:
        import traceback
        print(f"\n💥 CRASH: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())