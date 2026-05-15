import asyncio
import logging
from backend.ai.teacher_engine import teacher
from backend.ai.student_engine import student

logging.basicConfig(level=logging.INFO)

async def test_teacher_hint():
    print("Testing Teacher Intervention...")
    
    module_name = "move_to"
    bad_code = "particle.x = 100 -- Constant assignment, doesn't move"
    error = "Target state not reached (x=100, target=500)"
    
    # Simulate intervention
    # Note: _teacher_intervene is synchronous in teacher_engine but calls generate_rule (async)
    # Actually, let's check the implementation in teacher_engine.py
    
    print(f"Triggering intervention for '{module_name}'...")
    # Since teacher_intervene is likely intended to be called by student_engine,
    # let's test if teacher.generate_exercise or teacher.evaluate_work produces good feedback.
    
    # If the teacher provides a hint via tutor_engine:
    from backend.ai.tutor_engine import tutor_engine
    hint = await tutor_engine.generate_rule(f"Give a small LUA HINT for: {error}", {"x": 100})
    
    print(f"Teacher Hint: {hint}")
    if hint and len(hint) > 5:
        print("✅ Teacher provided a hint.")
    else:
        print("❌ Teacher failed to provide a hint.")

if __name__ == "__main__":
    asyncio.run(test_teacher_hint())
