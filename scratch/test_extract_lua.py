#!/usr/bin/env python3
"""Test _extract_lua with problematic model responses."""
import sys
sys.path.insert(0, '/home/john/www/cpt_simulator_v5')

from backend.ai.student_engine import student

test_cases = [
    # (input, expected_output, description)
    (
        "local t = 0\nfunction update_particle(particle)\n    t = t + 1\n    particle.x = 50 * math.sin(t)\nend",
        "local t = 0\nfunction update_particle(particle)\n    t = t + 1\n    particle.x = 50 * math.sin(t)\nend",
        "Clean Lua code (no prose)"
    ),
    (
        "Here is the code you need:\nlocal t = 0\nfunction update_particle(particle)\n    t = t + 1\n    particle.x = 50 * math.sin(t)\nend",
        "local t = 0\nfunction update_particle(particle)\n    t = t + 1\n    particle.x = 50 * math.sin(t)\nend",
        "Prose before code"
    ),
    (
        "To make the particle oscillate, we need to use sine:\nlocal t = 0\nfunction update_particle(particle)\n    t = t + 1\n    particle.x = 50 * math.sin(t)\nend",
        "local t = 0\nfunction update_particle(particle)\n    t = t + 1\n    particle.x = 50 * math.sin(t)\nend",
        "Prose with 'need' before code"
    ),
    (
        "```lua\nlocal t = 0\nfunction update_particle(particle)\n    t = t + 1\n    particle.x = 50 * math.sin(t)\nend\n```",
        "local t = 0\nfunction update_particle(particle)\n    t = t + 1\n    particle.x = 50 * math.sin(t)\nend",
        "Markdown code block"
    ),
    (
        "Here's the solution:\n```lua\nlocal t = 0\nfunction update_particle(particle)\n    t = t + 1\n    particle.x = 50 * math.sin(t)\nend\n```",
        "local t = 0\nfunction update_particle(particle)\n    t = t + 1\n    particle.x = 50 * math.sin(t)\nend",
        "Prose + markdown code block"
    ),
    (
        "<think>I need to use sine for oscillation</think>\nlocal t = 0\nfunction update_particle(particle)\n    t = t + 1\n    particle.x = 50 * math.sin(t)\nend",
        "local t = 0\nfunction update_particle(particle)\n    t = t + 1\n    particle.x = 50 * math.sin(t)\nend",
        "Think tags + code"
    ),
    (
        "We need to compute the force F = q * v * B:\nlocal q = 1\nlocal v = particle.vx\nlocal B = 2\nlocal F = q * v * B\nparticle.vy = F",
        "local q = 1\nlocal v = particle.vx\nlocal B = 2\nlocal F = q * v * B\nparticle.vy = F",
        "Prose with formula before code"
    ),
    (
        "This is just prose with no code at all. Nothing here.",
        None,
        "Pure prose (no Lua) - should return None"
    ),
]

print("=" * 60)
print("Testing _extract_lua with problematic inputs")
print("=" * 60)

passed = 0
failed = 0
for i, (input_text, expected, desc) in enumerate(test_cases):
    result = student._extract_lua(input_text)
    status = "PASS" if result == expected else "FAIL"
    if status == "PASS":
        passed += 1
    else:
        failed += 1
    print(f"\n[{status}] Test {i+1}: {desc}")
    if status == "FAIL":
        print(f"  Input:    {repr(input_text[:80])}")
        print(f"  Expected: {repr(expected[:80] if expected else None)}")
        print(f"  Got:      {repr(result[:80] if result else None)}")

print(f"\n{'=' * 60}")
print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)}")
print("=" * 60)
