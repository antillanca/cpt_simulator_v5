#!/usr/bin/env python3
"""Seed the CPT Simulator v5 syllabus from the chronological physics/math curriculum.

Converts curriculum topics into sandbox-testable objectives with target states.
Each objective is a Lua-rule generation task that produces a measurable physics behavior.

All text is English (internal language).
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.persistence.database import init_db, SessionLocal, SyllabusItem
from backend.core.syllabus_manager import syllabus_manager

# Syllabus items derived from the curriculum, ordered historically.
# Each has: title, objective (for LLM), target_state (for validation).
# Target states use the particle model: {x, y, vx, vy}

SYLLABUS_ITEMS = [
    # === LAYER 0: EXISTENCE AND STATE ===
    {
        "title": "State Definition - Equality",
        "objective": "Set the particle's vertical velocity (vy) to exactly 5.0 units. This establishes the most basic ability to set a state.",
        "target_state": {"vy": 5.0},
        "order": 10,
    },

    # === LAYER 1: DISCRETE COUNTING ===
    {
        "title": "Discrete Increment",
        "objective": "Starting with particle.x at its current position, increment particle.x by exactly 1.0 unit. (Hint: x = x + 1).",
        "target_state": {"x": 1.0}, # Assuming start at 0
        "order": 20,
    },

    # === LAYER 2: FUNDAMENTAL OPERATIONS ===
    {
        "title": "Arithmetic Transformation - Multiplication",
        "objective": "Double the current horizontal velocity. If vx is 2.0, set vx to exactly 4.0.",
        "target_state": {"vx": 4.0},
        "order": 30,
    },

    # === LAYER 10: KINEMATICS ===
    {
        "title": "Uniform Motion - Constant Velocity",
        "objective": "Set the particle to move at exactly vx=5.0 and vy=0.0. This represents the simplest form of kinematic simulation.",
        "target_state": {"vx": 5.0, "vy": 0.0},
        "order": 100,
    },
]


def seed():
    """Populate the database with syllabus items."""
    init_db()
    
    db = SessionLocal()
    existing = db.query(SyllabusItem).count()
    db.close()
    
    if existing > 0:
        print(f"Syllabus already has {existing} items. Clearing and re-seeding...")
        db = SessionLocal()
        db.query(SyllabusItem).delete()  # Clear syllabus; keep learning logs.
        db.commit()
        db.close()
    
    for item in SYLLABUS_ITEMS:
        syllabus_manager.add_item(
            title=item["title"],
            objective=item["objective"],
            target_state=item["target_state"],
            order=item["order"],
        )
        print(f"  Added: [{item['order']}] {item['title']}")
    
    print(f"\nSeeded {len(SYLLABUS_ITEMS)} syllabus items.")


if __name__ == "__main__":
    seed()
