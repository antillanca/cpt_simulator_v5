import logging
"""
Learning Orchestrator - Connects student, teacher, and curriculum.
Manages the learning loop: generate → test → confirm/reject → next level.
"""

import logging

logger = logging.getLogger(__name__)

import json
import asyncio
from pathlib import Path
from typing import Optional

from backend.ai.student_engine import student, StudentEngine
from backend.ai.teacher_engine import teacher, TeacherEngine
from backend.sandbox.sandbox_manager import sandbox_manager

BASE_DIR = Path(__file__).parent.parent.parent
CURRICULUM_FILE = BASE_DIR / "curriculo_cronologico_fisica_matematica.md"


class LearningOrchestrator:
    """Orchestrates the full learning cycle.
    
    Flow:
    1. Load curriculum (chronological order)
    2. For each level:
       a. Teacher generates exercise
       b. Student generates Lua code
       c. Test in sandbox
       d. If pass → confirm module (click)
       e. If fail → retry or teacher intervenes
    3. Report progress
    """

    def __init__(self):
        self.student = student
        self.teacher = teacher
        self.is_running = False
        self.current_module = None

    async def start(self):
        """Start the learning loop (async)."""
        self.is_running = True
        print("[Orchestrator] Starting learning loop...")

        # Get pending modules sorted by level
        modules = self.student.modules.get("modules", {})
        pending = sorted(
            [(name, mod) for name, mod in modules.items() if mod.get("status") == "pending"],
            key=lambda x: x[1].get("level", 999),
        )

        if not pending:
            print("[Orchestrator] No pending modules!")
            self.is_running = False
            return

        print(f"[Orchestrator] {len(pending)} modules to learn")

        for module_name, module in pending:
            if not self.is_running:
                break

            self.current_module = module_name
            level = module.get("level", 0)
            subject = module.get("subject", "unknown")

            print(f"\n{'='*60}")
            print(f"[Orchestrator] Level {level}: {module_name}")
            print(f"[Orchestrator] Subject: {subject}")
            print(f"[Orchestrator] Description: {module['description']}")
            print(f"{'='*60}")

            # Teacher provides historical context
            explanation = self.teacher.generate_curriculum_explanation(level, subject)
            print(f"[Teacher] {explanation}")

            # Generate exercise (async - uses LLM)
            exercise = await self.teacher.generate_exercise(
                module_name, module["description"], level
            )
            logger.info(f"[Teacher] Exercise: {exercise['question']}")

            # Student learns (async)
            success = await self.student.learn_module(
                module_name=module_name,
                test_fn=lambda code: self._test_code(code, exercise),
                teacher_fn=self._teacher_intervene,
            )

            if success:
                logger.info(f"[Orchestrator] ✅ Level {level} complete!")
            else:
                logger.warning(f"[Orchestrator] ❌ Level {level} failed, moving on...")

            await asyncio.sleep(1)

        self.is_running = False
        self._print_summary()

    def stop(self):
        """Stop the learning loop."""
        self.is_running = False
        print("[Orchestrator] Stopping...")

    def _test_code(self, lua_code: str, exercise: dict) -> tuple:
        """Test student's Lua code."""
        test_cases = exercise.get("test_cases", [])
        if not test_cases:
            # No test codes = manual review needed
            return True, "No automated tests"

        passed, score, feedback = self.teacher.evaluate_lua(
            lua_code, self.current_module, test_cases
        )
        return passed, feedback

    def _teacher_intervene(self, module_name: str, lua_code: str, error: str):
        """Teacher intervention when student is stuck."""
        module = self.student.get_module(module_name)
        if not module:
            return

        failure_count = module.get("rejection_count", 0)
        level = module.get("level", 0)

        if self.teacher.should_intervene(module_name, failure_count, level):
            hint = self.teacher.provide_hint(module_name, module["description"], error)
            print(f"[Teacher] 💡 Hint: {hint}")

    def _print_summary(self):
        """Print learning summary."""
        mem = self.student.memory_usage()
        print(f"\n{'='*60}")
        logger.info(f"[Orchestrator] LEARNING SUMMARY")
        print(f"{'='*60}")
        print(f"Total modules: {mem['total_modules']}")
        print(f"Confirmed (layers): {mem['confirmed']}")
        print(f"Pending: {mem['pending']}")
        print(f"Rejected: {mem['rejected']}")
        print(f"Inactive: {mem['inactive']}")
        print(f"Total code size: {mem['total_code_bytes']} bytes")
        print(f"Model: {mem['ollama_model']} ({mem['ollama_model_size_mb']} MB)")

        # Print confirmed layers
        layers = self.student.get_confirmed_layers()
        if layers:
            print(f"\nConfirmed layers:")
            for name, mod in layers:
                print(f"  Level {mod['level']}: {name} - {mod['description']}")

    def get_status(self) -> dict:
        """Get current learning status with cognitive layer metadata."""
        mem = self.student.memory_usage()
        layers = self.student.get_confirmed_layers()

        # Determine cognitive layer name for the current active module
        current_layer_name = "Unknown"
        if self.current_module:
            module = self.student.get_module(self.current_module)
            if module:
                level = module.get("level", 0)
                if level <= 1: current_layer_name = "Logical Transformation"
                elif level <= 6: current_layer_name = "Mathematical Abstraction"
                elif level <= 9: current_layer_name = "Geometric Representation"
                elif level <= 17: current_layer_name = "Physical Simulation"
                else: current_layer_name = "Advanced Abstraction"

        return {
            "is_running": self.is_running,
            "current_module": self.current_module,
            "cognitive_layer": current_layer_name,
            "total_modules": mem["total_modules"],
            "confirmed_layers": mem["confirmed"],
            "pending": mem["pending"],
            "rejected": mem["rejected"],
            "memory_bytes": mem["total_code_bytes"],
            "layers": [
                {"name": name, "level": mod["level"], "subject": mod.get("subject", "")}
                for name, mod in layers
            ],
        }


# === Singleton ===
orchestrator = LearningOrchestrator()
