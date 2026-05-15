"""
SimulationOrchestrator - Physics simulation with pure/mutable separation.

Key design:
- simulate_rule(): PURE function. Does NOT mutate current_state. Safe for DPO pipeline.
- commit_rule(): MUTABLE. Applies rule to current_state. Use when rule is "chosen".
- process_new_rule(): Legacy wrapper (calls simulate_rule + commit_rule).
"""

import uuid
import json
from copy import deepcopy
from backend.sandbox.sandbox_manager import sandbox_manager
from backend.validation.validator import validator
from backend.persistence.database import Rule, Snapshot, SessionLocal
from backend.memory.engine_embeddings import embeddings_engine
from backend.memory.engine_memory import memory_engine
from backend.config import DEFAULT_STATE, SIM_MULTISTEP_FRAMES


class SimulationOrchestrator:
    def __init__(self):
        self.current_state = dict(DEFAULT_STATE)
        self.active_rules = []
        self.is_paused = False

    # =========================================================================
    # PURE FUNCTION: simulate_rule
    # Does NOT mutate current_state. Safe for parallel DPO evaluation.
    # =========================================================================
    def simulate_rule(self, rule_text: str, initial_state: dict = None,
                      frames: int = None) -> dict:
        """Pure simulation: returns result WITHOUT mutating current_state.
        
        Use this for:
        - DPO pipeline (evaluate multiple rule variations in parallel)
        - Testing rules without affecting the simulation
        - Student learning (test before committing)
        
        Returns: {status, result: {particle}} or {status, message}
        """
        if frames is None:
            frames = SIM_MULTISTEP_FRAMES

        # Static validation (security check only, no particle requirement)
        is_valid, error = validator.validate_rule(rule_text, require_particle=False)
        if not is_valid:
            return {"status": "error", "message": error}

        # Use provided state or copy of current (NEVER mutate current_state)
        state = deepcopy(initial_state if initial_state else self.current_state)

        # Single Docker call with multi-frame Lua loop
        result = sandbox_manager.run_rule(rule_text, state, frames=frames)

        if result.get("status") == "ok" and "particle" in result:
            final_state = result["particle"]
            # Dynamic validation
            is_valid, error = validator.validate_state(final_state)
            if not is_valid:
                print(f"[Orchestrator] Validation failed: {error}")
                return {"status": "error", "message": error}
            return {"status": "ok", "result": {"particle": final_state}}
        else:
            msg = result.get("message", "Sandbox execution failed.")
            print(f"[Orchestrator] Simulation failed: {msg}")
            return {"status": "error", "message": msg}

    # =========================================================================
    # MUTABLE FUNCTION: commit_rule
    # Applies rule to current_state. Use when rule is "chosen".
    # =========================================================================
    def commit_rule(self, rule_text: str, frames: int = None) -> dict:
        """Apply rule to current_state (MUTABLE).
        
        Use this when:
        - Rule has been tested and approved
        - Adding to active simulation loop
        - Student has confirmed the rule works
        """
        result = self.simulate_rule(rule_text, frames=frames)
        if result["status"] == "ok":
            self.current_state = result["result"]["particle"]
        return result

    # =========================================================================
    # LEGACY: process_new_rule (wraps simulate + commit)
    # =========================================================================
    def process_new_rule(self, rule_text: str, frames: int = None) -> dict:
        """Legacy wrapper: simulates then commits (for backward compat)."""
        return self.commit_rule(rule_text, frames=frames)

    # =========================================================================
    # Active rules management
    # =========================================================================
    def add_active_rule(self, rule_text: str):
        """Add a rule to the active simulation loop."""
        self.active_rules.append(rule_text)

    def remove_active_rule(self, rule_text: str):
        """Remove a rule from the active simulation loop."""
        if rule_text in self.active_rules:
            self.active_rules.remove(rule_text)

    def step(self) -> dict:
        """Execute one simulation step by running all active rules."""
        if self.is_paused:
            return self.current_state

        for rule in self.active_rules:
            result = sandbox_manager.run_rule(rule, self.current_state)
            if result.get("status") == "ok" and "particle" in result:
                self.current_state.update(result["particle"])
            else:
                print(f"[Orchestrator] Rule failed: {result.get('message', 'unknown')}")

        return self.current_state

    def reset_state(self):
        """Reset to initial state."""
        self.current_state = dict(DEFAULT_STATE)

    # =========================================================================
    # Persistence
    # =========================================================================
    def save_rule(self, rule_text: str) -> str:
        """Persist a rule to the database."""
        db = SessionLocal()
        try:
            rule_id = str(uuid.uuid4())[:8]
            rule = Rule(
                id=rule_id,
                rule_text=rule_text,
                embedding=json.dumps(embeddings_engine.embed(rule_text)),
            )
            db.add(rule)
            db.commit()
            return rule_id
        finally:
            db.close()


# === Singleton ===
orchestrator = SimulationOrchestrator()
