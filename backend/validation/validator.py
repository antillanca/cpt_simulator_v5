"""Validator - Lua rule security validation.

All internal text is English. Checks for prohibited tokens
and validates rule structure before sandbox execution.
"""
from backend.config import RULE_MAX_LENGTH, PROHIBITED_TOKENS, CANVAS_WIDTH, CANVAS_HEIGHT
from backend.verifiers import verify_simulation
from backend.traces.schema import canonicalize_trace, TraceValidationError


class Validator:
    def validate_rule(self, rule_text: str, require_particle: bool = True) -> tuple[bool, str]:
        """Validate a Lua rule for security and structure.
        
        Args:
            rule_text: Lua code to validate
            require_particle: If True, requires 'particle' in code (physics rules).
                            If False, allows pure math functions.
        """
        if not rule_text or not rule_text.strip():
            return False, "Rule text is empty."

        if len(rule_text) > RULE_MAX_LENGTH:
            return False, f"Rule exceeds maximum length ({RULE_MAX_LENGTH} chars)."

        # Check for prohibited tokens
        rule_lower = rule_text.lower()
        for token in PROHIBITED_TOKENS:
            # Check as whole word (not part of a larger word like 'requirement')
            # Simple heuristic: token preceded by non-alpha or start of string
            import re
            pattern = r'(?<!\w)' + re.escape(token) + r'(?!\w)'
            if re.search(pattern, rule_lower):
                return False, f"Prohibited token found: '{token}'"

        # Check for balanced control structures
        if rule_text.count("if") > rule_text.count("then") + rule_text.count("elseif"):
            # Might have unbalanced if/then
            pass  # Soft check, don't block

        # Check for particle reference (optional for pure math modules)
        if require_particle and "particle" not in rule_lower:
            return False, "Rule must reference 'particle' object."

        return True, ""

    def validate_state(self, state: dict) -> tuple[bool, str]:
        """Validate that particle state is within canvas bounds."""
        x = state.get("x", 0)
        y = state.get("y", 0)
        vx = state.get("vx", 0)
        vy = state.get("vy", 0)

        # Allow slight overshoot (will bounce next frame)
        if x < -100 or x > CANVAS_WIDTH + 100:
            return False, f"Particle x={x:.1f} out of extended bounds."
        if y < -100 or y > CANVAS_HEIGHT + 100:
            return False, f"Particle y={y:.1f} out of extended bounds."

        # Velocity sanity check
        if abs(vx) > 1000 or abs(vy) > 1000:
            return False, f"Particle velocity too high: vx={vx:.1f}, vy={vy:.1f}."

        return True, ""

    def validate_trace(self, trace: dict, invariant_set: list[str]) -> dict:
        """Validate a simulation trace against deterministic invariants."""
        return verify_simulation(trace, invariant_set)

    def validate_trace_schema(self, trace: dict) -> tuple[bool, str]:
        """Validate that a trace already matches the structured trace schema."""
        try:
            canonicalize_trace(trace).validate()
        except TraceValidationError as exc:
            return False, str(exc)
        return True, ""


validator = Validator()
