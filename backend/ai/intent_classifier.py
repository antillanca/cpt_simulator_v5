"""Intent Classifier - Maps natural language to physics modules (i@ agent).

Avoids 'ruminating' (re-calculating known intents) by using heuristics and caching.
"""
import re
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Heuristic mapping: keywords -> module_name
INTENT_MAP = {
    "move_to": ["moverse", "ir a", "viajar", "move to", "go to", "travel", "mueve", "mueva", "dirigir"],
    "velocity": ["velocidad", "rapidez", "speed", "velocity", "vuela"],
    "acceleration": ["aceleración", "acelerar", "acceleration", "accelerate", "acelera", "pisa"],
    "reflect_angle": ["rebotar", "reflejar", "bounce", "reflect", "rebota", "choque"],
    "force": ["fuerza", "empujar", "force", "push", "empuja", "golpea"],
    "energy_kinetic": ["energía", "cinética", "energy", "kinetic", "calor"],
    "area": ["área", "superficie", "area", "surface", "espacio"],
    "distance": ["distancia", "lejos", "distance", "far", "lejanía"],
}

# In-memory intent cache to prevent token waste
INTENT_CACHE: Dict[str, str] = {}

def classify_intent(text: str) -> Optional[str]:
    """Classify natural language intent into a known module name.
    
    Uses heuristics first, then returns the module name.
    """
    if not text:
        return None
        
    text_lower = text.lower().strip()
    
    # 1. Check Cache
    if text_lower in INTENT_CACHE:
        logger.info(f"[i@] Intent Cache Hit: '{text_lower}' -> {INTENT_CACHE[text_lower]}")
        return INTENT_CACHE[text_lower]
        
    # 2. Heuristic Check
    for module_name, keywords in INTENT_MAP.items():
        for kw in keywords:
            if kw in text_lower:
                logger.info(f"[i@] Heuristic Match: '{text_lower}' -> {module_name}")
                INTENT_CACHE[text_lower] = module_name
                return module_name
                
    # 3. Fallback (Future: Use LLM to classify if complex)
    # For now, we stay minimalist and deterministic
    logger.warning(f"[i@] Unrecognized Intent: '{text_lower}'")
    return None
