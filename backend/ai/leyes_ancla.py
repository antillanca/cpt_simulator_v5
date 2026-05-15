"""Leyes Ancla - Physical and ethical constraints for the i@ agent.

This module provides fast heuristic checks to protect the 'Anchor Laws'
of the simulation before calling the LLM.
"""
import re
from typing import Optional

ANCLA_VIOLATIONS = [
    {
        "id": "energy_conservation",
        "keywords": [
            "infinito", "eterno", "perpetuo", "infinite", "perpetual", 
            "sin fin", "never end", "always accelerate", "siempre acelera"
        ],
        "message": "Error: Violación detectada. El agente i@ no permite la creación de sistemas de energía infinita (Conservación de Energía)."
    },
    {
        "id": "bounds_destruction",
        "keywords": [
            "destruir", "romper", "borrar", "eliminar", "destroy", "break", "delete",
            "salir del canvas", "escape canvas"
        ],
        "message": "Error: El agente i@ protege la integridad del sistema. No se permite la destrucción o evasión de los límites del universo."
    },
    {
        "id": "matter_creation",
        "keywords": [
            "crear materia", "aparecer", "spawn", "create matter", "teletransportar", "teleport"
        ],
        "message": "Error: El agente i@ no permite la creación de materia o teletransportación instantánea en este nivel."
    }
]

def check_intent_violation(text: str) -> Optional[str]:
    """Check natural language for Anchor Law violations (Fast Heuristic).
    
    Returns a translated error message if a violation is found, else None.
    """
    if not text:
        return None
        
    text_lower = text.lower()
    for ancla in ANCLA_VIOLATIONS:
        for kw in ancla["keywords"]:
            # Match whole words or phrases
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text_lower):
                return ancla["message"]
                
    return None
