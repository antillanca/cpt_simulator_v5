import math

class SymbolEngine:
    """Mapea estados físicos y dinámicos a una representación simbólica abstracta."""
    
    def __init__(self):
        # El vocabulario se construye sobre la Verdad Inicial (Diferencia, Cambio)
        self.symbols = {
            "AGENT": "[SELF]",
            "OBJECT": "[OTHER]",
            "OBSTACLE": "[BLOCK]",
            "TARGET": "[GOAL]",
            "ACTION_PUSH": "[PUSH]",
            "ACTION_MOVE": "[MOVE]",
            "REL_APPROACH": "[APPROACH]",
            "REL_RETREAT": "[RETREAT]",
            "REL_AVOID": "[AVOID]",
            "STATE_HIT": "[HIT]",
            "STATE_NEAR": "[NEAR]"
        }

    def parse_path(self, path, target_pos=None, obstacles=None):
        """Traduce una trayectoria del planificador a una secuencia de símbolos."""
        if not path or len(path) < 2:
            return [self.symbols["AGENT"], self.symbols["ACTION_MOVE"]]
            
        start = path[0]
        end = path[-1]
        
        symbols = [self.symbols["AGENT"]]
        
        # 1. Detectar intención de movimiento
        dist_start = self._dist(start, target_pos) if target_pos else 999
        dist_end = self._dist(end, target_pos) if target_pos else 999
        
        if dist_end < dist_start:
            symbols.append(self.symbols["REL_APPROACH"])
            symbols.append(self.symbols["TARGET"])
        else:
            symbols.append(self.symbols["ACTION_MOVE"])

        # 2. Detectar interacción con obstáculos (Poda Neural)
        if obstacles:
            for obs in obstacles:
                # Si el camino pasa cerca de un obstáculo, estamos evitándolo
                for step in path:
                    if self._dist(step, (obs["x"], obs["y"])) < obs.get("radius", 50) + 10:
                        symbols.append(self.symbols["REL_AVOID"])
                        symbols.append(self.symbols["OBSTACLE"])
                        break
        
        return symbols

    def _dist(self, p1, p2):
        if not p1 or not p2: return 999
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

symbol_engine = SymbolEngine()
