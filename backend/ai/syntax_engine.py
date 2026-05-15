class SyntaxEngine:
    """Motor de sintaxis que aplica gramática SVO sobre tokens simbólicos."""
    
    def __init__(self):
        # Mapeo de tokens a palabras inglesas
        self.dictionary = {
            "[SELF]": "I",
            "[OTHER]": "the object",
            "[BLOCK]": "the obstacle",
            "[GOAL]": "the target",
            "[PUSH]": "push",
            "[MOVE]": "move",
            "[APPROACH]": "approach",
            "[RETREAT]": "move away from",
            "[AVOID]": "avoid",
            "[HIT]": "hit",
            "[NEAR]": "near"
        }

    def compose(self, symbols):
        """Convierte una secuencia de símbolos en una oración coherente en inglés."""
        if not symbols:
            return "..."
            
        # 1. Traducir tokens a palabras
        words = []
        for s in symbols:
            word = self.dictionary.get(s)
            if word:
                words.append(word)
        
        if not words:
            return "..."

        # 2. Construcción SVO (Sujeto Verbo Objeto)
        # El primer elemento suele ser el agente ([SELF])
        # El segundo suele ser la acción o relación
        # Los siguientes son los objetos
        
        if len(words) == 1:
            sentence = words[0]
        elif len(words) == 2:
            sentence = f"{words[0]} {words[1]}"
        else:
            # Manejo de múltiples objetos (Ej: I approach target and avoid obstacle)
            subject = words[0]
            verb = words[1]
            objects = words[2:]
            
            if len(objects) > 1:
                obj_str = f"{objects[0]} to {objects[1]}"
                # Si hay un verbo de evasión, lo conectamos
                if "avoid" in objects:
                    obj_str = f"{objects[0]} while I {objects[1]} {objects[2]}"
            else:
                obj_str = objects[0]
                
            sentence = f"{subject} {verb} {obj_str}"

        # Limpieza y puntuación
        return sentence.strip().capitalize() + "."

syntax_engine = SyntaxEngine()
