import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from planner.planner import plan_astar
import time

def run_navigation_test():
    print("🚢 CPT Simulator v5 - Prueba de Navegación (A* + Intuición Neural)")
    
    # Definimos el reto: Ir del extremo superior izquierdo al centro-derecha
    start_state = {"position": [100, 100], "velocity": [0, 0]}
    target_state = {"position": [500, 300]}
    
    print(f"📍 Origen: {start_state['position']}")
    print(f"🎯 Destino: {target_state['position']}")
    print("-" * 40)
    
    start_time = time.time()
    
    # Ejecutamos el planificador
    path = plan_astar(start_state, target_state, max_nodes=300, tolerance=55.0)
    
    duration = time.time() - start_time
    
    print("-" * 40)
    if path:
        print(f"✨ ¡ÉXITO! Se encontró un camino de {len(path)} pasos.")
        print(f"⏱️ Tiempo de búsqueda: {duration:.2f} segundos.")
        print("\nSecuencia de acciones recomendada:")
        for i, action in enumerate(path):
            print(f"  Paso {i+1}: vx={action['vx']}, vy={action['vy']}")
    else:
        print("❌ El planificador no pudo encontrar un camino válido.")
        print(f"⏱️ Tiempo de búsqueda: {duration:.2f} segundos.")

if __name__ == "__main__":
    run_navigation_test()
