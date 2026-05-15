import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from planner.tabular_filter import tabular_filter
from world_model.state_encoder import encode, encode_action

def test():
    print(f"Modo del filtro: {tabular_filter.mode}")
    
    # Un estado y acción cualquiera
    state = {"position": [400, 300], "velocity": [0, 0]}
    action = {"vx": 5, "vy": 0}
    
    features = encode(state) + encode_action(action)
    is_promising, prob = tabular_filter.predict(features)
    
    print(f"Probabilidad de éxito: {prob:.4f}")
    print(f"¿Es prometedora?: {is_promising}")

if __name__ == "__main__":
    test()
