import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from planner import plan, tabular_filter
    print("✅ Successfully imported plan and tabular_filter from planner package")
    
    # Test tabular_filter scan
    print(f"DEBUG: Models loaded: {list(tabular_filter.models.keys())}")
    
    # Test a simple prediction (will return True if model missing)
    features = [0.0] * 8
    success, prob = tabular_filter.predict(features, subject="action")
    print(f"DEBUG: Prediction for 'action': {success} (prob: {prob})")
    
except Exception as e:
    print(f"❌ Error importing or testing planner: {e}")
    import traceback
    traceback.print_exc()
