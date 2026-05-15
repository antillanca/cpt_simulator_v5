import sys, os
sys.path.insert(0, os.getcwd())
from backend.sandbox.sandbox_manager import sandbox_manager

tests = {
    "math_vectors": {
        "code": "particle.vx = 10 * math.cos(math.rad(0))\nparticle.vy = 10 * math.sin(math.rad(0))",
        "state": {"x":300,"y":50,"vx":0,"vy":0},
        "frames": 1,
        "target": {"vx": 10, "vy": 0}
    },
    "math_trigonometry": {
        "code": "particle.vx = 5 * math.cos(math.rad(60))\nparticle.vy = 5 * math.sin(math.rad(60))",
        "state": {"x":300,"y":50,"vx":0,"vy":0},
        "frames": 1,
        "target": {"vx": 2.5, "vy": 4.33}
    },
    "magnetism_lorentz_force": {
        "code": "particle.vx = 5\nlocal q = 1\nlocal B = 2\nparticle.vy = q * particle.vx * B",
        "state": {"x":300,"y":50,"vx":0,"vy":0},
        "frames": 1,
        "target": {"vy": 10}
    },
    "numerical_analysis_euler": {
        "code": "function update_particle(p)\n  p.vy = (p.vy or 0) + 9.81\nend",
        "state": {"x":300,"y":50,"vx":0,"vy":0},
        "frames": 5,
        "target": {"vy": 49.05}
    },
    "cosmology_expansion": {
        "code": "particle.x = 100\nparticle.vx = 70 * particle.x",
        "state": {"x":300,"y":50,"vx":0,"vy":0},
        "frames": 1,
        "target": {"vx": 7000.0}
    }
}

for name, t in tests.items():
    res = sandbox_manager.run_rule(t["code"], t["state"], frames=t["frames"])
    p = res.get("particle", {})
    status = res.get("status")
    
    print(f"--- {name} ---")
    print(f"  Status: {status}")
    
    ok = True
    for key, expected in t["target"].items():
        actual = p.get(key, "MISSING")
        delta = abs(actual - expected) if isinstance(actual, (int, float)) else 999
        mark = "✅" if delta < 1.0 else "❌"
        print(f"  {key}: expected={expected}, got={actual}, delta={delta:.2f} {mark}")
        if delta >= 1.0:
            ok = False
    
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print()
