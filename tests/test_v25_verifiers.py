from backend.verifiers import verify_simulation


def test_verify_simulation_returns_expected_shape():
    trace = {
        "initial_state": {"x": 0, "y": 0, "vx": 1, "vy": 0, "mass": 1},
        "final_state": {"x": 0, "y": 0, "vx": 1, "vy": 0, "mass": 1},
    }
    result = verify_simulation(trace, ["energy_conservation", "momentum_conservation", "logic_basic"])
    assert set(result.keys()) == {"passed", "violations", "metrics"}
    assert result["passed"] is True

