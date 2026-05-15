from backend.dsl.compiler import compile_dsl


def test_compile_dsl_produces_lua_and_metadata():
    dsl = """
law:
  name: newton_second_law
inputs:
  - force
  - mass
  - dt
equations:
  - acceleration = force / mass
  - velocity = velocity + acceleration * dt
invariants:
  - energy_conservation
  - momentum_conservation
"""
    compiled = compile_dsl(dsl)
    assert "state.acceleration" in compiled["lua_code"]
    assert compiled["metadata"]["law"]["name"] == "newton_second_law"
    assert len(compiled["tests"]) == 2
    assert "energy_conservation" in compiled["documentation"]

