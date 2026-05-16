from backend.benchmarks.cpt_bench.suite import CPTBenchSuite


def test_default_bench_uses_real_curriculum_layers():
    cases = CPTBenchSuite.default_cases()
    names = {case.name for case in cases}
    layers = {case.curriculum_layer for case in cases}

    assert len(cases) >= 6
    assert "energy_kinetic" in names
    assert "quantum_double_slit_logic" in names
    assert 12 in layers
    assert 34 in layers

