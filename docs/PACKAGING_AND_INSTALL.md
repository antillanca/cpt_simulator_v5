# Packaging and Install Guide

## Base Install

```bash
pip install core-runtime-engine
```

The base install includes the core runtime engine with numpy as the only
mandatory dependency. This is sufficient for the `linear_system` domain
and all deterministic guarantees.

## Optional Extras

### Circuit Domain

```bash
pip install core-runtime-engine[circuits]
```

Adds `torch>=2.0` and `faiss-cpu>=1.7` for neural surrogate models
and vector similarity retrieval.

### Linear System Domain

```bash
pip install core-runtime-engine[linear-system]
```

No additional dependencies beyond the base install (numpy).

### Development Tools

```bash
pip install core-runtime-engine[dev]
```

Adds `pytest>=7.0`, `pytest-xdist`, and `matplotlib>=3.5`.

### Documentation Tools

```bash
pip install core-runtime-engine[docs]
```

Adds `markdown>=3.4` for documentation generation.

### Everything

```bash
pip install core-runtime-engine[all]
```

Installs all optional extras.

## From Source

```bash
git clone <repo-url>
cd cpt_simulator_v5
pip install -e ".[linear-system,dev]"
```

## Pre-release Versions

v3.2.0a1 is the current pre-release. Install with:

```bash
pip install --pre core-runtime-engine>=3.2.0a1
```

## Docker

```bash
docker build -t core-runtime-engine:3.2 .
docker run --rm core-runtime-engine:3.2 python scripts/run_smoke_benchmark.py
```

## Verification

After installation, verify with:

```bash
python -c "import core_runtime; print('OK')"
python -m pytest tests/ -q
python scripts/run_smoke_benchmark.py --samples 5
```

## Key Design Decisions

- **No mandatory circuit dependencies**: torch and faiss-cpu are optional.
  This keeps the base install lightweight for non-circuit domains.
- **Deterministic by default**: All runs are reproducible with fixed seeds.
- **No GPU required for base usage**: The linear_system domain and all
  core tests run on CPU.
