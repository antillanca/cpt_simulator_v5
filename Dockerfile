# CORE v3.2 — Official Docker Image
# Minimal, deterministic install for core-runtime-engine
#
# Build:
#   docker build -t core-runtime-engine:3.2 .
#
# Run smoke benchmark:
#   docker run --rm core-runtime-engine:3.2 python scripts/run_smoke_benchmark.py
#
# Run tutorial:
#   docker run --rm core-runtime-engine:3.2 python examples/01_linear_system_walkthrough.py
#
# Run tests:
#   docker run --rm core-runtime-engine:3.2 python -m pytest tests/ -q

FROM python:3.11-slim AS base

# System deps for faiss-cpu + numpy
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/core-runtime-engine

# Install Python dependencies first (layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[linear-system,dev]" 2>/dev/null || \
    pip install --no-cache-dir -e .

# Copy source
COPY core_runtime/ core_runtime/
COPY backend/ backend/
COPY scripts/ scripts/
COPY examples/ examples/
COPY tests/ tests/
COPY conftest.py ./

# Default: run smoke benchmark
CMD ["python", "scripts/run_smoke_benchmark.py", "--seed", "42", "--samples", "10"]
