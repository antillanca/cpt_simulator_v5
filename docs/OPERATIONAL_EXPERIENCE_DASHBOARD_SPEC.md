# Operational Experience Dashboard Specification

## Overview

The operational experience dashboard is a **static local HTML report**
generated from operational experience data. It requires no web server,
no backend, and no JS framework dependencies.

## Data Sources

- `data/operational_experience/manifest.json` — entry list with SHA-256 anchors
- `data/operational_experience/entries/*.json` — individual experience records
- `data/runtime_traces/manifest.json` — trace list
- `data/runtime_traces/*.json` — individual traces
- `data/benchmarks/operational_stats.json` — aggregate statistics

## Dashboard Sections

### 1. Projection Iterations

- Histogram of projection iteration counts
- Average, median, and P95 statistics
- Breakdown by system size (family)

### 2. Warmstart Effectiveness

- Comparison: warm-started vs standard projection iteration counts
- Improvement ratio distribution
- Cases where warmstart was rejected (initial_residual_after >= initial_residual_standard)

### 3. Escalation Rate

- Fraction of tasks that escalated to higher budget or oracle verification
- Escalation by family (system size)
- Trend over collection batches

### 4. Trajectory Classes

- Distribution of trajectory lengths
- Converged vs non-converged trajectories
- Projection iteration distribution

### 5. Routing Distribution

- Pie chart / bar chart of routing decisions
- exact_cache_hit, semantic_retrieval, warmstart_projection,
  standard_projection, increased_budget, oracle_verification, degraded_execution

### 6. Failure Types

- Breakdown of failure modes
- Degraded execution count and rate
- Non-converged projection count
- Evaluation mismatches

### 7. Runtime Distribution

- Histogram of execution times
- Fast (<1ms), Medium (1-10ms), Slow (>10ms) buckets
- Average, median, P95

## Generation

```bash
python scripts/generate_operational_dashboard.py
```

Output: `workspace/operational_dashboard/index.html`

## Design Constraints

- **No web server**: opens directly in a browser via `file://`
- **No backend**: all data is embedded in the HTML
- **No JS framework**: vanilla JS or inline CSS only
- **Deterministic**: same input data produces same HTML output
- **Minimal dependencies**: only matplotlib for PNG chart generation (optional)

## Chart Strategy

Two approaches, both acceptable:

1. **Inline SVG** (preferred): Generate SVG charts directly in Python
   and embed them in the HTML. Zero external dependencies.

2. **Matplotlib PNG**: Generate PNG images with matplotlib and embed
   as base64 data URIs. Requires matplotlib but produces polished charts.

The generator should default to inline SVG and fall back to matplotlib
if available.
