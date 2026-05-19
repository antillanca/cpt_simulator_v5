#!/usr/bin/env python3
"""CORE v3.2 — Generate Operational Dashboard.

Generates a static local HTML report from operational experience data.
No web server, no backend, no JS framework dependency.

Output: workspace/operational_dashboard/index.html

Usage:
  python scripts/generate_operational_dashboard.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATA_DIR = REPO / "data"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
OP_EXP_DIR = DATA_DIR / "operational_experience"
OUTPUT_DIR = REPO / "workspace" / "operational_dashboard"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stats() -> dict[str, Any]:
    """Load aggregate operational statistics."""
    stats_path = BENCHMARKS_DIR / "operational_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return {}


def load_entries() -> list[dict[str, Any]]:
    """Load all operational experience entries."""
    entries_dir = OP_EXP_DIR / "entries"
    if not entries_dir.exists():
        return []
    entries = []
    for path in sorted(entries_dir.glob("*.json")):
        with open(path) as f:
            entries.append(json.load(f))
    return entries


# ---------------------------------------------------------------------------
# SVG chart generation (inline, no dependencies)
# ---------------------------------------------------------------------------

def svg_bar_chart(
    data: dict[str, int | float],
    title: str,
    width: int = 600,
    height: int = 300,
    bar_color: str = "#4a90d9",
) -> str:
    """Generate an inline SVG bar chart."""
    if not data:
        return f'<p><em>No data for: {title}</em></p>'

    labels = list(data.keys())
    values = [float(v) for v in data.values()]
    max_val = max(values) if values else 1.0
    if max_val == 0:
        max_val = 1.0

    margin_left = 80
    margin_bottom = 60
    margin_top = 40
    chart_w = width - margin_left - 20
    chart_h = height - margin_bottom - margin_top
    bar_w = max(10, min(60, chart_w // max(1, len(labels)) - 8))

    svg_parts = [
        f'<svg width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:monospace;font-size:11px;">',
        f'<text x="{width//2}" y="20" text-anchor="middle" '
        f'font-size="14" font-weight="bold">{title}</text>',
    ]

    # Y-axis
    for i in range(5):
        y_val = max_val * (4 - i) / 4
        y_pos = margin_top + int(chart_h * i / 4)
        svg_parts.append(
            f'<text x="{margin_left-5}" y="{y_pos+4}" '
            f'text-anchor="end" fill="#666">{y_val:.1f}</text>'
        )
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y_pos}" '
            f'x2="{width-20}" y2="{y_pos}" stroke="#ddd" stroke-width="1"/>'
        )

    # Bars
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = margin_left + int(chart_w * (idx + 0.3) / max(1, len(labels)))
        bar_h = int(chart_h * value / max_val)
        y = margin_top + chart_h - bar_h

        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" '
            f'fill="{bar_color}" rx="2"/>'
        )
        svg_parts.append(
            f'<text x="{x + bar_w//2}" y="{y - 4}" '
            f'text-anchor="middle" font-size="9" fill="#333">{value:.1f}</text>'
        )
        # X label
        label_x = x + bar_w // 2
        label_y = margin_top + chart_h + 15
        display_label = str(label)[:12]
        svg_parts.append(
            f'<text x="{label_x}" y="{label_y}" '
            f'text-anchor="middle" font-size="9" fill="#666" '
            f'transform="rotate(-30,{label_x},{label_y})">{display_label}</text>'
        )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def svg_histogram(
    values: list[float],
    title: str,
    bins: int = 10,
    width: int = 600,
    height: int = 300,
    bar_color: str = "#5cb85c",
) -> str:
    """Generate an inline SVG histogram."""
    if not values:
        return f'<p><em>No data for: {title}</em></p>'

    import numpy as _np
    counts, edges = _np.histogram(values, bins=bins)
    max_count = max(counts) if len(counts) > 0 else 1
    if max_count == 0:
        max_count = 1

    margin_left = 80
    margin_bottom = 60
    margin_top = 40
    chart_w = width - margin_left - 20
    chart_h = height - margin_bottom - margin_top
    bar_w = max(8, chart_w // bins - 4)

    svg_parts = [
        f'<svg width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:monospace;font-size:11px;">',
        f'<text x="{width//2}" y="20" text-anchor="middle" '
        f'font-size="14" font-weight="bold">{title}</text>',
    ]

    # Y-axis
    for i in range(5):
        y_val = max_count * (4 - i) / 4
        y_pos = margin_top + int(chart_h * i / 4)
        svg_parts.append(
            f'<text x="{margin_left-5}" y="{y_pos+4}" '
            f'text-anchor="end" fill="#666">{y_val:.0f}</text>'
        )
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y_pos}" '
            f'x2="{width-20}" y2="{y_pos}" stroke="#ddd" stroke-width="1"/>'
        )

    # Bars
    for idx, count in enumerate(counts):
        x = margin_left + int(chart_w * (idx + 0.3) / bins)
        bar_h = int(chart_h * count / max_count) if max_count > 0 else 0
        y = margin_top + chart_h - bar_h

        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" '
            f'fill="{bar_color}" rx="2"/>'
        )
        if count > 0:
            svg_parts.append(
                f'<text x="{x + bar_w//2}" y="{y - 4}" '
                f'text-anchor="middle" font-size="9" fill="#333">{count}</text>'
            )
        # X label
        label_x = x + bar_w // 2
        label_y = margin_top + chart_h + 15
        svg_parts.append(
            f'<text x="{label_x}" y="{label_y}" '
            f'text-anchor="middle" font-size="8" fill="#666" '
            f'transform="rotate(-30,{label_x},{label_y})">'
            f'{edges[idx]:.1f}</text>'
        )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_dashboard() -> str:
    """Generate the complete HTML dashboard."""
    stats = load_stats()
    entries = load_entries()

    # Extract data for charts
    proj_iters = [e.get("projection_iterations", 0) for e in entries]
    surr_residuals = [e.get("surrogate_residual", 0) for e in entries]
    proj_residuals = [e.get("projection_residual", 0) for e in entries]
    runtimes = [e.get("runtime_ms", 0) for e in entries]
    converged = [e.get("converged", False) for e in entries]

    # Routing distribution
    routing_dist = stats.get("routing_distribution", {})
    # Runtime distribution
    runtime_dist = stats.get("runtime_distribution", {})
    # Family convergence
    family_conv = stats.get("family_convergence", {})

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CORE v3.2 — Operational Dashboard</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", monospace;
    background: #1a1a2e;
    color: #e0e0e0;
    margin: 0;
    padding: 20px;
}}
.container {{
    max-width: 960px;
    margin: 0 auto;
}}
h1 {{
    color: #4a90d9;
    border-bottom: 2px solid #4a90d9;
    padding-bottom: 10px;
}}
h2 {{
    color: #5cb85c;
    margin-top: 30px;
}}
.grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}}
.card {{
    background: #16213e;
    border: 1px solid #0f3460;
    border-radius: 8px;
    padding: 16px;
}}
.stat {{
    text-align: center;
    padding: 12px;
}}
.stat-value {{
    font-size: 28px;
    font-weight: bold;
    color: #4a90d9;
}}
.stat-label {{
    font-size: 12px;
    color: #888;
    text-transform: uppercase;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
}}
th, td {{
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #0f3460;
}}
th {{
    color: #5cb85c;
}}
.chart-container {{
    text-align: center;
    margin: 10px 0;
}}
.footer {{
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #0f3460;
    font-size: 12px;
    color: #666;
}}
</style>
</head>
<body>
<div class="container">
<h1>CORE v3.2 — Operational Dashboard</h1>
<p>Generated: {datetime.now(timezone.utc).isoformat()}</p>

<div class="grid">
<div class="card stat">
<div class="stat-value">{stats.get("total_tasks", 0)}</div>
<div class="stat-label">Total Tasks</div>
</div>
<div class="card stat">
<div class="stat-value">{stats.get("convergence_count", 0)}</div>
<div class="stat-label">Converged</div>
</div>
<div class="card stat">
<div class="stat-value">{stats.get("degraded_count", 0)}</div>
<div class="stat-label">Degraded</div>
</div>
<div class="card stat">
<div class="stat-value">{stats.get("avg_runtime_ms", 0):.2f} ms</div>
<div class="stat-label">Avg Runtime</div>
</div>
</div>

<h2>Projection Iterations</h2>
<div class="chart-container">
{svg_histogram(proj_iters, "Projection Iterations Distribution", bins=10, bar_color="#4a90d9")}
</div>
<div class="grid">
<div class="card stat">
<div class="stat-value">{stats.get("avg_projection_iterations", 0):.1f}</div>
<div class="stat-label">Avg Iterations</div>
</div>
<div class="card stat">
<div class="stat-value">{sum(1 for c in converged if c)}</div>
<div class="stat-label">Converged</div>
</div>
</div>

<h2>Residuals</h2>
<div class="chart-container">
{svg_histogram([r for r in proj_residuals if r > 0], "Projection Residual Distribution", bins=10, bar_color="#5cb85c")}
</div>
<div class="grid">
<div class="card stat">
<div class="stat-value">{stats.get("avg_surrogate_residual", 0):.2e}</div>
<div class="stat-label">Avg Surrogate Residual</div>
</div>
<div class="card stat">
<div class="stat-value">{stats.get("avg_projection_residual", 0):.2e}</div>
<div class="stat-label">Avg Projection Residual</div>
</div>
</div>

<h2>Routing Distribution</h2>
<div class="chart-container">
{svg_bar_chart(routing_dist, "Routing Decisions", bar_color="#f0ad4e")}
</div>

<h2>Runtime Distribution</h2>
<div class="chart-container">
{svg_bar_chart(runtime_dist, "Runtime Buckets", bar_color="#d9534f")}
</div>

<h2>Convergence by Family (System Size)</h2>
<table>
<tr><th>Size</th><th>Total</th><th>Converged</th><th>Rate</th><th>Avg Iters</th></tr>
"""

    for size_key, family_data in sorted(family_conv.items()):
        total = family_data.get("total", 0)
        conv = family_data.get("converged", 0)
        rate = f"{100 * conv / total:.1f}%" if total > 0 else "N/A"
        avg_iters = family_data.get("avg_iters", 0.0)
        html += (
            f"<tr><td>{size_key}</td><td>{total}</td>"
            f"<td>{conv}</td><td>{rate}</td>"
            f"<td>{avg_iters:.1f}</td></tr>\n"
        )

    html += """</table>

<h2>Failure Types</h2>
<table>
<tr><th>Type</th><th>Count</th><th>Rate</th></tr>
"""

    total = stats.get("total_tasks", 0) or 1
    degraded = stats.get("degraded_count", 0)
    non_converged = total - stats.get("convergence_count", 0)
    html += (
        f"<tr><td>Degraded Execution</td><td>{degraded}</td>"
        f"<td>{100*degraded/total:.1f}%</td></tr>\n"
        f"<tr><td>Non-Converged Projection</td><td>{non_converged}</td>"
        f"<td>{100*non_converged/total:.1f}%</td></tr>\n"
    )

    html += f"""</table>

<div class="footer">
<p>CORE v3.2 Operational Dashboard — Deterministic, Auditable, Reproducible</p>
<p>Report generated by scripts/generate_operational_dashboard.py</p>
</div>

</div>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    html = generate_dashboard()

    output_path = OUTPUT_DIR / "index.html"
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Dashboard generated: {output_path}")
    print(f"  Size: {len(html):,} bytes")


if __name__ == "__main__":
    main()
