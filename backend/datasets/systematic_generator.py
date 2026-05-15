"""Systematic dataset generator built exclusively from the deterministic sandbox."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

from backend.core_truth.sandbox import sandbox_manager
from backend.dsl.compiler import compile_dsl


def _default_spaces():
    return {
        "force": [0, 2, 4, 6],
        "mass": [1, 2, 4],
        "dt": [0.1, 0.5],
    }


class SystematicDatasetGenerator:
    """Enumerate parameter spaces and serialize sandbox-derived samples."""

    def __init__(self, output_path: str | Path, spaces: dict[str, list[Any]] | None = None):
        self.output_path = Path(output_path)
        self.spaces = spaces or _default_spaces()

    def _parameter_grid(self):
        keys = list(self.spaces.keys())
        values = [self.spaces[key] for key in keys]
        for combo in product(*values):
            yield dict(zip(keys, combo))

    def generate(self, dsl_input: str | dict[str, Any], limit: int | None = None) -> Path:
        compiled = compile_dsl(dsl_input)
        rows = []
        for index, state in enumerate(self._parameter_grid()):
            if limit is not None and index >= limit:
                break
            result = sandbox_manager.run_rule(compiled["lua_code"], state, frames=1, collect_trace=True)
            rows.append(
                {
                    "question": f"Evaluate {compiled['metadata']['law']['name']} for state #{index}",
                    "structured_state": {"initial_state": state, "metadata": compiled["metadata"]},
                    "reasoning_trace": result.get("trace", {}).get("steps", []),
                    "answer": result.get("particle", result),
                }
            )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return self.output_path

