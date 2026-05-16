"""Deterministic oracle dataset generation from sandbox execution.

The generator only derives answers from the Lua sandbox. LLMs may verbalize
questions externally, but the structured state, trace, and final answer all
come from deterministic execution.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Iterable

from backend.core_truth.sandbox import sandbox_manager
from backend.traces.schema import ReasoningTrace, TraceStep
from backend.verifiers import verify_simulation


def _load_modules_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        return {"modules": {}}
    return json.loads(registry_path.read_text(encoding="utf-8"))


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _lua_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "nil"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, dict):
        parts = []
        for key in sorted(value):
            parts.append(f"[{json.dumps(str(key))}] = {_lua_literal(value[key])}")
        return "{ " + ", ".join(parts) + " }"
    if isinstance(value, list):
        return "{ " + ", ".join(_lua_literal(item) for item in value) + " }"
    return json.dumps(str(value))


def _render_assignment_rule(target_state: dict[str, Any]) -> str:
    assignments = []
    for key in sorted(target_state):
        assignments.append(f"particle.{key} = {_lua_literal(target_state[key])}")
    if not assignments:
        assignments.append("-- no-op oracle rule")
    return "\n".join(assignments)


def _default_question(module_key: str, module: dict[str, Any], parameters: dict[str, Any]) -> str:
    description = module.get("description", "").strip()
    if parameters:
        ordered = ", ".join(f"{key}={parameters[key]}" for key in sorted(parameters))
        return f"{description} ({module_key}). Parameters: {ordered}."
    return f"{description} ({module_key})."


def _equations_from_rule(rule_text: str) -> list[str]:
    return [line.strip() for line in rule_text.splitlines() if line.strip() and not line.strip().startswith("--")]


def _build_trace_export(
    rule_text: str,
    sandbox_trace: dict[str, Any],
    module_key: str,
    module_version: str,
    seed: int,
    sample_index: int,
    invariants: list[str],
    verification: dict[str, Any],
) -> ReasoningTrace:
    raw_steps = list((sandbox_trace or {}).get("steps", []))
    ordered_steps: list[TraceStep] = []
    base_timestamp = float(seed) + float(sample_index)
    for idx, step in enumerate(raw_steps):
        step_payload = {
            "step_id": idx,
            "rule": module_key,
            "equation": rule_text,
            "inputs": {
                "before": step.get("before", {}),
                "frame": step.get("frame", idx + 1),
            },
            "operation": "sandbox_execution",
            "intermediate_result": {
                "before": step.get("before", {}),
                "after": step.get("after", {}),
            },
            "invariants_checked": list(invariants),
            "verification": verification,
            "timestamp": base_timestamp + (idx / 1000.0),
        }
        ordered_steps.append(TraceStep.from_dict(step_payload))
    if not ordered_steps:
        ordered_steps.append(
            TraceStep(
                step_id=0,
                rule=module_key,
                equation=rule_text,
                inputs={"before": {}},
                operation="sandbox_execution",
                intermediate_result={"after": {}},
                invariants_checked=list(invariants),
                verification=verification,
                timestamp=base_timestamp,
            )
        )
    return ReasoningTrace(steps=ordered_steps, metadata={"module_key": module_key, "module_version": module_version, "seed": seed, "sample_index": sample_index})


def _module_executable_rule(module: dict[str, Any]) -> str | None:
    if module.get("lua_code"):
        return str(module["lua_code"])
    target_state = module.get("target_state")
    if isinstance(target_state, dict) and target_state:
        return _render_assignment_rule(target_state)
    return None


@dataclass
class OracleGenerationResult:
    output_path: Path
    manifest_path: Path
    samples_generated: int
    modules_used: list[str] = field(default_factory=list)
    seed: int = 0
    dataset_fingerprint: str = ""


class OracleDatasetGenerator:
    """Generate JSONL reasoning rows directly from sandbox execution."""

    def __init__(
        self,
        output_path: str | Path,
        modules_path: str | Path | None = None,
        seed: int = 0,
        parameter_sweeps: dict[str, list[Any]] | None = None,
        include_tabular_modules: bool = True,
    ):
        self.output_path = Path(output_path)
        self.modules_path = Path(modules_path or Path("backend/core_truth/modules.json"))
        self.seed = int(seed)
        self.parameter_sweeps = parameter_sweeps or {}
        self.include_tabular_modules = include_tabular_modules
        self.registry = _load_modules_registry(self.modules_path)

    def _module_items(self, module_keys: Iterable[str] | None = None) -> list[tuple[str, dict[str, Any]]]:
        modules = self.registry.get("modules", {})
        items = list(modules.items())
        if module_keys is not None:
            keyset = set(module_keys)
            items = [item for item in items if item[0] in keyset]
        if not self.include_tabular_modules:
            items = [item for item in items if item[1].get("engine_type") != "tabular"]
        return sorted(items, key=lambda item: (item[1].get("level", 0), item[1].get("order", 0), item[0]))

    def _parameter_grid(self) -> list[dict[str, Any]]:
        if not self.parameter_sweeps:
            return [{}]
        keys = sorted(self.parameter_sweeps)
        values = [list(self.parameter_sweeps[key]) for key in keys]
        combos = [dict(zip(keys, combo)) for combo in product(*values)]
        rng = random.Random(self.seed)
        rng.shuffle(combos)
        return combos

    def _build_rule(self, module: dict[str, Any]) -> str:
        rule = _module_executable_rule(module)
        if rule is None:
            raise ValueError("Module does not expose an executable rule or target_state.")
        return rule

    def _initial_state(self, module: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
        state = dict(module.get("initial_state", {}) or {})
        for key, value in parameters.items():
            state[key] = value
        return state

    def _sample_id(self, module_key: str, seed: int, sample_index: int, parameters: dict[str, Any]) -> str:
        payload = {"module_key": module_key, "seed": seed, "sample_index": sample_index, "parameters": parameters}
        return _stable_hash(payload)[:16]

    def _row_timestamp(self, sample_index: int) -> float:
        return float(self.seed) + (float(sample_index) / 1000.0)

    @staticmethod
    def _row_fingerprint(row: dict[str, Any]) -> str:
        payload = dict(row)
        payload.pop("row_fingerprint", None)
        return _stable_hash(payload)

    def generate_batch(
        self,
        module_keys: Iterable[str] | None = None,
        curriculum_layers: Iterable[int] | None = None,
        limit: int | None = None,
    ) -> OracleGenerationResult:
        rows: list[dict[str, Any]] = []
        modules_used: list[str] = []
        sample_index = 0
        layer_filter = set(int(layer) for layer in curriculum_layers) if curriculum_layers is not None else None

        for module_key, module in self._module_items(module_keys):
            if layer_filter is not None and int(module.get("level", -1)) not in layer_filter:
                continue
            module_version = _stable_hash(module)
            invariants = list(module.get("invariants", []))
            if not invariants:
                invariants = ["logic_basic"]
            try:
                rule_text = self._build_rule(module)
            except ValueError:
                continue
            modules_used.append(module_key)
            for parameters in self._parameter_grid():
                if limit is not None and sample_index >= limit:
                    break
                initial_state = self._initial_state(module, parameters)
                sandbox_result = sandbox_manager.run_rule(rule_text, initial_state, frames=module.get("simulation_frames", 1), collect_trace=True)
                if sandbox_result.get("status") != "ok":
                    continue

                verification = verify_simulation(sandbox_result.get("trace", {}), invariants)
                trace_export = _build_trace_export(
                    rule_text,
                    sandbox_result.get("trace", {}),
                    module_key,
                    module_version,
                    self.seed,
                    sample_index,
                    invariants,
                    verification,
                )
                row = {
                    "question": _default_question(module_key, module, parameters),
                    "structured_state": {
                        "initial_state": initial_state,
                        "parameters": parameters,
                        "module": module_key,
                        "module_version": module_version,
                    },
                    "reasoning_trace": trace_export.to_dict().get("steps", []),
                    "equations_used": _equations_from_rule(rule_text),
                    "invariants_checked": invariants,
                    "final_answer": sandbox_result.get("particle", sandbox_result),
                    "verification_status": verification,
                    "module_source": f"{self.modules_path}::{module_key}",
                    "curriculum_layer": int(module.get("level", 0)),
                    "module_version": module_version,
                    "trace_export": trace_export.to_dict(),
                    "sample_id": self._sample_id(module_key, self.seed, sample_index, parameters),
                    "seed": self.seed,
                    "timestamp": self._row_timestamp(sample_index),
                    "module_key": module_key,
                }
                row["row_fingerprint"] = self._row_fingerprint(row)
                rows.append(row)
                sample_index += 1
            if limit is not None and sample_index >= limit:
                break

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

        manifest_path = self.output_path.with_suffix(".manifest.json")
        dataset_fingerprint = _stable_hash(
            {
                "seed": self.seed,
                "modules_path": str(self.modules_path),
                "modules_used": modules_used,
                "rows": [row["row_fingerprint"] for row in rows],
            }
        )
        manifest = {
            "output_path": str(self.output_path),
            "modules_path": str(self.modules_path),
            "seed": self.seed,
            "samples_generated": len(rows),
            "modules_used": modules_used,
            "parameter_sweeps": self.parameter_sweeps,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(self.seed or 0)),
            "dataset_fingerprint": dataset_fingerprint,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        return OracleGenerationResult(
            output_path=self.output_path,
            manifest_path=manifest_path,
            samples_generated=len(rows),
            modules_used=modules_used,
            seed=self.seed,
            dataset_fingerprint=dataset_fingerprint,
        )

    def validate_output(self, dataset_path: str | Path | None = None, validation_path: str | Path | None = None) -> dict[str, Any]:
        from scripts.validation_runner import validate_dataset_file

        dataset_path = Path(dataset_path or self.output_path)
        validation_path = Path(validation_path) if validation_path is not None else dataset_path.with_suffix(".validation.json")
        return validate_dataset_file(dataset_path, validation_path=validation_path)
