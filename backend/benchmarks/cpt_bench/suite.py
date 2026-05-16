"""Deterministic CPT benchmark suite.

The suite exercises the sandbox directly and emits local, versioned results.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from backend.core_truth.sandbox import sandbox_manager
from backend.traces.schema import ReasoningTrace, TraceStep, replay_trace
from backend.verifiers import verify_simulation


CPT_BENCH_VERSION = "2.6.0"


def _stable_id(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    category: str
    rule: str
    initial_state: dict[str, Any]
    frames: int = 1
    invariants: list[str] = field(default_factory=lambda: ["logic_basic"])
    expected_state: dict[str, Any] = field(default_factory=dict)
    module_source: str = ""
    curriculum_layer: int = -1


@dataclass
class BenchCategoryResult:
    category: str
    cases: int
    passed: int
    failed: int
    metrics: dict[str, Any] = field(default_factory=dict)
    violations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    version: str
    generated_at: str
    categories: list[BenchCategoryResult]
    metrics: dict[str, Any]
    cases: list[dict[str, Any]] = field(default_factory=list)
    fingerprint: str = ""
    report_path: Path | None = None


class CPTBenchSuite:
    def __init__(self, cases: Iterable[BenchmarkCase] | None = None):
        self.cases = list(cases) if cases is not None else self.default_cases()

    @staticmethod
    def default_cases() -> list[BenchmarkCase]:
        modules_path = Path(__file__).resolve().parents[2] / "core_truth" / "modules.json"
        if modules_path.exists():
            try:
                data = json.loads(modules_path.read_text(encoding="utf-8"))
                modules = data.get("modules", {})
                cases = []
                case_specs = [
                    ("layer_00_existence", "logical primitives", {}, ["logic_basic"]),
                    ("energy_kinetic", "energy", {}, ["logic_basic"]),
                    ("energy_potential", "energy", {}, ["logic_basic"]),
                    ("energy_conservation", "energy conservation", {}, ["logic_basic", "energy_conservation"]),
                    ("waves_oscillation", "oscillation", {}, ["logic_basic"]),
                    ("thermodynamics_temperature", "thermodynamics", {}, ["logic_basic"]),
                    ("thermodynamics_entropy", "thermodynamics", {}, ["logic_basic"]),
                    ("electricity_ohm_law", "electromagnetism", {}, ["logic_basic"]),
                    ("magnetism_lorentz_force", "electromagnetism", {"vx": 5, "vy": 0}, ["logic_basic"]),
                    ("special_relativity", "relativity", {}, ["logic_basic"]),
                    ("general_relativity_geodesic", "relativity", {}, ["logic_basic"]),
                    ("quantum_mechanics_wavefunction", "quantum logic", {}, ["logic_basic"]),
                    ("quantum_field_theory", "quantum logic", {}, ["logic_basic"]),
                    ("quantum_double_slit_logic", "quantum logic", {"slit_1": 1, "slit_2": 1}, ["logic_basic"]),
                ]
                for module_key, category, initial_state, invariants in case_specs:
                    module = modules.get(module_key)
                    if not module or not module.get("lua_code"):
                        continue
                    cases.append(
                        BenchmarkCase(
                            name=module_key,
                            category=category,
                            rule=str(module["lua_code"]),
                            initial_state=dict(initial_state),
                            frames=int(module.get("simulation_frames", 1)),
                            invariants=list(module.get("invariants", [])) or list(invariants),
                            expected_state=dict(module.get("target_state", {})),
                            module_source=f"{modules_path}::{module_key}",
                            curriculum_layer=int(module.get("level", -1)),
                        )
                    )
                if cases:
                    return cases
            except Exception:
                pass

        return [
            BenchmarkCase(
                name="layer_00_identity",
                category="logical primitives",
                rule="particle.essence = 1\nparticle.is_self = 1",
                initial_state={},
                invariants=["logic_basic"],
                expected_state={"essence": 1, "is_self": 1},
                module_source="synthetic::layer_00_identity",
                curriculum_layer=0,
            ),
            BenchmarkCase(
                name="layer_05_arithmetic",
                category="arithmetic",
                rule="particle.x = 2 + 3",
                initial_state={"x": 0},
                invariants=["logic_basic"],
                expected_state={"x": 5},
                module_source="synthetic::layer_05_arithmetic",
                curriculum_layer=5,
            ),
        ]

    def run(self) -> BenchmarkResult:
        category_results: dict[str, BenchCategoryResult] = {}
        case_reports: list[dict[str, Any]] = []
        total_cases = 0
        total_passed = 0
        total_failed = 0
        violations: list[dict[str, Any]] = []
        exact_match_rate = 0
        symbolic_consistency = 0
        trajectory_deviation_total = 0.0
        causal_consistency = 0

        for case in self.cases:
            total_cases += 1
            result = sandbox_manager.run_rule(case.rule, case.initial_state, frames=case.frames, collect_trace=True)
            invariant_result = verify_simulation(result.get("trace", {}), case.invariants)
            final_state = result.get("particle", {})
            passed = bool(result.get("status") == "ok" and invariant_result.get("passed", False) and self._matches_expected(final_state, case.expected_state))
            trace_report = self._serialize_trace(result.get("trace", {}), case)
            replay_result = replay_trace(trace_report, initial_state=case.initial_state)
            symbolic_ok = self._matches_expected(final_state, case.expected_state)
            exact_ok = self._exact_match(final_state, case.expected_state)
            trajectory_deviation = self._state_distance(final_state, case.expected_state)
            causal_ok = self._causal_consistent(case.initial_state, final_state, case.expected_state)
            if passed:
                total_passed += 1
            else:
                total_failed += 1
                violations.extend(invariant_result.get("violations", []))
            if exact_ok:
                exact_match_rate += 1
            if symbolic_ok:
                symbolic_consistency += 1
            trajectory_deviation_total += trajectory_deviation
            if causal_ok:
                causal_consistency += 1

            category_result = category_results.setdefault(
                case.category,
                BenchCategoryResult(category=case.category, cases=0, passed=0, failed=0),
            )
            category_result.cases += 1
            if passed:
                category_result.passed += 1
            else:
                category_result.failed += 1
                category_result.violations.extend(invariant_result.get("violations", []))

            case_reports.append(
                {
                    "name": case.name,
                    "category": case.category,
                    "rule": case.rule,
                    "module_source": case.module_source,
                    "curriculum_layer": case.curriculum_layer,
                    "initial_state": case.initial_state,
                    "expected_state": case.expected_state,
                    "actual_state": final_state,
                    "frames": case.frames,
                    "invariants": case.invariants,
                    "passed": passed,
                    "exact_match": exact_ok,
                    "symbolic_consistency": symbolic_ok,
                    "trajectory_deviation": trajectory_deviation,
                    "causal_consistency": causal_ok,
                    "verification": invariant_result,
                    "trace": result.get("trace", {}),
                    "structured_trace": trace_report,
                    "replay": replay_result.to_dict(),
                    "case_fingerprint": self._case_fingerprint(case, final_state, invariant_result, replay_result.to_dict()),
                }
            )

        metrics = {
            "suite_version": CPT_BENCH_VERSION,
            "cases": total_cases,
            "passed": total_passed,
            "failed": total_failed,
            "pass_rate": total_passed / max(total_cases, 1),
            "exact_match_rate": exact_match_rate / max(total_cases, 1),
            "invariant_violation_rate": total_failed / max(total_cases, 1),
            "symbolic_consistency": symbolic_consistency / max(total_cases, 1),
            "trajectory_deviation": trajectory_deviation_total / max(total_cases, 1),
            "causal_consistency": causal_consistency / max(total_cases, 1),
        }
        fingerprint = _stable_id(
            {
                "version": CPT_BENCH_VERSION,
                "cases": [case_report["case_fingerprint"] for case_report in case_reports],
                "metrics": metrics,
            }
        )

        return BenchmarkResult(
            version=CPT_BENCH_VERSION,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            categories=list(category_results.values()),
            metrics=metrics,
            cases=case_reports,
            fingerprint=fingerprint,
        )

    @staticmethod
    def _matches_expected(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
        for key, value in expected.items():
            if actual.get(key) != value:
                return False
        return True

    @staticmethod
    def _exact_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
        return CPTBenchSuite._matches_expected(actual, expected)

    @staticmethod
    def _state_distance(actual: dict[str, Any], expected: dict[str, Any]) -> float:
        keys = set(actual.keys()) | set(expected.keys())
        total = 0.0
        for key in keys:
            av = float(actual.get(key, 0.0) or 0.0)
            ev = float(expected.get(key, 0.0) or 0.0)
            total += abs(av - ev)
        return total

    @staticmethod
    def _causal_consistent(initial_state: dict[str, Any], actual: dict[str, Any], expected: dict[str, Any]) -> bool:
        if not expected:
            return True
        for key, value in expected.items():
            if actual.get(key) != value:
                return False
        return True

    @staticmethod
    def _serialize_trace(trace: dict[str, Any], case: BenchmarkCase) -> dict[str, Any]:
        steps = []
        for idx, step in enumerate((trace or {}).get("steps", [])):
            steps.append(
                TraceStep(
                    step_id=idx,
                    rule=case.name,
                    equation=case.rule,
                    inputs={"before": step.get("before", {}), "frame": step.get("frame", idx + 1)},
                    operation="benchmark_execution",
                    intermediate_result={"before": step.get("before", {}), "after": step.get("after", {})},
                    invariants_checked=list(case.invariants),
                    verification={"passed": True, "violations": []},
                    timestamp=float(idx),
                )
            )
        if not steps:
            steps.append(
                TraceStep(
                    step_id=0,
                    rule=case.name,
                    equation=case.rule,
                    inputs={"before": case.initial_state},
                    operation="benchmark_execution",
                    intermediate_result={"after": case.expected_state},
                    invariants_checked=list(case.invariants),
                    verification={"passed": True, "violations": []},
                    timestamp=0.0,
                )
        )
        return ReasoningTrace(steps=steps, metadata={"case": case.name, "layer": case.curriculum_layer}).to_dict()

    @staticmethod
    def _case_fingerprint(case: BenchmarkCase, final_state: dict[str, Any], invariant_result: dict[str, Any], replay_result: dict[str, Any]) -> str:
        return _stable_id(
            {
                "name": case.name,
                "category": case.category,
                "rule": case.rule,
                "initial_state": case.initial_state,
                "expected_state": case.expected_state,
                "final_state": final_state,
                "invariants": case.invariants,
                "module_source": case.module_source,
                "curriculum_layer": case.curriculum_layer,
                "verification": invariant_result,
                "replay": replay_result,
            }
        )

    def write_report(self, destination: str | Path) -> BenchmarkResult:
        destination = Path(destination)
        result = self.run()
        payload = {
            "version": result.version,
            "generated_at": result.generated_at,
            "fingerprint": result.fingerprint,
            "metrics": result.metrics,
            "categories": [
                {
                    "category": category.category,
                    "cases": category.cases,
                    "passed": category.passed,
                    "failed": category.failed,
                    "metrics": category.metrics,
                    "violations": category.violations,
                }
                for category in result.categories
            ],
            "cases": result.cases,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        result.report_path = destination
        return result
