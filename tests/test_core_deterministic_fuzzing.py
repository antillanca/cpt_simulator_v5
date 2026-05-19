"""CORE v3.2 — Deterministic Fuzzing Tests.

Runs the fuzzer as a guardrail against hidden nondeterminism.
These tests must pass on every CI run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root on sys.path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core_runtime.core.specs.task_hashing import compute_task_hash
from core_runtime.domains.linear_system import (
    LinearSystemTask,
    execute_linear_system_pipeline,
)
from backend.core_runtime.task_runtime import RuntimeTask

from scripts.fuzz_runtime_deterministic import (
    FuzzReport,
    generate_fuzz_tasks,
    run_fuzz,
)


class TestDeterministicFuzzing:
    """Guardrail tests: run the fuzzer and assert zero mismatches."""

    def test_task_hash_stability(self):
        """Same task -> same hash, always (Principle 6)."""
        tasks = generate_fuzz_tasks(seed=123, count=20)
        for task in tasks:
            # Build compat task twice from same data
            metadata = {}
            for k, v in task.metadata.items():
                if isinstance(v, np.ndarray):
                    metadata[k] = v.tolist()
                else:
                    metadata[k] = v

            compat1 = RuntimeTask(
                task_id=task.task_id,
                domain=task.domain_name,
                input_artifact=task.input_artifact,
                oracle_name="LinearSystemOracle",
                surrogate_name="LinearSystemSurrogate",
                projection_enabled=True,
                metadata=metadata,
            )
            compat2 = RuntimeTask(
                task_id=task.task_id,
                domain=task.domain_name,
                input_artifact=task.input_artifact,
                oracle_name="LinearSystemOracle",
                surrogate_name="LinearSystemSurrogate",
                projection_enabled=True,
                metadata=metadata,
            )
            h1 = compute_task_hash(compat1)
            h2 = compute_task_hash(compat2)
            assert h1 == h2, f"Hash mismatch for {task.task_id}: {h1} != {h2}"

    def test_pipeline_determinism(self):
        """Same task + same budget -> same projection outcome (Principle 1)."""
        tasks = generate_fuzz_tasks(seed=456, count=10)
        budget = 50
        for task in tasks:
            r1 = execute_linear_system_pipeline(task, budget=budget)
            r2 = execute_linear_system_pipeline(task, budget=budget)

            # Trace fields must match
            for key in [
                "task_id", "domain_name", "fingerprint",
                "surrogate_method", "projection_iterations",
                "projection_converged", "projection_method",
                "evaluation_correct", "trajectory_length",
            ]:
                assert (
                    r1["trace"].get(key) == r2["trace"].get(key)
                ), f"Trace mismatch on {key} for {task.task_id}"

            # Projection solutions must be numerically identical
            assert np.allclose(
                r1["projection"]["solution"],
                r2["projection"]["solution"],
                atol=1e-14,
            ), f"Projection solution mismatch for {task.task_id}"

            # Residuals must match within machine epsilon
            assert abs(
                r1["projection"]["residual"]
                - r2["projection"]["residual"]
            ) <= 1e-12, f"Residual mismatch for {task.task_id}"

    def test_routing_stability(self):
        """Same task -> same routing decision (Principle 6)."""
        tasks = generate_fuzz_tasks(seed=789, count=10)
        for task in tasks:
            r1 = execute_linear_system_pipeline(task, budget=50)
            r2 = execute_linear_system_pipeline(task, budget=50)

            assert (
                r1["trace"]["surrogate_method"]
                == r2["trace"]["surrogate_method"]
            ), f"Surrogate routing mismatch for {task.task_id}"

            assert (
                r1["trace"]["projection_method"]
                == r2["trace"]["projection_method"]
            ), f"Projection routing mismatch for {task.task_id}"

    def test_fuzz_report_overall_pass(self):
        """Full fuzzer run must report PASS (zero mismatches)."""
        report = run_fuzz(seed=42, task_count=20, budget=50)
        assert isinstance(report, FuzzReport)
        assert report.overall_pass, (
            f"Fuzzing found {report.total_mismatches} mismatches: "
            f"hash={report.hash_mismatches}, "
            f"trace={report.trace_mismatches}, "
            f"projection={report.projection_mismatches}, "
            f"residual={report.residual_mismatches}, "
            f"routing={report.routing_mismatches}"
        )

    def test_fuzz_task_generation_determinism(self):
        """Task generation itself must be deterministic."""
        tasks1 = generate_fuzz_tasks(seed=99, count=5)
        tasks2 = generate_fuzz_tasks(seed=99, count=5)

        for t1, t2 in zip(tasks1, tasks2):
            assert t1.task_id == t2.task_id
            for key in ["A", "b"]:
                assert np.allclose(
                    t1.metadata[key], t2.metadata[key], atol=1e-15
                ), f"Task generation nondeterminism on {key} for {t1.task_id}"

    def test_fuzz_detects_nondeterministic_hash_tampering(self):
        """Verify the test framework can detect hash instability.

        We deliberately modify a task between runs to ensure the
        detection mechanism works. This is a meta-test.
        """
        tasks = generate_fuzz_tasks(seed=11, count=5)
        task = tasks[0]

        metadata = {}
        for k, v in task.metadata.items():
            if isinstance(v, np.ndarray):
                metadata[k] = v.tolist()
            else:
                metadata[k] = v

        compat = RuntimeTask(
            task_id=task.task_id,
            domain=task.domain_name,
            input_artifact=task.input_artifact,
            oracle_name="LinearSystemOracle",
            surrogate_name="LinearSystemSurrogate",
            projection_enabled=True,
            metadata=metadata,
        )
        h1 = compute_task_hash(compat)

        # Modify the task — hash MUST differ
        metadata_tampered = dict(metadata)
        metadata_tampered["b"] = [0.0] * len(metadata_tampered["b"])
        compat_tampered = RuntimeTask(
            task_id=task.task_id,
            domain=task.domain_name,
            input_artifact=task.input_artifact,
            oracle_name="LinearSystemOracle",
            surrogate_name="LinearSystemSurrogate",
            projection_enabled=True,
            metadata=metadata_tampered,
        )
        h2 = compute_task_hash(compat_tampered)

        assert h1 != h2, "Hash tampering detection failed: hashes should differ"
