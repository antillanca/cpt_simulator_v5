"""CORE v3.2 — Training Sandbox Boundary Tests.

Verify that the sandbox boundary:
  - exists and is inert
  - operates on copies, never originals
  - produces only candidate artifacts
  - requires explicit promotion
  - never auto-degrades production runtime
  - all operations are auditable
"""

from __future__ import annotations

import pytest

from core_runtime.core.training_sandbox import (
    CandidateArtifact,
    PromotionLedger,
    SandboxExperiment,
    SandboxState,
    TrainingSandbox,
)


# ---------------------------------------------------------------------------
# SandboxExperiment tests
# ---------------------------------------------------------------------------

class TestSandboxExperiment:
    def test_initial_state_is_created(self):
        exp = SandboxExperiment(experiment_id="exp_01", domain_name="linear_system")
        assert exp.state == SandboxState.CREATED

    def test_start_transitions_to_running(self):
        exp = SandboxExperiment(experiment_id="exp_01", domain_name="linear_system")
        exp.start()
        assert exp.state == SandboxState.RUNNING
        assert exp.started_at != ""

    def test_cannot_start_twice(self):
        exp = SandboxExperiment(experiment_id="exp_01", domain_name="linear_system")
        exp.start()
        with pytest.raises(ValueError, match="Cannot start"):
            exp.start()

    def test_complete_transitions_to_completed(self):
        exp = SandboxExperiment(experiment_id="exp_01", domain_name="linear_system")
        exp.start()
        artifact = CandidateArtifact(
            artifact_id="a1",
            experiment_id="exp_01",
            domain_name="linear_system",
            artifact_type="test",
            payload_hash="abc123",
        )
        exp.complete(artifact)
        assert exp.state == SandboxState.COMPLETED
        assert exp.candidate_artifact is not None

    def test_fail_transitions_to_failed(self):
        exp = SandboxExperiment(experiment_id="exp_01", domain_name="linear_system")
        exp.start()
        exp.fail("something went wrong")
        assert exp.state == SandboxState.FAILED
        assert exp.error_message == "something went wrong"


# ---------------------------------------------------------------------------
# CandidateArtifact tests
# ---------------------------------------------------------------------------

class TestCandidateArtifact:
    def test_artifact_is_frozen(self):
        import dataclasses
        assert dataclasses.is_dataclass(CandidateArtifact)
        assert CandidateArtifact.__dataclass_params__.frozen is True

    def test_artifact_default_state_is_candidate(self):
        artifact = CandidateArtifact(
            artifact_id="a1",
            experiment_id="exp_01",
            domain_name="linear_system",
            artifact_type="test",
            payload_hash="abc123",
        )
        assert artifact.state == "candidate"

    def test_artifact_auto_sets_created_at(self):
        artifact = CandidateArtifact(
            artifact_id="a1",
            experiment_id="exp_01",
            domain_name="linear_system",
            artifact_type="test",
            payload_hash="abc123",
        )
        assert artifact.created_at != ""


# ---------------------------------------------------------------------------
# PromotionLedger tests
# ---------------------------------------------------------------------------

class TestPromotionLedger:
    def test_ledger_starts_empty(self):
        ledger = PromotionLedger()
        assert len(ledger.records) == 0

    def test_promote_records_decision(self):
        ledger = PromotionLedger()
        artifact = CandidateArtifact(
            artifact_id="a1",
            experiment_id="exp_01",
            domain_name="linear_system",
            artifact_type="test",
            payload_hash="abc123",
        )
        record = ledger.promote(artifact, promoted_by="test")
        assert record["artifact_id"] == "a1"
        assert "promoted_by" in record
        assert "anchor" in record
        assert len(record["anchor"]) == 64  # SHA-256 hex

    def test_reject_records_decision(self):
        ledger = PromotionLedger()
        artifact = CandidateArtifact(
            artifact_id="a1",
            experiment_id="exp_01",
            domain_name="linear_system",
            artifact_type="test",
            payload_hash="abc123",
        )
        record = ledger.reject(artifact, reason="insufficient quality")
        assert "rejected_by" in record
        assert record["reason"] == "insufficient quality"

    def test_cannot_promote_non_candidate_artifact(self):
        ledger = PromotionLedger()
        import dataclasses
        artifact = CandidateArtifact(
            artifact_id="a1",
            experiment_id="exp_01",
            domain_name="linear_system",
            artifact_type="test",
            payload_hash="abc123",
            state="promoted",
        )
        with pytest.raises(ValueError, match="Cannot promote"):
            ledger.promote(artifact)

    def test_promoted_artifact_ids(self):
        ledger = PromotionLedger()
        a1 = CandidateArtifact(
            artifact_id="a1", experiment_id="e1",
            domain_name="linear_system", artifact_type="test",
            payload_hash="h1",
        )
        a2 = CandidateArtifact(
            artifact_id="a2", experiment_id="e2",
            domain_name="linear_system", artifact_type="test",
            payload_hash="h2",
        )
        ledger.promote(a1)
        ledger.reject(a2)
        assert "a1" in ledger.promoted_artifact_ids
        assert "a2" not in ledger.promoted_artifact_ids


# ---------------------------------------------------------------------------
# TrainingSandbox tests
# ---------------------------------------------------------------------------

class TestTrainingSandbox:
    def test_sandbox_creates_experiment(self):
        sandbox = TrainingSandbox()
        exp = sandbox.create_experiment("exp_01", "linear_system")
        assert exp.experiment_id == "exp_01"
        assert exp.domain_name == "linear_system"

    def test_sandbox_prevents_duplicate_experiment_ids(self):
        sandbox = TrainingSandbox()
        sandbox.create_experiment("exp_01", "linear_system")
        with pytest.raises(ValueError, match="already exists"):
            sandbox.create_experiment("exp_01", "linear_system")

    def test_sandbox_runs_placeholder_experiment(self):
        sandbox = TrainingSandbox()
        sandbox.create_experiment("exp_01", "linear_system")
        artifact = sandbox.run_experiment("exp_01")
        assert artifact.state == "candidate"
        assert artifact.artifact_type == "sandbox_candidate"

    def test_sandbox_runs_custom_experiment(self):
        sandbox = TrainingSandbox()
        sandbox.create_experiment("exp_01", "linear_system")
        result = sandbox.run_experiment("exp_01", experiment_fn=lambda s: {"loss": 0.5})
        assert result.payload_hash != ""

    def test_sandbox_operates_on_surrogate_copy(self):
        """Sandbox must never modify the original surrogate."""
        sandbox = TrainingSandbox()
        original_surrogate = {"weights": [1.0, 2.0, 3.0]}
        sandbox.create_experiment("exp_01", "linear_system", surrogate=original_surrogate)

        # Modify inside experiment — should not affect original
        def modify_fn(s):
            if s is not None:
                s["weights"][0] = 999.0
            return {"modified": True}

        sandbox.run_experiment("exp_01", experiment_fn=modify_fn)
        assert original_surrogate["weights"][0] == 1.0

    def test_sandbox_promotion_requires_explicit_action(self):
        sandbox = TrainingSandbox()
        sandbox.create_experiment("exp_01", "linear_system")
        artifact = sandbox.run_experiment("exp_01")

        # Not promoted yet
        assert not sandbox.is_artifact_promoted(artifact.artifact_id)

        # Promote explicitly
        record = sandbox.promote_artifact(artifact, promoted_by="test_user")
        assert record["artifact_id"] == artifact.artifact_id
        assert sandbox.is_artifact_promoted(artifact.artifact_id)

    def test_sandbox_rejection_is_recorded(self):
        sandbox = TrainingSandbox()
        sandbox.create_experiment("exp_01", "linear_system")
        artifact = sandbox.run_experiment("exp_01")
        record = sandbox.reject_artifact(artifact, reason="quality check failed")
        assert record["reason"] == "quality check failed"
        assert not sandbox.is_artifact_promoted(artifact.artifact_id)

    def test_sandbox_experiment_failure(self):
        sandbox = TrainingSandbox()
        sandbox.create_experiment("exp_01", "linear_system")

        def failing_fn(s):
            raise RuntimeError("training crashed")

        with pytest.raises(RuntimeError, match="training crashed"):
            sandbox.run_experiment("exp_01", experiment_fn=failing_fn)

        exp = sandbox.experiments["exp_01"]
        assert exp.state == SandboxState.FAILED

    def test_sandbox_ledger_is_auditable(self):
        """All operations leave a trace in the ledger."""
        sandbox = TrainingSandbox()
        sandbox.create_experiment("exp_01", "linear_system")
        artifact = sandbox.run_experiment("exp_01")
        sandbox.promote_artifact(artifact)

        assert len(sandbox.ledger.records) == 1
        record = sandbox.ledger.records[0]
        assert "anchor" in record
        assert len(record["anchor"]) == 64

    def test_sandbox_does_not_auto_degrade_production(self):
        """The sandbox never automatically affects the production runtime."""
        sandbox = TrainingSandbox()
        sandbox.create_experiment("exp_01", "linear_system")
        artifact = sandbox.run_experiment("exp_01")

        # After running, the artifact is still just a candidate
        assert artifact.state == "candidate"
        # No promotion happened automatically
        assert not sandbox.is_artifact_promoted(artifact.artifact_id)
        # The ledger has no records of automatic promotion
        assert len(sandbox.ledger.records) == 0
