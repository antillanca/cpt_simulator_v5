"""CORE v3.2 — Training Sandbox Boundary.

This is ONLY a boundary, not a learning system. Responsibilities:
  - Load a copy of a surrogate
  - Run offline experiments in isolation
  - Produce a candidate artifact
  - Require explicit promotion before production use

No training logic is required yet. Only the sandbox boundary and contract.

IMPORTANT:
  - The production runtime NEVER degrades automatically due to sandbox
  - Sandbox artifacts are NOT live until explicitly promoted
  - All sandbox operations are isolated from the production runtime
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Sandbox states
# ---------------------------------------------------------------------------

class SandboxState(Enum):
    """Lifecycle states for a sandbox experiment."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PROMOTED = "promoted"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Candidate artifact
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateArtifact:
    """An artifact produced by a sandbox experiment.

    This artifact is inert until explicitly promoted. The production
    runtime will never automatically use a candidate artifact.
    """
    artifact_id: str
    experiment_id: str
    domain_name: str
    artifact_type: str
    payload_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    state: str = "candidate"

    def __post_init__(self) -> None:
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Sandbox experiment record
# ---------------------------------------------------------------------------

@dataclass
class SandboxExperiment:
    """Record of a sandbox experiment.

    Tracks the lifecycle of an isolated experiment that produces
    a candidate artifact. The experiment itself is inert — it does
    not modify production state.
    """
    experiment_id: str
    domain_name: str
    state: SandboxState = SandboxState.CREATED
    surrogate_copy: Any = None
    candidate_artifact: CandidateArtifact | None = None
    started_at: str = ""
    completed_at: str = ""
    error_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Mark the experiment as running."""
        if self.state != SandboxState.CREATED:
            raise ValueError(f"Cannot start experiment in state {self.state}")
        self.state = SandboxState.RUNNING
        self.started_at = datetime.now(timezone.utc).isoformat()

    def complete(self, artifact: CandidateArtifact) -> None:
        """Mark the experiment as completed with a candidate artifact."""
        if self.state != SandboxState.RUNNING:
            raise ValueError(f"Cannot complete experiment in state {self.state}")
        self.state = SandboxState.COMPLETED
        self.candidate_artifact = artifact
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def fail(self, error: str) -> None:
        """Mark the experiment as failed."""
        self.state = SandboxState.FAILED
        self.error_message = error
        self.completed_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Promotion ledger
# ---------------------------------------------------------------------------

class PromotionLedger:
    """Immutable ledger of promotion decisions.

    Every promotion is recorded with a SHA-256 anchor.
    No promotion can be silently overwritten.
    """

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def promote(
        self,
        artifact: CandidateArtifact,
        promoted_by: str = "manual",
    ) -> dict[str, Any]:
        """Record a promotion decision.

        The artifact state is checked — only COMPLETED experiments
        can be promoted.
        """
        if artifact.state != "candidate":
            raise ValueError(
                f"Cannot promote artifact in state '{artifact.state}'. "
                f"Only 'candidate' artifacts can be promoted."
            )

        record = {
            "artifact_id": artifact.artifact_id,
            "experiment_id": artifact.experiment_id,
            "domain_name": artifact.domain_name,
            "artifact_type": artifact.artifact_type,
            "payload_hash": artifact.payload_hash,
            "promoted_by": promoted_by,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "anchor": "",  # filled below
        }
        record["anchor"] = hashlib.sha256(
            json.dumps(record, sort_keys=True, default=str).encode()
        ).hexdigest()

        self._records.append(record)
        return record

    def reject(
        self,
        artifact: CandidateArtifact,
        reason: str = "",
        rejected_by: str = "manual",
    ) -> dict[str, Any]:
        """Record a rejection decision."""
        record = {
            "artifact_id": artifact.artifact_id,
            "experiment_id": artifact.experiment_id,
            "rejected_by": rejected_by,
            "reason": reason,
            "rejected_at": datetime.now(timezone.utc).isoformat(),
            "anchor": "",
        }
        record["anchor"] = hashlib.sha256(
            json.dumps(record, sort_keys=True, default=str).encode()
        ).hexdigest()

        self._records.append(record)
        return record

    @property
    def records(self) -> list[dict[str, Any]]:
        """Return a copy of all promotion/rejection records."""
        return list(self._records)

    @property
    def promoted_artifact_ids(self) -> list[str]:
        """Return IDs of all promoted artifacts."""
        return [
            r["artifact_id"] for r in self._records
            if "promoted_by" in r
        ]


# ---------------------------------------------------------------------------
# Training Sandbox
# ---------------------------------------------------------------------------

class TrainingSandbox:
    """Boundary for future training experiments.

    This sandbox provides isolation guarantees:
      1. It operates on a COPY of the surrogate, never the original
      2. It produces CANDIDATE artifacts, never production artifacts
      3. Candidate artifacts require EXPLICIT promotion
      4. The production runtime never reads sandbox artifacts
      5. All operations are auditable via the promotion ledger

    The sandbox does NOT contain training logic. It only provides
    the boundary and contract for future training systems.
    """

    def __init__(self) -> None:
        self._experiments: dict[str, SandboxExperiment] = {}
        self._ledger = PromotionLedger()

    def create_experiment(
        self,
        experiment_id: str,
        domain_name: str,
        surrogate: Any = None,
    ) -> SandboxExperiment:
        """Create a new isolated experiment.

        If a surrogate is provided, a deep copy is made so the
        original is never modified.
        """
        if experiment_id in self._experiments:
            raise ValueError(f"Experiment {experiment_id} already exists")

        surrogate_copy = copy.deepcopy(surrogate) if surrogate else None

        experiment = SandboxExperiment(
            experiment_id=experiment_id,
            domain_name=domain_name,
            surrogate_copy=surrogate_copy,
        )
        self._experiments[experiment_id] = experiment
        return experiment

    def run_experiment(
        self,
        experiment_id: str,
        experiment_fn: Any = None,
    ) -> CandidateArtifact:
        """Run an experiment in isolation.

        The experiment_fn is a callable that receives the surrogate copy
        and returns a candidate artifact payload dict.

        If no experiment_fn is provided, a placeholder artifact is
        created (the sandbox boundary exists even without training).
        """
        experiment = self._experiments.get(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment.start()

        try:
            if experiment_fn is not None:
                payload = experiment_fn(experiment.surrogate_copy)
            else:
                payload = {"placeholder": True, "no_training_logic": True}

            payload_json = json.dumps(payload, sort_keys=True, default=str)
            payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()

            artifact = CandidateArtifact(
                artifact_id=f"artifact_{experiment_id}",
                experiment_id=experiment_id,
                domain_name=experiment.domain_name,
                artifact_type="sandbox_candidate",
                payload_hash=payload_hash,
                metadata={"sandbox_boundary": True},
            )

            experiment.complete(artifact)
            return artifact

        except Exception as e:
            experiment.fail(str(e))
            raise

    def promote_artifact(
        self,
        artifact: CandidateArtifact,
        promoted_by: str = "manual",
    ) -> dict[str, Any]:
        """Promote a candidate artifact to production.

        This is the ONLY way a sandbox artifact can affect production.
        The promotion is recorded in the immutable ledger.
        """
        return self._ledger.promote(artifact, promoted_by=promoted_by)

    def reject_artifact(
        self,
        artifact: CandidateArtifact,
        reason: str = "",
        rejected_by: str = "manual",
    ) -> dict[str, Any]:
        """Reject a candidate artifact."""
        return self._ledger.reject(artifact, reason=reason, rejected_by=rejected_by)

    @property
    def ledger(self) -> PromotionLedger:
        """Access the promotion ledger."""
        return self._ledger

    @property
    def experiments(self) -> dict[str, SandboxExperiment]:
        """Return a copy of all experiments."""
        return dict(self._experiments)

    def is_artifact_promoted(self, artifact_id: str) -> bool:
        """Check if an artifact has been promoted."""
        return artifact_id in self._ledger.promoted_artifact_ids
