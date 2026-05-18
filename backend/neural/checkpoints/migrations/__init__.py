"""Checkpoint migration graph."""

from backend.neural.checkpoints.migrations.v2_7_5_to_v2_7_6 import MigrationStepResult, migrate_payload_v275_to_v276

MIGRATION_GRAPH = {
    "2.7.5": {"2.7.6": migrate_payload_v275_to_v276},
}

__all__ = ["MigrationStepResult", "migrate_payload_v275_to_v276", "MIGRATION_GRAPH"]
