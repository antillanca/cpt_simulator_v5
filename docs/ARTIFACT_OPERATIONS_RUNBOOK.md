# CPT Artifact Operations Runbook

This runbook covers the default governed workflow for checkpoints, datasets, and evaluation reports.

## Policy File

- Default policy: `configs/artifact_policy.yaml`
- Policy schema: `1.0`
- Enforcement defaults to strict mode unless overridden by a command flag.
- Legacy read is enabled, legacy write is disabled.

## Validate a Checkpoint

```bash
python scripts/validate_checkpoint.py --path path/to/checkpoint.pt
```

Use `--json` for machine output or `--markdown` for a compact human summary.

## Migrate a Checkpoint

```bash
python scripts/migrate_checkpoint.py --path path/to/checkpoint.pt --target-version 2.7.6
```

Use `--dry-run` to inspect the migration without rewriting the file.

## Generate an Evaluation Report

```bash
python scripts/generate_eval_report.py --input path/to/eval.json --markdown --output reports/eval.md
```

For JSON output, use `--json`. The report includes policy context, checkpoint metadata, and failure summaries.

## Compare Two Evaluation Runs

```bash
python scripts/compare_eval_reports.py --baseline reports/a.json --candidate reports/b.json
```

Add `--markdown` or `--json` if you want a specific output format.

## Generate a Dataset Export

```bash
python scripts/generate_large_dataset.py --output-dir data/oracle_run --num-samples 1000000
```

The export writes a shard manifest and a dataset manifest. Policy validation is applied when the default policy is loaded.

## Run a Retention Sweep

```bash
python scripts/retention_sweeper.py --root . --dry-run
```

Use `--execute --yes` only when you want to delete artifacts according to the policy plan.

## Generate a Retention Report

```bash
python scripts/generate_retention_report.py --root . --markdown --output reports/retention.md
```

The report shows pinned artifacts, archive candidates, reclaimable storage, and policy violations.

## Archive Artifacts

```bash
python scripts/archive_artifacts.py --path path/to/checkpoint.pt --path path/to/report.json --output bundles/export.tar.gz
```

## Export a Bundle From a Root

```bash
python scripts/export_artifact_bundle.py --root . --output bundles/root_export.tar.gz
```

## Build an Inventory Index

```bash
python scripts/build_inventory.py --workspace . --index inventory_index.json
```

## Query an Inventory Index

```bash
python scripts/query_inventory.py --index inventory_index.json --artifact-type checkpoint
```

## Generate a Workspace Summary

```bash
python scripts/generate_workspace_summary.py --index inventory_index.json --markdown --output reports/workspace_summary.md
```

## Export a Lineage Graph

```bash
python scripts/export_lineage_graph.py --index inventory_index.json --output reports/lineage_graph.json
```

## Export Inventory Metadata

```bash
python scripts/export_inventory.py --index inventory_index.json --output reports/inventory_export.json
```

## Inspect the Artifact Registry

```bash
python -c "from backend.governance.artifact_registry import ArtifactRegistry; print(ArtifactRegistry().to_dict())"
```

## Troubleshooting

- Fingerprint mismatch:
  - Re-run the validator with the same seed and policy file.
  - Confirm the checkpoint or report was not edited after generation.
  - If the artifact is legacy, migrate it explicitly before comparing fingerprints.
- Compatibility mismatch:
  - Check the `compatibility` block in `configs/artifact_policy.yaml`.
  - Legacy read may be enabled, but legacy write remains disabled.
- Missing required fields:
  - Rebuild the artifact with the current generator or migrate it through the supported upgrade path.
- Unknown artifact type:
  - The policy file is the source of truth. Add the type explicitly or route the artifact through an existing type.

## Versioning Expectations

- `v2.7.5` checkpoints may be read and migrated.
- `v2.7.6` checkpoints and reports are the current governed format.
- New artifacts should include fingerprints and provenance metadata.
- No silent coercion is performed during policy enforcement.
