"""Tests for CPT v2.7: dataset export contract, manifest, sharding, loader,
training scaffold, evaluation harness, and fingerprint determinism.

All tests run locally with no GPU or external dependencies.
"""

import json
import tempfile
from pathlib import Path

import pytest

# --- Export Contract tests ---

from backend.datasets.export_contract import (
    EXPORT_SCHEMA_VERSION,
    STRICT_EXPORT_FIELDS,
    export_fingerprint,
    normalize_export_row,
    row_to_contract,
    validate_export_row,
)


def _make_valid_row(**overrides) -> dict:
    """Create a minimal valid v2.7 export row."""
    row = {
        "sample_id": "abc123",
        "question": "What is 1+1?",
        "structured_state": {"initial_state": {}, "module": "test_mod"},
        "reasoning_trace": [{"step_id": 0, "rule": "test"}],
        "equations_used": ["x = 1"],
        "invariants_checked": ["logic_basic"],
        "final_answer": {"x": 1},
        "verification_status": {"passed": True, "violations": []},
        "module_source": "test::mod",
        "curriculum_layer": 0,
        "seed": 42,
        "timestamp": 42.0,
        "dataset_version": "2.7.0",
        "snapshot_hash": "snap123",
        "module_hash": "mod123",
    }
    row.update(overrides)
    return row


def test_export_schema_version_is_270():
    assert EXPORT_SCHEMA_VERSION == "2.7.0"


def test_strict_export_fields_count():
    assert len(STRICT_EXPORT_FIELDS) == 15


def test_validate_valid_row():
    errors = validate_export_row(_make_valid_row())
    assert errors == []


def test_validate_missing_field():
    row = _make_valid_row()
    del row["sample_id"]
    errors = validate_export_row(row)
    assert any("sample_id" in e for e in errors)


def test_validate_wrong_type_curriculum_layer():
    row = _make_valid_row(curriculum_layer="zero")
    errors = validate_export_row(row)
    assert any("curriculum_layer" in e for e in errors)


def test_validate_wrong_type_seed():
    row = _make_valid_row(seed="forty-two")
    errors = validate_export_row(row)
    assert any("seed" in e for e in errors)


def test_normalize_adds_missing_fields():
    v26_row = {
        "sample_id": "abc",
        "question": "q",
        "structured_state": {},
        "reasoning_trace": [],
        "equations_used": [],
        "invariants_checked": [],
        "final_answer": {},
        "verification_status": {},
        "module_source": "m",
        "curriculum_layer": 0,
        "seed": 1,
        "timestamp": 0.0,
    }
    result = normalize_export_row(v26_row, dataset_version="2.7.0", snapshot_hash="snap", module_hash="mod")
    assert result["dataset_version"] == "2.7.0"
    assert result["snapshot_hash"] == "snap"
    assert result["module_hash"] == "mod"


def test_export_fingerprint_is_deterministic():
    row = _make_valid_row()
    fp1 = export_fingerprint(row)
    fp2 = export_fingerprint(row)
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex


def test_export_fingerprint_changes_on_content_change():
    row_a = _make_valid_row(final_answer={"x": 1})
    row_b = _make_valid_row(final_answer={"x": 2})
    assert export_fingerprint(row_a) != export_fingerprint(row_b)


def test_row_to_contract_raises_on_invalid():
    bad_row = {"sample_id": "x"}  # Missing most fields
    with pytest.raises(ValueError, match="Missing required field"):
        row_to_contract(bad_row, "2.7.0", "snap", "mod")


def test_row_to_contract_succeeds_on_valid():
    row = _make_valid_row()
    result = row_to_contract(row, "2.7.0", "snap", "mod")
    assert "row_fingerprint" in result
    assert result["dataset_version"] == "2.7.0"


# --- Manifest tests ---

from backend.datasets.manifest import DatasetManifest, validate_manifest


def test_manifest_compute_fingerprint():
    m = DatasetManifest(generation_seed=42, record_count=10, snapshot_hash="abc")
    fp = m.compute_fingerprint()
    assert len(fp) == 64
    assert fp != ""


def test_manifest_fingerprint_is_deterministic():
    m1 = DatasetManifest(generation_seed=42, record_count=10, snapshot_hash="abc")
    m2 = DatasetManifest(generation_seed=42, record_count=10, snapshot_hash="abc")
    assert m1.compute_fingerprint() == m2.compute_fingerprint()


def test_manifest_fingerprint_changes_with_seed():
    m1 = DatasetManifest(generation_seed=42, record_count=10)
    m2 = DatasetManifest(generation_seed=99, record_count=10)
    assert m1.compute_fingerprint() != m2.compute_fingerprint()


def test_manifest_roundtrip_json(tmp_path):
    m = DatasetManifest(generation_seed=42, record_count=5, modules_used=["mod_a", "mod_b"], snapshot_hash="snap1")
    path = m.save(tmp_path / "test.manifest.json")
    loaded = DatasetManifest.from_file(path)
    assert loaded.generation_seed == 42
    assert loaded.record_count == 5
    assert loaded.fingerprint == m.fingerprint


def test_validate_manifest_good():
    m = DatasetManifest(generation_seed=1, record_count=1, snapshot_hash="x", module_hash="y")
    errors = validate_manifest(m.to_dict())
    assert errors == []


def test_validate_manifest_missing_field():
    errors = validate_manifest({"dataset_version": "2.7.0"})
    assert len(errors) > 0


def test_validate_manifest_detects_fingerprint_tamper():
    m = DatasetManifest(generation_seed=1, record_count=1, snapshot_hash="x")
    data = m.to_dict()
    data["fingerprint"] = "tampered_bad_hash"
    errors = validate_manifest(data)
    assert any("Fingerprint mismatch" in e for e in errors)


# --- Sharding tests ---

from backend.datasets.sharding import (
    iter_dataset_from_shards,
    reassemble_dataset,
    save_shard_manifest,
    shard_dataset,
    validate_shard_manifest,
)


def _make_jsonl_file(tmp_path, n=25) -> Path:
    p = tmp_path / "data.jsonl"
    lines = []
    for i in range(n):
        row = {"sample_id": f"s{i:04d}", "question": f"Q{i}", "final_answer": {"x": i}, "curriculum_layer": i % 5, "seed": 42, "timestamp": float(i)}
        lines.append(json.dumps(row, sort_keys=True))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_shard_dataset_creates_shards(tmp_path):
    jsonl = _make_jsonl_file(tmp_path, n=25)
    shard_dir = tmp_path / "shards"
    manifest = shard_dataset(jsonl, shard_dir, shard_size=10)
    assert manifest["total_records"] == 25
    assert len(manifest["shards"]) == 3  # 10 + 10 + 5
    assert manifest["shards"][0]["records"] == 10
    assert manifest["shards"][2]["records"] == 5
    assert "fingerprint" in manifest


def test_shard_manifest_valid(tmp_path):
    jsonl = _make_jsonl_file(tmp_path, n=25)
    shard_dir = tmp_path / "shards"
    manifest = shard_dataset(jsonl, shard_dir, shard_size=10)
    errors = validate_shard_manifest(manifest)
    assert errors == []


def test_shard_manifest_detects_tamper(tmp_path):
    jsonl = _make_jsonl_file(tmp_path, n=10)
    shard_dir = tmp_path / "shards"
    manifest = shard_dataset(jsonl, shard_dir, shard_size=5)
    manifest["fingerprint"] = "bad"
    errors = validate_shard_manifest(manifest)
    assert any("Fingerprint mismatch" in e for e in errors)


def test_reassemble_matches_original(tmp_path):
    jsonl = _make_jsonl_file(tmp_path, n=25)
    shard_dir = tmp_path / "shards"
    manifest = shard_dataset(jsonl, shard_dir, shard_size=10)

    reassembled = reassemble_dataset(shard_dir, manifest, tmp_path / "reassembled.jsonl")
    original = json.loads(jsonl.read_text().splitlines()[0], )
    reassembled_first = json.loads(reassembled.read_text().splitlines()[0])

    # Check same record count
    assert len(reassembled.read_text().splitlines()) == 25
    assert reassembled_first["sample_id"] == original["sample_id"]


def test_iter_dataset_from_shards(tmp_path):
    jsonl = _make_jsonl_file(tmp_path, n=15)
    shard_dir = tmp_path / "shards"
    manifest = shard_dataset(jsonl, shard_dir, shard_size=5)
    records = list(iter_dataset_from_shards(shard_dir, manifest))
    assert len(records) == 15


# --- Loader tests ---

from backend.datasets.loader import DatasetLoadError, load_jsonl, load_with_manifest, upgrade_v26_row


def test_load_jsonl_valid(tmp_path):
    jsonl = _make_jsonl_file(tmp_path, n=5)
    rows = load_jsonl(jsonl, validate=False)
    assert len(rows) == 5


def test_load_jsonl_missing_file(tmp_path):
    with pytest.raises(DatasetLoadError, match="not found"):
        load_jsonl(tmp_path / "nonexistent.jsonl")


def test_load_jsonl_with_validation(tmp_path):
    # Write rows that pass contract validation
    rows = [_make_valid_row(sample_id=f"s{i:04d}") for i in range(3)]
    p = tmp_path / "valid.jsonl"
    p.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    loaded = load_jsonl(p, validate=True)
    assert len(loaded) == 3


def test_upgrade_v26_row():
    v26 = {"sample_id": "x", "question": "q", "structured_state": {}, "reasoning_trace": [],
           "equations_used": [], "invariants_checked": [], "final_answer": {}, "verification_status": {},
           "module_source": "m", "curriculum_layer": 0, "seed": 1, "timestamp": 0.0}
    upgraded = upgrade_v26_row(v26, snapshot_hash="snap", module_hash="mod")
    assert upgraded["dataset_version"] == "2.7.0"
    assert upgraded["snapshot_hash"] == "snap"
    assert upgraded["module_hash"] == "mod"


# --- Dataloader split tests ---

from backend.neural.dataloaders import DatasetShardLoader


def test_dataloader_split_deterministic(tmp_path):
    jsonl = _make_jsonl_file(tmp_path, n=20)
    loader1 = DatasetShardLoader(dataset_path=jsonl, seed=42)
    loader2 = DatasetShardLoader(dataset_path=jsonl, seed=42)
    assert loader1.train_count == loader2.train_count
    assert loader1.eval_count == loader2.eval_count
    # Same records
    assert [r["sample_id"] for r in loader1.train] == [r["sample_id"] for r in loader2.train]


def test_dataloader_different_seeds_different_splits(tmp_path):
    jsonl = _make_jsonl_file(tmp_path, n=20)
    loader1 = DatasetShardLoader(dataset_path=jsonl, seed=42)
    loader2 = DatasetShardLoader(dataset_path=jsonl, seed=99)
    assert [r["sample_id"] for r in loader1.train] != [r["sample_id"] for r in loader2.train]


def test_dataloader_split_ratio(tmp_path):
    jsonl = _make_jsonl_file(tmp_path, n=100)
    loader = DatasetShardLoader(dataset_path=jsonl, seed=42, train_split=0.8)
    assert loader.train_count == 80
    assert loader.eval_count == 20


# --- Evaluation harness tests ---

from backend.validation.model_eval import (
    ModelEvaluator,
    ModelEvaluationResult,
    answer_consistency,
    exact_match_rate,
    invariant_violation_rate,
    replay_consistency,
    token_or_struct_match_rate,
    trace_consistency,
    trajectory_deviation,
)


def test_exact_match_rate_perfect():
    preds = [{"x": 1}, {"y": 2}]
    oracles = [{"x": 1}, {"y": 2}]
    assert exact_match_rate(preds, oracles) == 1.0


def test_exact_match_rate_zero():
    preds = [{"x": 1}]
    oracles = [{"x": 2}]
    assert exact_match_rate(preds, oracles) == 0.0


def test_token_or_struct_match():
    preds = [{"x": 1, "y": "a"}]
    oracles = [{"x": 2, "y": "b"}]
    assert token_or_struct_match_rate(preds, oracles) == 1.0  # Same keys, same types


def test_invariant_violation_rate_no_violations():
    preds = [{"verification_status": {"passed": True}}]
    invs = [["logic_basic"]]
    assert invariant_violation_rate(preds, invs) == 0.0


def test_invariant_violation_rate_with_violations():
    preds = [{"verification_status": {"passed": False}}]
    invs = [["logic_basic"]]
    assert invariant_violation_rate(preds, invs) == 1.0


def test_trajectory_deviation_zero():
    preds = [{"reasoning_trace": [1, 2, 3]}]
    oracles = [[1, 2, 3]]
    assert trajectory_deviation(preds, oracles) == 0.0


def test_trajectory_deviation_nonzero():
    preds = [{"reasoning_trace": [1, 2]}]
    oracles = [[1, 2, 3]]
    assert trajectory_deviation(preds, oracles) == 1.0


def test_model_evaluator_full_evaluation():
    oracle_records = [
        {"final_answer": {"x": 1}, "reasoning_trace": [{"step": 1}], "invariants_checked": ["logic"],
         "curriculum_layer": 0, "module_source": "mod_a", "module_key": "mod_a", "verification_status": {"passed": True}},
        {"final_answer": {"x": 2}, "reasoning_trace": [{"step": 1}], "invariants_checked": ["logic"],
         "curriculum_layer": 1, "module_source": "mod_b", "module_key": "mod_b", "verification_status": {"passed": True}},
    ]
    predictions = [dict(r) for r in oracle_records]  # Perfect predictions

    evaluator = ModelEvaluator(model_type="test")
    result = evaluator.evaluate(predictions, oracle_records)
    assert result.total_samples == 2
    assert result.metrics["exact_match_rate"] == 1.0
    assert result.metrics["answer_consistency"] == 1.0
    assert 0 in result.by_layer
    assert 1 in result.by_layer


def test_evaluation_result_fingerprint(tmp_path):
    result = ModelEvaluationResult(model_type="test", total_samples=10, metrics={"exact_match_rate": 0.5})
    fp = result.compute_fingerprint()
    assert len(fp) == 64
    # Save and reload
    path = result.save(tmp_path / "eval.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["fingerprint"] == fp


# --- Fingerprint stability regression tests ---

def test_same_seed_same_fingerprint():
    row = _make_valid_row()
    fp1 = export_fingerprint(row)
    fp2 = export_fingerprint(row)
    assert fp1 == fp2


def test_manifest_fingerprint_stable():
    m = DatasetManifest(generation_seed=7, record_count=3, snapshot_hash="h", module_hash="m")
    fp1 = m.compute_fingerprint()
    m2 = DatasetManifest(generation_seed=7, record_count=3, snapshot_hash="h", module_hash="m")
    fp2 = m2.compute_fingerprint()
    assert fp1 == fp2


def test_eval_result_fingerprint_stable():
    r1 = ModelEvaluationResult(model_type="x", total_samples=5, metrics={"exact_match_rate": 0.8})
    r2 = ModelEvaluationResult(model_type="x", total_samples=5, metrics={"exact_match_rate": 0.8})
    assert r1.compute_fingerprint() == r2.compute_fingerprint()


def test_eval_result_fingerprint_detects_change():
    r1 = ModelEvaluationResult(model_type="x", total_samples=5, metrics={"exact_match_rate": 0.8})
    r2 = ModelEvaluationResult(model_type="x", total_samples=5, metrics={"exact_match_rate": 0.9})
    assert r1.compute_fingerprint() != r2.compute_fingerprint()


# --- Kaggle hooks tests ---

from backend.neural.kaggle_hooks import TrainingProfile, package_dataset_for_upload


def test_training_profile_roundtrip(tmp_path):
    p = TrainingProfile(name="test", model_type="gnn", epochs=5, lr=1e-3)
    path = p.save(tmp_path / "profile.json")
    loaded = TrainingProfile.from_file(path)
    assert loaded.name == "test"
    assert loaded.model_type == "gnn"
    assert loaded.epochs == 5


def test_package_dataset_creates_files(tmp_path):
    jsonl = _make_jsonl_file(tmp_path, n=5)
    output_dir = package_dataset_for_upload(jsonl, output_dir=tmp_path / "kaggle_pkg")
    assert (output_dir / jsonl.name).exists()
    assert (output_dir / "training_profile.json").exists()
    assert (output_dir / "README.md").exists()


# --- Backward compatibility test ---

def test_v26_dataset_auto_upgrade(tmp_path):
    """Verify that v2.6 oracle output can be loaded and upgraded to v2.7."""
    v26_rows = [
        {
            "sample_id": "old1",
            "question": "Q",
            "structured_state": {"initial_state": {}, "module": "mod"},
            "reasoning_trace": [],
            "equations_used": [],
            "invariants_checked": [],
            "final_answer": {"x": 1},
            "verification_status": {"passed": True},
            "module_source": "mod",
            "curriculum_layer": 0,
            "seed": 42,
            "timestamp": 0.0,
            "module_version": "v1",
            "row_fingerprint": "old_fp",
        }
    ]
    p = tmp_path / "v26.jsonl"
    p.write_text("\n".join(json.dumps(r, sort_keys=True) for r in v26_rows) + "\n", encoding="utf-8")

    # Load without strict validation (v2.6 rows lack v2.7 fields)
    rows = load_jsonl(p, validate=False)
    assert len(rows) == 1

    # Upgrade
    upgraded = upgrade_v26_row(rows[0], snapshot_hash="snap_v27", module_hash="mod_v27")
    errors = validate_export_row(upgraded)
    assert errors == [], f"Upgraded row should be valid, got: {errors}"
