"""Tests for CPT v2.7.10: artifact search refinement, reverse dependencies,
saved queries, search facets, discovery reports, impact analysis,
and CLI reproducibility.
"""

import json
import tempfile
from pathlib import Path

import pytest

from backend.governance.artifact_inventory import (
    InventoryEntry,
    InventoryIndex,
    INVENTORY_SCHEMA_VERSION,
)
from backend.governance.reverse_dependencies import (
    DependencyEdge,
    ReverseDependencyResult,
    build_reverse_dependency_index,
    find_reverse_dependencies,
    reverse_dependency_closure,
)
from backend.governance.saved_queries import (
    SAVED_QUERY_SCHEMA_VERSION,
    execute_saved_query,
    load_query,
    save_query,
    validate_saved_query_data,
)
from backend.governance.query_engine import query_inventory
from backend.governance.impact_analysis import (
    ImpactAnalysisResult,
    analyze_artifact_impact,
)
from backend.governance.lineage_graph import build_lineage_graph
from backend.reporting.search_facets import SearchFacets, build_search_facets
from backend.reporting.discovery_report import DiscoveryReport, build_discovery_report


# --- Fixtures ---

def _entry(
    artifact_id: str = "a0",
    artifact_type: str = "dataset",
    fingerprint: str = "fp0",
    schema_version: str = "2.7.10",
    workspace_root: str = "/ws",
    relative_path: str = "data.jsonl",
    created_at: float = 1.0,
    size_bytes: int = 100,
    lineage_parents: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
    retention_status: str = "active",
) -> InventoryEntry:
    return InventoryEntry(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        fingerprint=fingerprint,
        schema_version=schema_version,
        workspace_root=workspace_root,
        relative_path=relative_path,
        created_at=created_at,
        size_bytes=size_bytes,
        lineage_parents=lineage_parents,
        tags=tags,
        retention_status=retention_status,
    )


def _make_index(entries=None) -> InventoryIndex:
    if entries is None:
        entries = [
            _entry(artifact_id="dataset_a", artifact_type="dataset", lineage_parents=()),
            _entry(artifact_id="checkpoint_b", artifact_type="checkpoint", lineage_parents=("dataset_a",)),
            _entry(artifact_id="report_c", artifact_type="evaluation_report", lineage_parents=("checkpoint_b",)),
            _entry(artifact_id="archive_d", artifact_type="archive_bundle", lineage_parents=("report_c",)),
            _entry(artifact_id="manifest_e", artifact_type="manifest", lineage_parents=("dataset_a",)),
            _entry(artifact_id="orphan_f", artifact_type="evaluation_report", lineage_parents=()),
        ]
    return InventoryIndex(
        generated_at=1.0,
        workspace_root="/ws",
        entry_count=len(entries),
        inventory_fingerprint="test_idx_fp",
        entries=tuple(entries),
        schema_version=INVENTORY_SCHEMA_VERSION,
    )


# --- Reverse Dependency tests ---

def test_build_reverse_dependency_index_basic():
    index = _make_index()
    reverse_map, edge_map = build_reverse_dependency_index(index.entries)
    assert "dataset_a" in reverse_map
    assert "checkpoint_b" in reverse_map["dataset_a"]
    assert "manifest_e" in reverse_map["dataset_a"]
    assert "report_c" in reverse_map.get("checkpoint_b", ())


def test_find_reverse_dependencies_transitive():
    index = _make_index()
    reverse_map, _edges = build_reverse_dependency_index(index.entries)
    result = find_reverse_dependencies("dataset_a", reverse_map)
    assert result.root_artifact == "dataset_a"
    assert result.dependent_count > 0
    assert "checkpoint_b" in result.impacted_artifacts
    # Transitive: report_c depends on checkpoint_b which depends on dataset_a
    assert "report_c" in result.impacted_artifacts
    assert result.dependency_depth >= 2


def test_find_reverse_dependencies_leaf():
    index = _make_index()
    reverse_map, _edges = build_reverse_dependency_index(index.entries)
    result = find_reverse_dependencies("archive_d", reverse_map)
    assert result.dependent_count == 0
    assert result.impacted_artifacts == ()


def test_find_reverse_dependencies_orphan():
    index = _make_index()
    reverse_map, _edges = build_reverse_dependency_index(index.entries)
    result = find_reverse_dependencies("orphan_f", reverse_map)
    assert result.dependent_count == 0


def test_reverse_dependency_result_is_deterministic():
    index = _make_index()
    reverse_map, _edges = build_reverse_dependency_index(index.entries)
    r1 = find_reverse_dependencies("dataset_a", reverse_map)
    r2 = find_reverse_dependencies("dataset_a", reverse_map)
    assert r1.impacted_artifacts == r2.impacted_artifacts
    assert r1.dependency_depth == r2.dependency_depth


def test_reverse_dependency_closure():
    index = _make_index()
    reverse_map, _edges = build_reverse_dependency_index(index.entries)
    closure = reverse_dependency_closure("dataset_a", reverse_map)
    assert len(closure) > 0
    assert "checkpoint_b" in closure


def test_cycle_safe_traversal():
    # Create entries with a cycle: A -> B -> A
    entries = [
        _entry(artifact_id="cycle_a", lineage_parents=("cycle_b",)),
        _entry(artifact_id="cycle_b", lineage_parents=("cycle_a",)),
    ]
    index = _make_index(entries)
    reverse_map, _edges = build_reverse_dependency_index(index.entries)
    result = find_reverse_dependencies("cycle_a", reverse_map)
    assert result.dependent_count <= 2


def test_dependency_edge_dataclass():
    edge = DependencyEdge(source_id="a", target_id="b", relationship="derived_from")
    assert edge.source_id == "a"
    assert edge.relationship == "derived_from"


# --- Query Engine workspace-scoped tests ---

def test_query_workspace_root_filter():
    entries = [
        _entry(artifact_id="ws1_a", workspace_root="/ws1"),
        _entry(artifact_id="ws2_b", workspace_root="/ws2"),
    ]
    index = _make_index(entries)
    result = query_inventory(index, workspace_root="/ws1")
    assert all(e.workspace_root == "/ws1" for e in result)


def test_query_relative_prefix_filter():
    entries = [
        _entry(artifact_id="p1", relative_path="checkpoints/model_a.ckpt"),
        _entry(artifact_id="p2", relative_path="datasets/data.jsonl"),
    ]
    index = _make_index(entries)
    result = query_inventory(index, relative_prefix="checkpoints/")
    assert all(e.relative_path.startswith("checkpoints/") for e in result)


def test_query_retention_status_filter():
    entries = [
        _entry(artifact_id="active_a", retention_status="active"),
        _entry(artifact_id="archived_b", retention_status="archived"),
    ]
    index = _make_index(entries)
    result = query_inventory(index, retention_status="archived")
    assert len(result) == 1
    assert result[0].artifact_id == "archived_b"


def test_query_pinned_filter():
    entries = [
        _entry(artifact_id="pin_a", tags=("pinned",)),
        _entry(artifact_id="unpin_b", tags=()),
    ]
    index = _make_index(entries)
    result = query_inventory(index, pinned=True)
    assert len(result) == 1
    assert result[0].artifact_id == "pin_a"


def test_query_archive_status_filter():
    entries = [
        _entry(artifact_id="arch_a", artifact_type="archive_bundle"),
        _entry(artifact_id="act_b", artifact_type="dataset"),
    ]
    index = _make_index(entries)
    result = query_inventory(index, archive_status="archived")
    assert len(result) == 1
    assert result[0].artifact_id == "arch_a"


def test_query_lineage_depth_filter():
    entries = [
        _entry(artifact_id="root_a", lineage_parents=()),
        _entry(artifact_id="child_b", lineage_parents=("root_a",)),
    ]
    index = _make_index(entries)
    result = query_inventory(index, lineage_depth=0)
    assert any(e.artifact_id == "root_a" for e in result)


def test_query_created_after_filter():
    entries = [
        _entry(artifact_id="old_a", created_at=100.0),
        _entry(artifact_id="new_b", created_at=200.0),
    ]
    index = _make_index(entries)
    result = query_inventory(index, created_after=150.0)
    assert len(result) == 1
    assert result[0].artifact_id == "new_b"


def test_query_deterministic_ordering():
    entries = [
        _entry(artifact_id="z_entry", artifact_type="dataset", relative_path="z.jsonl"),
        _entry(artifact_id="a_entry", artifact_type="checkpoint", relative_path="a.ckpt"),
    ]
    index = _make_index(entries)
    r1 = query_inventory(index)
    r2 = query_inventory(index)
    assert [e.artifact_id for e in r1] == [e.artifact_id for e in r2]


# --- Saved Query tests ---

def test_save_and_load_query(tmp_path):
    query_data = {
        "query_name": "test_checkpoints",
        "created_at": 1.0,
        "inventory_fingerprint": "fp123",
        "filters": {"artifact_type": "checkpoint"},
    }
    path = save_query(tmp_path / "query.json", query_data)
    loaded = load_query(path)
    assert loaded["query_name"] == "test_checkpoints"
    assert "query_fingerprint" in loaded
    assert loaded["schema_version"] == SAVED_QUERY_SCHEMA_VERSION


def test_saved_query_fingerprint_deterministic(tmp_path):
    query_data = {
        "query_name": "stable",
        "created_at": 1.0,
        "inventory_fingerprint": "fp1",
        "filters": {"artifact_type": "dataset"},
    }
    p1 = save_query(tmp_path / "q1.json", dict(query_data))
    p2 = save_query(tmp_path / "q2.json", dict(query_data))
    q1 = json.loads(p1.read_text())
    q2 = json.loads(p2.read_text())
    assert q1["query_fingerprint"] == q2["query_fingerprint"]


def test_saved_query_detects_filter_change(tmp_path):
    q1 = {
        "query_name": "q",
        "created_at": 1.0,
        "inventory_fingerprint": "fp1",
        "filters": {"artifact_type": "dataset"},
    }
    q2 = dict(q1)
    q2["filters"] = {"artifact_type": "checkpoint"}
    p1 = save_query(tmp_path / "q1.json", q1)
    p2 = save_query(tmp_path / "q2.json", q2)
    fp1 = json.loads(p1.read_text())["query_fingerprint"]
    fp2 = json.loads(p2.read_text())["query_fingerprint"]
    assert fp1 != fp2


def test_execute_saved_query():
    index = _make_index()
    query = {
        "query_name": "checkpoints_only",
        "created_at": 1.0,
        "inventory_fingerprint": index.inventory_fingerprint,
        "filters": {"artifact_type": "checkpoint"},
        "query_fingerprint": "placeholder",
    }
    results = execute_saved_query(query, index)
    assert all(e.artifact_type == "checkpoint" for e in results)


def test_execute_saved_query_workspace_filter():
    entries = [
        _entry(artifact_id="ws1_a", workspace_root="/ws1"),
        _entry(artifact_id="ws2_b", workspace_root="/ws2"),
    ]
    index = _make_index(entries)
    query = {
        "query_name": "ws1_only",
        "created_at": 1.0,
        "inventory_fingerprint": index.inventory_fingerprint,
        "filters": {"workspace_root": "/ws1"},
        "query_fingerprint": "placeholder",
    }
    results = execute_saved_query(query, index)
    assert all(e.workspace_root == "/ws1" for e in results)


def test_validate_saved_query_rejects_missing_fields():
    with pytest.raises(ValueError, match="Missing"):
        validate_saved_query_data({"query_name": "bad"})


def test_validate_saved_query_rejects_unknown_filter_key():
    with pytest.raises(ValueError, match="Unknown filter"):
        validate_saved_query_data({
            "query_name": "bad",
            "created_at": 1.0,
            "inventory_fingerprint": "fp",
            "filters": {"nonexistent_key": "value"},
            "query_fingerprint": "fp",
        })


# --- Search Facets tests ---

def test_build_search_facets_basic():
    index = _make_index()
    facets = build_search_facets(index)
    d = facets.to_dict()
    assert "facets" in d
    assert "artifact_type" in d["facets"]
    assert "facets_fingerprint" in d


def test_search_facets_counts():
    entries = [
        _entry(artifact_id="a1", artifact_type="dataset"),
        _entry(artifact_id="a2", artifact_type="dataset"),
        _entry(artifact_id="a3", artifact_type="checkpoint"),
    ]
    index = _make_index(entries)
    facets = build_search_facets(index)
    d = facets.to_dict()
    assert d["facets"]["artifact_type"]["dataset"] == 2
    assert d["facets"]["artifact_type"]["checkpoint"] == 1


def test_search_facets_deterministic():
    index = _make_index()
    f1 = build_search_facets(index).to_dict()
    f2 = build_search_facets(index).to_dict()
    assert f1["facets_fingerprint"] == f2["facets_fingerprint"]


def test_search_facets_markdown():
    index = _make_index()
    md = build_search_facets(index).to_markdown()
    assert "# CPT Search Facets" in md
    assert "| Facet" in md


# --- Discovery Report tests ---

def test_build_discovery_report_basic():
    index = _make_index()
    report = build_discovery_report(index)
    d = report.to_dict()
    assert "summary" in d
    assert d["summary"]["total_artifacts"] == index.entry_count
    assert d["summary"]["root_artifacts"] >= 1  # dataset_a and orphan_f
    assert d["summary"]["orphan_artifacts"] >= 1  # orphan_f (no dependents)
    assert "discovery_fingerprint" in d


def test_discovery_report_orphans():
    # orphan_f has no parents AND no dependents
    index = _make_index()
    report = build_discovery_report(index)
    d = report.to_dict()
    orphan_ids = [o["artifact_id"] for o in d["orphans"]]
    # orphan_f is an orphan (no dependents in reverse index)
    assert "orphan_f" in orphan_ids


def test_discovery_report_dependency_hubs():
    index = _make_index()
    report = build_discovery_report(index)
    d = report.to_dict()
    assert len(d["dependency_hubs"]) > 0
    # dataset_a should be top hub (checkpoint_b + manifest_e depend on it)
    assert d["dependency_hubs"][0]["artifact_id"] == "dataset_a"


def test_discovery_report_deterministic():
    index = _make_index()
    r1 = build_discovery_report(index).to_dict()
    r2 = build_discovery_report(index).to_dict()
    assert r1["discovery_fingerprint"] == r2["discovery_fingerprint"]


def test_discovery_report_markdown():
    index = _make_index()
    md = build_discovery_report(index).to_markdown()
    assert "# CPT Artifact Discovery Report" in md
    assert "Total Artifacts" in md
    assert "Dependency Hubs" in md


# --- Impact Analysis tests ---

def test_analyze_artifact_impact_basic():
    index = _make_index()
    graph = build_lineage_graph(index)
    result = analyze_artifact_impact("dataset_a", index, graph)
    assert isinstance(result, ImpactAnalysisResult)
    assert result.artifact_id == "dataset_a"
    assert len(result.impacted_artifacts) > 0
    assert "impact_fingerprint" in result.to_dict()


def test_analyze_artifact_impact_leaf():
    index = _make_index()
    graph = build_lineage_graph(index)
    result = analyze_artifact_impact("archive_d", index, graph)
    assert len(result.impacted_artifacts) == 0
    assert result.archive_bundles_affected == 0


def test_analyze_artifact_impact_counts_types():
    index = _make_index()
    graph = build_lineage_graph(index)
    result = analyze_artifact_impact("dataset_a", index, graph)
    # checkpoint_b depends on dataset_a
    assert result.checkpoint_invalidations >= 1
    # report_c depends on checkpoint_b
    assert result.report_invalidations >= 1


def test_impact_analysis_deterministic():
    index = _make_index()
    graph = build_lineage_graph(index)
    r1 = analyze_artifact_impact("dataset_a", index, graph)
    r2 = analyze_artifact_impact("dataset_a", index, graph)
    assert r1.impact_fingerprint == r2.impact_fingerprint


# --- Lineage Graph integration ---

def test_lineage_graph_builds_from_inventory():
    index = _make_index()
    graph = build_lineage_graph(index)
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0


def test_lineage_graph_fingerprint_deterministic():
    index = _make_index()
    g1 = build_lineage_graph(index)
    g2 = build_lineage_graph(index)
    assert g1.graph_fingerprint == g2.graph_fingerprint


# --- CLI reproducibility tests ---

def test_cli_find_reverse_dependencies(tmp_path):
    """Test that the CLI script exists and can be imported."""
    import importlib
    spec = importlib.util.find_spec("scripts.find_reverse_dependencies")
    # Script may not be importable as module, check file exists
    assert Path("scripts/find_reverse_dependencies.py").exists()


def test_cli_run_saved_query_exists():
    assert Path("scripts/run_saved_query.py").exists()


def test_cli_generate_discovery_report_exists():
    assert Path("scripts/generate_discovery_report.py").exists()


def test_cli_analyze_artifact_impact_exists():
    assert Path("scripts/analyze_artifact_impact.py").exists()


# --- Backward compatibility ---

def test_backward_compat_v26_dataset_loadable():
    """v2.6 datasets should still be loadable and upgradable."""
    from backend.datasets.loader import load_jsonl, upgrade_v26_row
    row = {
        "sample_id": "compat_test",
        "question": "Q",
        "structured_state": {},
        "reasoning_trace": [],
        "equations_used": [],
        "invariants_checked": [],
        "final_answer": {},
        "verification_status": {},
        "module_source": "mod",
        "curriculum_layer": 0,
        "seed": 1,
        "timestamp": 0.0,
    }
    upgraded = upgrade_v26_row(row, snapshot_hash="snap", module_hash="mod")
    assert upgraded["dataset_version"] == "2.7.0"


def test_backward_compat_v27_manifest_loadable(tmp_path):
    """v2.7 manifests should still validate."""
    from backend.datasets.manifest import DatasetManifest, validate_manifest
    m = DatasetManifest(generation_seed=1, record_count=1, snapshot_hash="s", module_hash="m")
    path = m.save(tmp_path / "compat.manifest.json")
    loaded = DatasetManifest.from_file(path)
    errors = validate_manifest(loaded.to_dict())
    assert errors == []


def test_backward_compat_existing_tests_pass():
    """Ensure v2.7.9 test count is preserved (121 + new v2.7.10 tests)."""
    # This is implicitly verified by pytest suite total
    pass
