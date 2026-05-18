"""Tests for CPT v2.14 — Retrieval Memory, Semantic Warm-Start & Cost Estimation.

Covers:
1. RetrievalEntry: construction, validation, frozen, JSON roundtrip, fingerprint
2. RetrievalMemory: add, search, duplicates, compact, stats, atomic persistence
3. EmbeddingResult: from_tensor, normalization, SHA-256 determinism
4. extract_graph_embedding: GNN encoder output
5. compute_embedding_sha256: determinism
6. FaissRuntime: add, search, NaN rejection, degraded rejection, persistence
7. WarmstartRuntime: evaluate_warmstart, acceptance/rejection, fallback
8. CostEstimator: estimation, difficulty classification, determinism
9. CapabilityRouter v2.14: new actions, retrieval-aware routing
10. ProjectionExperienceMemory: add, family_stats, persistence
11. E2E integration: full pipeline with retrieval + warmstart
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from backend.runtime.retrieval_memory import RetrievalEntry, RetrievalMemory
from backend.runtime.embedding_runtime import (
    EmbeddingResult,
    extract_graph_embedding,
    normalize_embedding,
    compute_embedding_sha256,
)
from backend.runtime.cost_estimator import ExecutionCostEstimate, CostEstimator
from backend.runtime.warmstart_runtime import WarmStartResult, WarmstartRuntime
from backend.runtime.projection_experience import (
    ProjectionExperienceEntry,
    ProjectionExperienceMemory,
)
from backend.core_runtime.capability_router import CapabilityRouter, RoutingDecision
from backend.core_runtime.confidence_runtime import ConfidenceEstimate
from backend.core_runtime.execution_policy import ExecutionPolicy
from backend.core_runtime.task_runtime import RuntimeTask

try:
    from backend.runtime.faiss_runtime import FaissRuntime, TopKSimilarityResult
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


def _make_entry(task_hash="hash_a", topo="series", **overrides) -> RetrievalEntry:
    defaults = dict(
        task_hash=task_hash,
        embedding_sha256="emb_" + task_hash[:8],
        topology_family=topo,
        node_count=5,
        edge_count=4,
        confidence=0.85,
        projection_iterations=8,
        kcl_residual=0.001,
        kvl_residual=0.002,
        timestamp="2025-01-01T00:00:00Z",
        embedding_path="",
        trace_path="",
    )
    defaults.update(overrides)
    return RetrievalEntry(**defaults)


def _make_task(task_id="t1", topo="series") -> RuntimeTask:
    return RuntimeTask(
        task_id=task_id,
        domain="circuit",
        input_artifact=task_id,
        oracle_name="mna_oracle",
        surrogate_name="circuit_gnn",
        projection_enabled=True,
        metadata={"topology_family": topo},
    )


def _make_confidence(score=0.8, ood=False) -> ConfidenceEstimate:
    return ConfidenceEstimate(
        confidence_score=score,
        estimated_projection_iterations=5,
        likely_ood=ood,
    )


# ===================================================================
# 1. RetrievalEntry
# ===================================================================

class TestRetrievalEntry:

    def test_construction(self):
        e = _make_entry()
        assert e.task_hash == "hash_a"
        assert e.topology_family == "series"
        assert e.node_count == 5
        assert e.confidence == 0.85

    def test_frozen(self):
        e = _make_entry()
        with pytest.raises(AttributeError):
            e.task_hash = "changed"  # type: ignore

    def test_fingerprint_deterministic(self):
        e1 = _make_entry()
        e2 = _make_entry()
        assert e1.fingerprint == e2.fingerprint

    def test_different_entries_different_fingerprints(self):
        e1 = _make_entry(task_hash="hash_a")
        e2 = _make_entry(task_hash="hash_b")
        assert e1.fingerprint != e2.fingerprint

    def test_json_roundtrip(self):
        e = _make_entry()
        d = e.to_json_dict()
        e2 = RetrievalEntry.from_json_dict(d)
        assert e2.task_hash == e.task_hash
        assert e2.embedding_sha256 == e.embedding_sha256
        assert e2.confidence == e.confidence

    def test_validation_empty_task_hash(self):
        e = _make_entry(task_hash="")
        assert "task_hash must not be empty" in e.validate()

    def test_validation_negative_node_count(self):
        e = _make_entry(node_count=-1)
        assert "node_count must be non-negative" in e.validate()

    def test_validation_confidence_range(self):
        e = _make_entry(confidence=1.5)
        assert "confidence must be in [0, 1]" in e.validate()

    def test_validation_valid_entry(self):
        e = _make_entry()
        assert e.validate() == []


# ===================================================================
# 2. RetrievalMemory
# ===================================================================

class TestRetrievalMemory:

    def test_add_and_search(self, tmp_dir):
        mem = RetrievalMemory(str(Path(tmp_dir) / "retrieval"))
        e = _make_entry()
        assert mem.add(e) is True
        found = mem.search("hash_a")
        assert found is not None
        assert found.task_hash == "hash_a"

    def test_no_duplicate_insertion(self, tmp_dir):
        mem = RetrievalMemory(str(Path(tmp_dir) / "retrieval"))
        e = _make_entry()
        assert mem.add(e) is True
        assert mem.add(e) is False  # Duplicate rejected

    def test_search_missing(self, tmp_dir):
        mem = RetrievalMemory(str(Path(tmp_dir) / "retrieval"))
        assert mem.search("nonexistent") is None

    def test_search_by_topology(self, tmp_dir):
        mem = RetrievalMemory(str(Path(tmp_dir) / "retrieval"))
        mem.add(_make_entry(task_hash="h1", topo="series"))
        mem.add(_make_entry(task_hash="h2", topo="parallel"))
        mem.add(_make_entry(task_hash="h3", topo="series"))
        results = mem.search_by_topology("series")
        assert len(results) == 2
        # Deterministic ordering
        assert results[0].task_hash <= results[1].task_hash

    def test_count_and_stats(self, tmp_dir):
        mem = RetrievalMemory(str(Path(tmp_dir) / "retrieval"))
        mem.add(_make_entry(task_hash="h1", topo="series", confidence=0.9, projection_iterations=5))
        mem.add(_make_entry(task_hash="h2", topo="parallel", confidence=0.7, projection_iterations=10))
        stats = mem.stats()
        assert stats["total_entries"] == 2
        assert stats["topology_families"] == 2

    def test_deterministic_ordering(self, tmp_dir):
        mem = RetrievalMemory(str(Path(tmp_dir) / "retrieval"))
        for h in ["h3", "h1", "h2"]:
            mem.add(_make_entry(task_hash=h))
        entries = mem.all_entries()
        hashes = [e.task_hash for e in entries]
        assert hashes == sorted(hashes)

    def test_atomic_persistence(self, tmp_dir):
        path = str(Path(tmp_dir) / "retrieval")
        mem1 = RetrievalMemory(path)
        mem1.add(_make_entry(task_hash="persist_test"))
        # Reload from disk
        mem2 = RetrievalMemory(path)
        found = mem2.search("persist_test")
        assert found is not None

    def test_compact_removes_orphaned(self, tmp_dir):
        mem = RetrievalMemory(str(Path(tmp_dir) / "retrieval"))
        # Entry with non-existent embedding path
        mem.add(_make_entry(task_hash="h_orphan", embedding_path="/nonexistent/path.pt"))
        removed = mem.compact()
        assert removed == 1
        assert mem.search("h_orphan") is None

    def test_remove_entry(self, tmp_dir):
        mem = RetrievalMemory(str(Path(tmp_dir) / "retrieval"))
        mem.add(_make_entry(task_hash="h_rm"))
        assert mem.remove("h_rm") is True
        assert mem.remove("h_rm") is False
        assert mem.search("h_rm") is None

    def test_invalid_entry_rejected(self, tmp_dir):
        mem = RetrievalMemory(str(Path(tmp_dir) / "retrieval"))
        bad = _make_entry(task_hash="")
        with pytest.raises(ValueError):
            mem.add(bad)


# ===================================================================
# 3. EmbeddingResult
# ===================================================================

class TestEmbeddingResult:

    def test_from_tensor(self):
        t = torch.randn(64)
        result = EmbeddingResult.from_tensor(t, topology_family="series")
        assert len(result.vector) == 64
        assert result.sha256 != ""
        assert result.topology_family == "series"

    def test_deterministic_hashing(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        r1 = EmbeddingResult.from_tensor(t)
        r2 = EmbeddingResult.from_tensor(t)
        assert r1.sha256 == r2.sha256

    def test_different_tensors_different_hashes(self):
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0, 2.0, 3.1])
        r1 = EmbeddingResult.from_tensor(t1)
        r2 = EmbeddingResult.from_tensor(t2)
        assert r1.sha256 != r2.sha256

    def test_frozen(self):
        t = torch.randn(16)
        r = EmbeddingResult.from_tensor(t)
        with pytest.raises(AttributeError):
            r.sha256 = "changed"  # type: ignore

    def test_normalize_embedding_deterministic(self):
        t = torch.randn(32)
        n1 = normalize_embedding(t)
        n2 = normalize_embedding(t)
        assert torch.equal(n1, n2)

    def test_compute_embedding_sha256(self):
        t = torch.tensor([0.5, -0.3, 1.2, 0.0])
        h1 = compute_embedding_sha256(t)
        h2 = compute_embedding_sha256(t)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256

    def test_json_dict(self):
        t = torch.randn(8)
        r = EmbeddingResult.from_tensor(t, metadata={"key": "val"})
        d = r.to_json_dict()
        assert "vector_sha256" in d
        assert d["dim"] == 8


# ===================================================================
# 4. GNN Embedding Extraction
# ===================================================================

class TestGNNEmbeddingExtraction:

    def test_circuit_gnn_embedding(self):
        from backend.neural.models.circuit_gnn import CircuitGNN
        model = CircuitGNN(node_dim=8, hidden_dim=32)
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        emb = extract_graph_embedding(model, x, edge_index)
        assert emb.shape == (32,)  # hidden_dim
        assert not torch.isnan(emb).any()

    def test_deterministic_extraction(self):
        from backend.neural.models.circuit_gnn import CircuitGNN
        torch.manual_seed(42)
        model = CircuitGNN(node_dim=8, hidden_dim=32)
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        e1 = extract_graph_embedding(model, x, edge_index)
        e2 = extract_graph_embedding(model, x, edge_index)
        assert torch.equal(e1, e2)

    def test_edge_aware_gnn_embedding(self):
        from backend.neural.models.circuit_gnn import EdgeAwareCircuitGNN
        model = EdgeAwareCircuitGNN(node_dim=8, edge_dim=4, hidden_dim=32)
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_feat = torch.randn(4, 4)
        emb = extract_graph_embedding(model, x, edge_index, edge_feat)
        assert emb.shape == (32,)
        assert not torch.isnan(emb).any()


# ===================================================================
# 5. FaissRuntime (conditional on FAISS)
# ===================================================================

@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")
class TestFaissRuntime:

    def test_add_and_search(self, tmp_dir):
        fr = FaissRuntime(dim=8, base_dir=str(Path(tmp_dir) / "faiss"))
        emb = np.random.randn(8).astype(np.float32)
        entry = _make_entry(task_hash="faiss_1")
        assert fr.add_embedding("faiss_1", emb, entry) is True
        # Search with same embedding
        results = fr.search(emb, k=1)
        assert len(results) == 1
        assert results[0].task_hash == "faiss_1"
        assert results[0].similarity_score > 0.9

    def test_nan_rejection(self, tmp_dir):
        fr = FaissRuntime(dim=8, base_dir=str(Path(tmp_dir) / "faiss"))
        emb = np.array([1.0, float("nan"), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        entry = _make_entry(task_hash="nan_1")
        assert fr.add_embedding("nan_1", emb, entry) is False

    def test_duplicate_rejection(self, tmp_dir):
        fr = FaissRuntime(dim=8, base_dir=str(Path(tmp_dir) / "faiss"))
        emb = np.random.randn(8).astype(np.float32)
        entry = _make_entry(task_hash="dup_1")
        assert fr.add_embedding("dup_1", emb, entry) is True
        assert fr.add_embedding("dup_1", emb, entry) is False

    def test_deterministic_search_order(self, tmp_dir):
        fr = FaissRuntime(dim=8, base_dir=str(Path(tmp_dir) / "faiss"))
        query = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        # Add entries with known similarity
        for i in range(5):
            emb = np.zeros(8, dtype=np.float32)
            emb[0] = 1.0 - i * 0.1  # Decreasing similarity
            entry = _make_entry(task_hash=f"det_{i}")
            fr.add_embedding(f"det_{i}", emb, entry)
        results = fr.search(query, k=5)
        # Results should be ordered by similarity (descending)
        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_min_similarity_filter(self, tmp_dir):
        fr = FaissRuntime(dim=8, base_dir=str(Path(tmp_dir) / "faiss"))
        emb = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        entry = _make_entry(task_hash="filter_1")
        fr.add_embedding("filter_1", emb, entry)
        # Orthogonal query → low similarity
        query = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        results = fr.search(query, k=5, min_similarity=0.99)
        assert len(results) == 0

    def test_atomic_persistence(self, tmp_dir):
        path = str(Path(tmp_dir) / "faiss")
        fr1 = FaissRuntime(dim=8, base_dir=path)
        emb = np.random.randn(8).astype(np.float32)
        entry = _make_entry(task_hash="persist_1")
        fr1.add_embedding("persist_1", emb, entry)
        # Reload
        fr2 = FaissRuntime(dim=8, base_dir=path)
        assert fr2.ntotal == 1
        results = fr2.search(emb, k=1)
        assert len(results) == 1
        assert results[0].task_hash == "persist_1"

    def test_contains(self, tmp_dir):
        fr = FaissRuntime(dim=8, base_dir=str(Path(tmp_dir) / "faiss"))
        emb = np.random.randn(8).astype(np.float32)
        entry = _make_entry(task_hash="cont_1")
        fr.add_embedding("cont_1", emb, entry)
        assert fr.contains("cont_1") is True
        assert fr.contains("nonexistent") is False


# ===================================================================
# 6. WarmstartRuntime
# ===================================================================

class TestWarmstartRuntime:

    def test_accepted_warmstart(self):
        ws = WarmstartRuntime()
        result = ws.evaluate_warmstart(
            initial_residual_warmstart=0.1,
            initial_residual_standard=0.5,
            projected_residual=0.001,
            iterations_warmstart=3,
            iterations_standard=8,
            similarity_score=0.9,
        )
        assert result.accepted is True
        assert result.iterations_saved == 5
        assert result.convergence_gain > 0
        assert result.rejection_reason is None

    def test_rejected_similar_low_similarity(self):
        ws = WarmstartRuntime(min_similarity=0.5)
        result = ws.evaluate_warmstart(
            initial_residual_warmstart=0.1,
            initial_residual_standard=0.5,
            projected_residual=0.001,
            iterations_warmstart=3,
            iterations_standard=8,
            similarity_score=0.2,  # Below threshold
        )
        assert result.accepted is False
        assert "Similarity" in result.rejection_reason

    def test_rejected_worse_initial_residual(self):
        ws = WarmstartRuntime()
        result = ws.evaluate_warmstart(
            initial_residual_warmstart=0.6,  # Worse than standard
            initial_residual_standard=0.5,
            projected_residual=0.001,
            iterations_warmstart=5,
            iterations_standard=8,
            similarity_score=0.9,
        )
        assert result.accepted is False
        assert "not lower" in result.rejection_reason

    def test_rejected_insufficient_gain(self):
        ws = WarmstartRuntime(min_convergence_gain=0.01)
        result = ws.evaluate_warmstart(
            initial_residual_warmstart=0.499,
            initial_residual_standard=0.5,
            projected_residual=0.001,
            iterations_warmstart=7,
            iterations_standard=8,
            similarity_score=0.9,
        )
        assert result.accepted is False
        assert "gain" in result.rejection_reason.lower()

    def test_stats_tracking(self):
        ws = WarmstartRuntime()
        ws.evaluate_warmstart(0.1, 0.5, 0.001, 3, 8, 0.9)  # accepted
        ws.evaluate_warmstart(0.6, 0.5, 0.001, 5, 8, 0.9)  # rejected
        stats = ws.stats
        assert stats["total_attempts"] == 2
        assert stats["accepted"] == 1
        assert stats["rejected"] == 1
        assert stats["acceptance_rate"] == 0.5

    def test_warmstart_result_serialization(self):
        ws = WarmstartRuntime()
        result = ws.evaluate_warmstart(0.1, 0.5, 0.001, 3, 8, 0.9)
        d = result.to_json_dict()
        assert d["accepted"] is True
        assert "initial_residual" in d


# ===================================================================
# 7. CostEstimator
# ===================================================================

class TestCostEstimator:

    def test_trivial_circuit(self):
        ce = CostEstimator()
        est = ce.estimate(node_count=2, edge_count=1, confidence=1.0)
        assert est.estimated_difficulty == "trivial"
        assert est.estimated_projection_iterations >= 1

    def test_hard_circuit(self):
        ce = CostEstimator()
        est = ce.estimate(
            node_count=100,
            edge_count=200,
            resistance_range=(0.001, 1000000),
            likely_ood=True,
            confidence=0.1,
        )
        assert est.estimated_difficulty in ("hard", "extreme")
        assert est.estimated_projection_iterations > 5

    def test_deterministic(self):
        ce = CostEstimator()
        e1 = ce.estimate(node_count=10, edge_count=15, confidence=0.8)
        e2 = ce.estimate(node_count=10, edge_count=15, confidence=0.8)
        assert e1.estimated_projection_iterations == e2.estimated_projection_iterations
        assert e1.estimated_difficulty == e2.estimated_difficulty

    def test_ood_increases_cost(self):
        ce = CostEstimator()
        e_std = ce.estimate(node_count=10, edge_count=15, likely_ood=False)
        e_ood = ce.estimate(node_count=10, edge_count=15, likely_ood=True)
        assert e_ood.estimated_projection_iterations >= e_std.estimated_projection_iterations

    def test_low_confidence_increases_iterations(self):
        ce = CostEstimator()
        e_high = ce.estimate(node_count=10, edge_count=15, confidence=0.9)
        e_low = ce.estimate(node_count=10, edge_count=15, confidence=0.2)
        assert e_low.estimated_projection_iterations >= e_high.estimated_projection_iterations

    def test_invalid_difficulty_rejected(self):
        with pytest.raises(ValueError):
            ExecutionCostEstimate(
                estimated_projection_iterations=5,
                estimated_runtime_ms=10,
                estimated_memory_cost=0.01,
                estimated_difficulty="invalid",
                estimated_confidence=0.5,
            )

    def test_valid_difficulties(self):
        for d in ["trivial", "easy", "moderate", "hard", "extreme"]:
            est = ExecutionCostEstimate(
                estimated_projection_iterations=5,
                estimated_runtime_ms=10,
                estimated_memory_cost=0.01,
                estimated_difficulty=d,
                estimated_confidence=0.5,
            )
            assert est.estimated_difficulty == d

    def test_json_dict(self):
        ce = CostEstimator()
        est = ce.estimate(node_count=10, edge_count=15)
        d = est.to_json_dict()
        assert "estimated_difficulty" in d
        assert "estimated_projection_iterations" in d

    def test_prior_stats_adjustment(self):
        ce = CostEstimator(prior_stats={"topology_avg_iterations": {"series": 15.0}})
        est = ce.estimate(node_count=5, edge_count=4, topology_family="series")
        # Should blend with historical average (15), pushing up from pure heuristic
        assert est.estimated_projection_iterations > 3  # More than trivial


# ===================================================================
# 8. CapabilityRouter v2.14
# ===================================================================

class TestCapabilityRouterV214:

    def test_cache_hit(self):
        router = CapabilityRouter()
        task = _make_task()
        conf = _make_confidence()
        dec = router.route(task, conf, cache_hit=True)
        assert dec.action == "exact_cache_hit"

    def test_degraded_execution(self):
        router = CapabilityRouter()
        task = _make_task()
        conf = _make_confidence()
        dec = router.route(task, conf, is_degraded=True)
        assert dec.action == "degraded_execution"
        assert dec.force_oracle is True

    def test_warmstart_projection(self):
        router = CapabilityRouter()
        task = _make_task()
        conf = _make_confidence(score=0.7)
        dec = router.route(task, conf, retrieval_similarity=0.8)
        assert dec.action == "warmstart_projection"
        assert dec.retrieval_similarity == 0.8

    def test_semantic_retrieval(self):
        router = CapabilityRouter()
        task = _make_task()
        conf = _make_confidence(score=0.7)
        dec = router.route(task, conf, retrieval_similarity=0.35)
        assert dec.action == "semantic_retrieval"

    def test_standard_projection(self):
        router = CapabilityRouter()
        task = _make_task()
        conf = _make_confidence(score=0.8)
        dec = router.route(task, conf, cache_hit=False, retrieval_similarity=0.0)
        assert dec.action == "standard_projection"

    def test_ood_with_warmstart(self):
        router = CapabilityRouter()
        task = _make_task()
        conf = _make_confidence(score=0.3, ood=True)
        dec = router.route(task, conf, retrieval_similarity=0.7)
        # OOD + warmstart → warmstart_projection with high budget
        assert dec.action == "warmstart_projection"
        assert dec.force_oracle is True

    def test_ood_without_warmstart(self):
        router = CapabilityRouter()
        task = _make_task()
        conf = _make_confidence(score=0.3, ood=True)
        dec = router.route(task, conf, retrieval_similarity=0.0)
        assert dec.action == "increased_budget"

    def test_oracle_verification(self):
        router = CapabilityRouter()
        task = _make_task(topo="problematic")
        # Record 3 failures
        for _ in range(3):
            router.record_failure("problematic")
        conf = _make_confidence(score=0.8)
        dec = router.route(task, conf)
        assert dec.action == "oracle_verification"

    def test_cost_estimate_in_decision(self):
        router = CapabilityRouter()
        task = _make_task()
        conf = _make_confidence(score=0.8)
        ce = CostEstimator()
        est = ce.estimate(node_count=5, edge_count=4)
        dec = router.route(task, conf, cost_estimate=est)
        assert dec.estimated_cost is not None

    def test_invalid_action_rejected(self):
        with pytest.raises(ValueError):
            RoutingDecision(
                action="invalid_action",
                projection_budget=5,
                force_oracle=False,
                reason="test",
            )

    def test_cache_hit_priority_over_retrieval(self):
        """Exact cache ALWAYS wins, even with high similarity."""
        router = CapabilityRouter()
        task = _make_task()
        conf = _make_confidence()
        dec = router.route(task, conf, cache_hit=True, retrieval_similarity=0.99)
        assert dec.action == "exact_cache_hit"


# ===================================================================
# 9. ProjectionExperienceMemory
# ===================================================================

class TestProjectionExperienceMemory:

    def test_add_and_count(self, tmp_dir):
        pem = ProjectionExperienceMemory(str(Path(tmp_dir) / "proj_exp"))
        entry = ProjectionExperienceEntry(
            task_hash="p1",
            topology_family="series",
            initial_residual=0.5,
            final_residual=0.001,
            residual_slope=0.05,
            iterations=10,
            converged=True,
            kcl_residual=0.001,
            kvl_residual=0.002,
            used_warmstart=False,
            warmstart_similarity=0.0,
            timestamp="2025-01-01T00:00:00Z",
        )
        pem.add(entry)
        assert pem.count == 1

    def test_family_stats(self, tmp_dir):
        pem = ProjectionExperienceMemory(str(Path(tmp_dir) / "proj_exp"))
        for i in range(3):
            pem.add(ProjectionExperienceEntry(
                task_hash=f"p{i}",
                topology_family="series",
                initial_residual=0.5,
                final_residual=0.001,
                residual_slope=0.05,
                iterations=10,
                converged=True,
                kcl_residual=0.001,
                kvl_residual=0.002,
                used_warmstart=i == 0,
                warmstart_similarity=0.8 if i == 0 else 0.0,
                timestamp="2025-01-01T00:00:00Z",
            ))
        stats = pem.family_stats("series")
        assert stats["count"] == 3
        assert stats["convergence_rate"] == 1.0
        assert abs(stats["warmstart_usage_rate"] - 1/3) < 0.01

    def test_persistence(self, tmp_dir):
        path = str(Path(tmp_dir) / "proj_exp")
        pem1 = ProjectionExperienceMemory(path)
        pem1.add(ProjectionExperienceEntry(
            task_hash="persist_1",
            topology_family="series",
            initial_residual=0.5,
            final_residual=0.001,
            residual_slope=0.05,
            iterations=10,
            converged=True,
            kcl_residual=0.001,
            kvl_residual=0.002,
            used_warmstart=False,
            warmstart_similarity=0.0,
            timestamp="2025-01-01T00:00:00Z",
        ))
        pem2 = ProjectionExperienceMemory(path)
        assert pem2.count == 1

    def test_entry_fingerprint(self):
        entry = ProjectionExperienceEntry(
            task_hash="fp_test",
            topology_family="series",
            initial_residual=0.5,
            final_residual=0.001,
            residual_slope=0.05,
            iterations=10,
            converged=True,
            kcl_residual=0.001,
            kvl_residual=0.002,
            used_warmstart=False,
            warmstart_similarity=0.0,
            timestamp="2025-01-01T00:00:00Z",
        )
        assert len(entry.fingerprint) == 64

    def test_recent_entries(self, tmp_dir):
        pem = ProjectionExperienceMemory(str(Path(tmp_dir) / "proj_exp"))
        for i in range(20):
            pem.add(ProjectionExperienceEntry(
                task_hash=f"recent_{i}",
                topology_family="series",
                initial_residual=0.5,
                final_residual=0.001,
                residual_slope=0.05,
                iterations=10,
                converged=True,
                kcl_residual=0.001,
                kvl_residual=0.002,
                used_warmstart=False,
                warmstart_similarity=0.0,
                timestamp="2025-01-01T00:00:00Z",
            ))
        recent = pem.recent_entries(5)
        assert len(recent) == 5


# ===================================================================
# 10. E2E Integration
# ===================================================================

class TestE2EIntegration:

    def test_full_retrieval_pipeline(self, tmp_dir):
        """End-to-end: add to retrieval → search → warmstart → projection experience."""
        base = str(Path(tmp_dir) / "e2e")
        mem = RetrievalMemory(str(Path(base) / "retrieval"))
        pem = ProjectionExperienceMemory(str(Path(base) / "proj_exp"))
        ws = WarmstartRuntime()

        # Step 1: Add retrieval entry
        entry = _make_entry(task_hash="e2e_1", topo="series", confidence=0.9, projection_iterations=10)
        mem.add(entry)

        # Step 2: Search
        found = mem.search("e2e_1")
        assert found is not None

        # Step 3: Evaluate warmstart
        result = ws.evaluate_warmstart(
            initial_residual_warmstart=0.1,
            initial_residual_standard=0.5,
            projected_residual=0.001,
            iterations_warmstart=4,
            iterations_standard=10,
            similarity_score=0.9,
        )
        assert result.accepted is True

        # Step 4: Record projection experience
        pem.add(ProjectionExperienceEntry(
            task_hash="e2e_1",
            topology_family="series",
            initial_residual=0.1,
            final_residual=0.001,
            residual_slope=(0.1 - 0.001) / 4,
            iterations=4,
            converged=True,
            kcl_residual=0.001,
            kvl_residual=0.002,
            used_warmstart=True,
            warmstart_similarity=0.9,
            timestamp="2025-01-01T00:00:00Z",
        ))
        assert pem.count == 1

    @pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")
    def test_faiss_retrieval_warmstart_pipeline(self, tmp_dir):
        """FAISS search → warmstart evaluation → projection experience."""
        base = str(Path(tmp_dir) / "faiss_e2e")
        fr = FaissRuntime(dim=16, base_dir=str(Path(base) / "faiss"))
        mem = RetrievalMemory(str(Path(base) / "retrieval"))
        ws = WarmstartRuntime()
        ce = CostEstimator()

        # Add an embedding
        emb = np.random.randn(16).astype(np.float32)
        entry = _make_entry(task_hash="faiss_e2e_1", topo="series", confidence=0.9)
        mem.add(entry)
        fr.add_embedding("faiss_e2e_1", emb, entry)

        # Search with same embedding
        results = fr.search(emb, k=1)
        assert len(results) == 1
        assert results[0].task_hash == "faiss_e2e_1"

        # Evaluate warmstart
        ws_result = ws.evaluate_warmstart(
            initial_residual_warmstart=0.05,
            initial_residual_standard=0.4,
            projected_residual=0.001,
            iterations_warmstart=2,
            iterations_standard=8,
            similarity_score=results[0].similarity_score,
        )
        assert ws_result.accepted is True

        # Cost estimation
        cost = ce.estimate(node_count=5, edge_count=4, confidence=0.9)
        assert cost.estimated_difficulty in ("trivial", "easy")

    def test_deterministic_embedding_hashing_e2e(self):
        """Same graph → same embedding → same SHA-256 across runs."""
        from backend.neural.models.circuit_gnn import CircuitGNN
        torch.manual_seed(42)
        model = CircuitGNN(node_dim=8, hidden_dim=32)
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        emb1 = extract_graph_embedding(model, x, edge_index)
        emb2 = extract_graph_embedding(model, x, edge_index)

        h1 = compute_embedding_sha256(emb1)
        h2 = compute_embedding_sha256(emb2)
        assert h1 == h2

    def test_degraded_never_inserted_in_retrieval(self, tmp_dir):
        """Degraded executions must NOT be inserted into retrieval index."""
        mem = RetrievalMemory(str(Path(tmp_dir) / "retrieval"))
        # Valid entry
        good = _make_entry(task_hash="good_1", confidence=0.9, kcl_residual=0.001)
        mem.add(good)
        assert mem.search("good_1") is not None

        # The spec says: DO NOT cache degraded executions as valid hits
        # In practice, the benchmark/executor code checks failure_type
        # before inserting into retrieval. This test verifies that
        # RetrievalMemory itself doesn't accept invalid entries.
        bad = _make_entry(task_hash="bad_1", confidence=1.5)  # Invalid
        with pytest.raises(ValueError):
            mem.add(bad)

    def test_projection_integrity_preserved(self):
        """Warmstart must NOT bypass projection logic."""
        ws = WarmstartRuntime()
        # If warmstart makes things worse, it's rejected
        result = ws.evaluate_warmstart(
            initial_residual_warmstart=1.0,  # WORSE than standard
            initial_residual_standard=0.5,
            projected_residual=0.8,  # Also diverged
            iterations_warmstart=10,
            iterations_standard=8,
            similarity_score=0.9,
        )
        assert result.accepted is False
        assert "not lower" in result.rejection_reason

    def test_retrieval_order_deterministic(self, tmp_dir):
        """Same retrieval inputs → same output order."""
        mem = RetrievalMemory(str(Path(tmp_dir) / "retrieval"))
        # Add in random order
        for h in ["z_hash", "a_hash", "m_hash"]:
            mem.add(_make_entry(task_hash=h, topo="series"))
        results = mem.search_by_topology("series")
        hashes = [e.task_hash for e in results]
        # Must be sorted
        assert hashes == sorted(hashes)
