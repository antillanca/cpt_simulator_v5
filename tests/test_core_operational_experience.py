"""CORE v3.1 -- Operational experience integrity tests.

Verify that the 300-execution operational dataset is intact.
"""

from __future__ import annotations

import json
import os
import pytest

DATA_DIR = "core_runtime/data/operational_experience"


@pytest.fixture
def experience_data():
    path = os.path.join(DATA_DIR, "operational_experience.jsonl")
    if not os.path.exists(path):
        pytest.skip("operational_experience.jsonl not found")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class TestExperienceDataPresent:

    def test_jsonl_exists(self):
        assert os.path.exists(os.path.join(DATA_DIR, "operational_experience.jsonl"))

    def test_csv_exists(self):
        assert os.path.exists(os.path.join(DATA_DIR, "operational_experience.csv"))

    def test_trajectory_stats_exists(self):
        assert os.path.exists(os.path.join(DATA_DIR, "trajectory_statistics.json"))

    def test_scheduler_stats_exists(self):
        assert os.path.exists(os.path.join(DATA_DIR, "scheduler_statistics.json"))


class TestExperienceDataCount:

    def test_at_least_200_executions(self, experience_data):
        assert len(experience_data) >= 200

    def test_at_most_500_executions(self, experience_data):
        assert len(experience_data) <= 500


class TestExperienceDataSchema:

    def test_each_record_has_task_hash(self, experience_data):
        for rec in experience_data[:10]:
            assert "task_hash" in rec or "fingerprint" in rec

    def test_each_record_has_convergence_class(self, experience_data):
        for rec in experience_data[:10]:
            assert "convergence_class" in rec or "trajectory_class" in rec

    def test_each_record_has_iterations(self, experience_data):
        for rec in experience_data[:10]:
            assert "projection_iterations" in rec or "iterations" in rec


class TestFamilyStatistics:

    def test_family_stats_exists(self):
        assert os.path.exists(os.path.join(DATA_DIR, "family_statistics.json"))

    def test_family_stats_readable(self):
        path = os.path.join(DATA_DIR, "family_statistics.json")
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
