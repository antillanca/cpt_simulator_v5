"""CORE v3.1 -- Manifest integrity tests.

Verify that CORE_v3_FOUNDATION_MANIFEST.json is valid and complete.
"""

from __future__ import annotations

import json
import os
import pytest

MANIFEST_PATH = "CORE_v3_FOUNDATION_MANIFEST.json"


@pytest.fixture
def manifest():
    if not os.path.exists(MANIFEST_PATH):
        pytest.skip("Manifest not found")
    with open(MANIFEST_PATH) as f:
        return json.load(f)


class TestManifestPresent:

    def test_manifest_exists(self):
        assert os.path.exists(MANIFEST_PATH)

    def test_manifest_is_valid_json(self, manifest):
        assert isinstance(manifest, dict)


class TestManifestVersion:

    def test_has_manifest_version(self, manifest):
        assert "manifest_version" in manifest

    def test_has_package_version(self, manifest):
        assert "package_version" in manifest

    def test_has_frozen_runtime_version(self, manifest):
        assert "frozen_runtime_version" in manifest


class TestManifestCoreComponents:

    def test_has_domains(self, manifest):
        assert "domains" in manifest

    def test_has_test_counts(self, manifest):
        assert "test_counts" in manifest

    def test_has_operational_experience(self, manifest):
        assert "operational_experience" in manifest


class TestManifestTestData:

    def test_has_core_tests(self, manifest):
        tc = manifest["test_counts"]
        assert "core_tests" in tc
        assert tc["core_tests"] >= 30

    def test_has_total_tests(self, manifest):
        tc = manifest["test_counts"]
        assert "total_tests" in tc
        assert tc["total_tests"] >= 600


class TestManifestExperienceAssets:

    def test_experience_section_has_execution_count(self, manifest):
        oe = manifest["operational_experience"]
        assert "execution_count" in oe
        assert oe["execution_count"] >= 200

    def test_experience_section_has_location(self, manifest):
        oe = manifest["operational_experience"]
        assert "location" in oe

    def test_experience_section_has_sha256(self, manifest):
        oe = manifest["operational_experience"]
        assert "sha256" in oe
