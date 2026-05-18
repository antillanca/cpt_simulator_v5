"""Optional GPU/Kaggle training hooks for CPT v2.7.

Isolated module that prepares dataset packaging, shard upload,
and config-driven training profiles for future remote GPU use.
NOT required for local operation. No active GPU training here.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KAGGLE_HOOK_VERSION = "2.7.0"


@dataclass
class TrainingProfile:
    """Config-driven training profile for remote GPU jobs."""

    name: str = "default"
    model_type: str = "tiny_transformer"
    epochs: int = 10
    lr: float = 1e-4
    batch_size: int = 32
    seed: int = 42
    train_split: float = 0.8
    dataset_version: str = "2.7.0"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "train_split": self.train_split,
            "dataset_version": self.dataset_version,
            "extra": self.extra,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def from_file(cls, path: str | Path) -> TrainingProfile:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data.get("name", "default"),
            model_type=data.get("model_type", "tiny_transformer"),
            epochs=data.get("epochs", 10),
            lr=data.get("lr", 1e-4),
            batch_size=data.get("batch_size", 32),
            seed=data.get("seed", 42),
            train_split=data.get("train_split", 0.8),
            dataset_version=data.get("dataset_version", "2.7.0"),
            extra=data.get("extra", {}),
        )


def package_dataset_for_upload(
    dataset_path: str | Path,
    manifest_path: str | Path | None = None,
    output_dir: str | Path = "kaggle_upload",
    profile: TrainingProfile | None = None,
) -> Path:
    """Package a dataset for future upload to Kaggle or remote GPU.

    Creates a directory with:
    - dataset.jsonl (copied)
    - dataset.manifest.json (copied if exists)
    - training_profile.json
    - README.md with metadata

    Does NOT upload anything. Just prepares the package.
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy dataset
    dest_dataset = output_dir / dataset_path.name
    if dataset_path.exists():
        shutil.copy2(dataset_path, dest_dataset)

    # Copy manifest
    if manifest_path:
        manifest_path = Path(manifest_path)
        if manifest_path.exists():
            dest_manifest = output_dir / manifest_path.name
            shutil.copy2(manifest_path, dest_manifest)
    else:
        auto_manifest = dataset_path.with_suffix(".manifest.json")
        if auto_manifest.exists():
            shutil.copy2(auto_manifest, output_dir / auto_manifest.name)

    # Write profile
    profile = profile or TrainingProfile()
    profile.save(output_dir / "training_profile.json")

    # Write README
    readme = f"""# CPT v2.7 Dataset Package

- Dataset: {dataset_path.name}
- Version: {profile.dataset_version}
- Model type: {profile.model_type}
- Profile: {profile.name}

## Usage
1. Upload this directory to Kaggle as a dataset
2. Create a notebook using the training_profile.json
3. Train the model against the oracle dataset
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    return output_dir


def create_kaggle_metadata(
    dataset_title: str = "cpt-v27-oracle",
    owner_slug: str = "cpt-simulator",
) -> dict[str, Any]:
    """Create Kaggle dataset metadata.json for upload.

    This is a helper only — actual upload requires Kaggle API keys.
    """
    return {
        "title": dataset_title,
        "id": f"{owner_slug}/{dataset_title}",
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["physics", "reasoning", "distillation", "cpt-simulator"],
    }
