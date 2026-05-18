"""Deterministic dataloader for CPT v2.7 training scaffold.

Loads JSONL datasets or sharded datasets, provides deterministic
train/eval splits based on seed, and iterates records.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator


class DatasetShardLoader:
    """Deterministic dataset loader with train/eval splitting.

    Supports loading from:
    - A single JSONL file
    - A sharded dataset directory with manifest

    Split is deterministic: same seed always produces same split.
    """

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        shard_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        seed: int = 42,
        train_split: float = 0.8,
    ):
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.shard_dir = Path(shard_dir) if shard_dir else None
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.seed = seed
        self.train_split = train_split
        self._records: list[dict[str, Any]] | None = None
        self._train: list[dict[str, Any]] | None = None
        self._eval: list[dict[str, Any]] | None = None

    def load(self) -> list[dict[str, Any]]:
        """Load all records from the dataset."""
        if self._records is not None:
            return self._records

        if self.dataset_path and self.dataset_path.exists():
            from backend.datasets.loader import load_jsonl
            self._records = load_jsonl(self.dataset_path, validate=False)
        elif self.shard_dir and self.manifest_path:
            from backend.datasets.loader import load_sharded_dataset
            rows, _ = load_sharded_dataset(self.shard_dir, self.manifest_path, validate=False)
            self._records = rows
        else:
            raise ValueError("Must provide either dataset_path or (shard_dir + manifest_path)")

        return self._records

    def split_data(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Deterministically split data into train and eval sets.

        Returns:
            (train_records, eval_records)
        """
        if self._train is not None and self._eval is not None:
            return self._train, self._eval

        records = self.load()
        rng = random.Random(self.seed)
        indices = list(range(len(records)))
        rng.shuffle(indices)

        split_point = int(len(records) * self.train_split)
        train_indices = sorted(indices[:split_point])
        eval_indices = sorted(indices[split_point:])

        self._train = [records[i] for i in train_indices]
        self._eval = [records[i] for i in eval_indices]
        return self._train, self._eval

    @property
    def train(self) -> list[dict[str, Any]]:
        """Training split records."""
        train, _ = self.split_data()
        return train

    @property
    def eval(self) -> list[dict[str, Any]]:
        """Evaluation split records."""
        _, eval_data = self.split_data()
        return eval_data

    def iter_train(self) -> Iterator[dict[str, Any]]:
        """Iterate over training records."""
        yield from self.train

    def iter_eval(self) -> Iterator[dict[str, Any]]:
        """Iterate over evaluation records."""
        yield from self.eval

    @property
    def record_count(self) -> int:
        return len(self.load())

    @property
    def train_count(self) -> int:
        return len(self.train)

    @property
    def eval_count(self) -> int:
        return len(self.eval)
