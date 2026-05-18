#!/usr/bin/env python3
"""CPT v2.13 — Memory Store Compaction Utility.

Deduplicates and rewrites the memory JSONL file atomically.
Usage: python scripts/compact_memory_store.py [--base-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root on path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend.core_runtime.memory_runtime import MemoryRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact memory JSONL store")
    parser.add_argument("--base-dir", default=None, help="Base directory for memory store")
    args = parser.parse_args()

    memory = MemoryRuntime(args.base_dir)
    before = memory.count()
    after = memory.compact()
    print(f"Compaction: {before} -> {after} entries ({before - after} duplicates removed)")


if __name__ == "__main__":
    main()
