#!/usr/bin/env python3
"""Compatibility wrapper around the deterministic model evaluation runner."""

from __future__ import annotations

from scripts.model_eval_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
