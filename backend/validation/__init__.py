"""Validation layer."""

from .validator import Validator, validator
from .pipeline import ValidationPipeline, ValidationReport, ValidationCaseResult
from .thresholds import InvariantThresholds, invariant_family
from .oracle_arena import ArenaExample, ArenaResult, compare_oracle_vs_model, aggregate_arena_results
from .failure_taxonomy import classify_failure
