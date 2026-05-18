"""Reporting utilities for governed evaluation artifacts."""

from backend.reporting.eval_diff import EvalDiffResult, diff_eval_reports
from backend.reporting.discovery_report import DiscoveryReport, build_discovery_report
from backend.reporting.failure_summary import FailureSummary, detect_failure_trends, summarize_failures
from backend.reporting.report_builder import EvaluationReport, build_evaluation_report, validate_evaluation_report
from backend.reporting.retention_report import RetentionReport, build_retention_report
from backend.reporting.search_facets import SearchFacets, build_search_facets
from backend.reporting.workspace_summary import WorkspaceSummary, build_workspace_summary

__all__ = [
    "EvalDiffResult",
    "diff_eval_reports",
    "DiscoveryReport",
    "build_discovery_report",
    "FailureSummary",
    "detect_failure_trends",
    "summarize_failures",
    "EvaluationReport",
    "build_evaluation_report",
    "RetentionReport",
    "build_retention_report",
    "SearchFacets",
    "build_search_facets",
    "WorkspaceSummary",
    "build_workspace_summary",
    "validate_evaluation_report",
]
