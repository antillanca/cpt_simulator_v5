# CORE Operational Explainability

## Philosophy

CORE does not use language models for explanations.
All explanations are deterministic, rule-based, and reproducible.

## How It Works

Every execution produces:

1. **TrustworthinessAudit** — classification (CERTAIN/TRANSITIONAL/INDETERMINATE)
2. **ExecutionExplanation** — deterministic post-hoc explanation
3. **Optional FailureReport** — if the execution degraded or failed

These are composed into an **AuditTraceRecord** and stored in the
**AuditTraceStore**.

## Explanation Rules

| Condition | Explanation Type |
|-----------|-----------------|
| Cache hit | `exact_cache_hit` |
| Warmstart used | `warmstart_assisted` |
| Escalation triggered | `projection_escalated` |
| OOD topology | `ood_conservative` |
| Unstable trajectory | `unstable_convergence` |
| CERTAIN trust | `certain_execution` |
| TRANSITIONAL trust | `transitional_execution` |
| INDETERMINATE trust | `indeterminate_execution` |

Priority: cache_hit > escalation > unstable > OOD > warmstart > trust_level

## Example

```python
from core_runtime.core.trustworthiness_runtime import audit_execution
from core_runtime.core.explainability_runtime import ExplainabilityRuntime

audit = audit_execution(
    task_hash="task_001",
    confidence_score=0.95,
    uncertainty_score=0.05,
    projection_iterations=3,
    final_residual=0.0001,
    trajectory_class="fast_converging",
    escalation_required=False,
    ood=False,
)

explainer = ExplainabilityRuntime()
explanation = explainer.explain(audit)

print(explanation.summary)
# "Standard execution completed with high confidence and low residual."

print(explanation.explanation_type)
# "certain_execution"

print(explanation.key_factors)
# ("high_confidence", "residual=1.0000e-04", "confidence=0.950", "iterations=3")
```

## Traceability

Every explanation has a SHA-256 fingerprint derived from all its fields.
Same input -> same fingerprint. Guaranteed deterministic.

## What This Is NOT

- This is NOT an LLM-generated explanation system
- This does NOT score human values, morality, or ideology
- This does NOT alter execution in any way
- This does NOT replace human review of indeterminate cases
