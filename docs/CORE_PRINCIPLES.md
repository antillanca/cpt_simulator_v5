# CORE Principles

## The 10 Operational Guarantees

1. **Projection remains the final authority.**
   No scheduler decision, cache hit, or warmstart hint overrides
   the projection layer's correctness validation.

2. **The scheduler never modifies physics equations.**
   Scheduling controls budget allocation and stopping policies,
   not the mathematics of the domain.

3. **The scheduler never bypasses projection correctness.**
   Adaptive scheduling may reduce iterations, increase budget,
   or escalate to oracle, but it never skips projection.

4. **Exact cache always has priority.**
   If an exact match exists in the cache, it is returned
   without invoking surrogate or projection.

5. **Retrieval memory only provides initialization hints.**
   Retrieved similar solutions seed the initial guess; they
   never replace the projection convergence loop.

6. **Warmstart never replaces projection validation.**
   A warmstart solution must still pass through projection
   to verify correctness.

7. **All routing decisions are deterministic.**
   Given the same input, the capability router produces
   the same routing decision every time.

8. **Same input yields same execution trace.**
   Deterministic hashing, deterministic routing, deterministic
   scheduling: the entire execution pipeline is reproducible.

9. **Degraded executions never enter clean retrieval indexes.**
   Failed or degraded solutions are tagged and excluded from
   the FAISS retrieval index.

10. **Every execution is fully traceable.**
    Execution traces record every decision, every residual,
    every routing outcome, and every escalation.

## What CORE Is

CORE is not a simulator, nor a GNN framework, nor a physics solver.
It is a **deterministic hybrid runtime** that orchestrates oracle
computation, surrogate inference, constraint projection, memory
retrieval, and adaptive scheduling to execute verifiable cognitive
tasks.

Specifically:
- A scheduling, routing, caching, and tracing runtime
- Domain-agnostic: knows DomainTask, not CircuitTask
- Deterministic: same input = same trace
- Reproducible: frozen specs and hashes enable exact replay

## CPT Lineage

CORE's first validated domain is **CPT** (Circuit Projection Tool),
originating from `cpt_simulator_v5`. Circuits were the historical
first domain that validated the runtime architecture. The adaptive
scheduling, trajectory analysis, retrieval memory, and exact cache
were all designed and validated on circuit problems before being
abstracted into the domain-agnostic CORE framework.

CORE is the canonical runtime. CPT is the first-domain validation
lineage. The circuit domain code in `core_runtime/domains/circuits/`
carries forward all v2.15 guarantees unchanged.

## What CORE Is NOT

- NOT a learning system (no LoRA, no replay learning, no continual training)
- NOT a distributed runtime (single-process execution)
- NOT a physics engine (delegates domain computation to domain code)
- NOT a model trainer (surrogate models are provided by domains)
- NOT a replacement for projection (projection is the authority)
