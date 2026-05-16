# CPT Trace Schema

## Required Fields

Each structured reasoning step contains:

```json
{
  "step_id": 0,
  "rule": "string",
  "equation": "string",
  "inputs": {},
  "operation": "string",
  "intermediate_result": {},
  "invariants_checked": [],
  "verification": {
    "passed": true,
    "violations": []
  },
  "timestamp": 0.0
}
```

## Rules

1. Step ordering is deterministic and sorted by `step_id`.
1. Serialization is canonical and JSON-compatible.
1. Replay is replayable from the recorded before/after states.
1. Verification metadata is retained on every step.

## Replay Contract

Replay rehydrates the recorded step chain without re-running Lua. It checks continuity between recorded states and flags mismatches as violations.

