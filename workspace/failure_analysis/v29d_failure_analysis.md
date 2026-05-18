# CPT v2.9C Failure Analysis

- Deterministic: True
- Run A fingerprint: dd15928cb2caafa8141450b58374c52718a682861706ff60434f6112c7834eaa
- Run B fingerprint: dd15928cb2caafa8141450b58374c52718a682861706ff60434f6112c7834eaa

## Failure Summary
- Dominant failure: conservation_drift
- OOD cases: 64

## Invariants
- IID KCL violation: 4.92e-01
- OOD KCL violation: 2.85e+07
- IID KVL violation: 1.05e+01
- OOD KVL violation: 3.16e+02
- IID power violation: 2.64e+01
- OOD power violation: 1.02e+09

## Speed
- Oracle mean: 0.000044 s
- Oracle p95: 0.000060 s
- Surrogate mean: 0.005843 s
- Surrogate p95: 0.032300 s
- Speedup: 0.01x