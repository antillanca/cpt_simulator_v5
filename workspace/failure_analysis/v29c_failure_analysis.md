# CPT v2.9C Failure Analysis

- Deterministic: True
- Run A fingerprint: b992ce9cc63cb630ce119e8e825ef346e0190cefa6312d4da12cc2738f2f2391
- Run B fingerprint: b992ce9cc63cb630ce119e8e825ef346e0190cefa6312d4da12cc2738f2f2391

## Failure Summary
- Dominant failure: conservation_drift
- OOD cases: 64

## Invariants
- IID KCL violation: 2.06e-01
- OOD KCL violation: 4.89e+07
- IID KVL violation: 2.00e+01
- OOD KVL violation: 2.38e+03
- IID power violation: 2.44e+00
- OOD power violation: 3.01e+09

## Speed
- Oracle mean: 0.000064 s
- Oracle p95: 0.000089 s
- Surrogate mean: 0.056731 s
- Surrogate p95: 0.084863 s
- Speedup: 0.00x