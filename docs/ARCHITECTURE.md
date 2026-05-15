# CPT v2.5 — Arquitectura

> **Resumen**: Sistema neuro-simbólico de 5 capas con confianza jerárquica. Capas 0-1 protegidas (no LLM writes). Capa 2 validada por invariantes. Capa 3 razona dentro de límites físicos. Hermes es tooling, no autoridad.

---

## Separación de Capas

| Capa | Directorio | Protección |
|:---|:---|:---|
| 0 — Core Truth (Sandbox Lua) | `backend/core_truth/` | 🔒 No LLM writes |
| 1 — Verificadores de Invariantes | `backend/verifiers/` | 🔒 No LLM writes |
| 2 — Filtros Neuronales Tabulares | `models/*.pt` | Validado por invariantes |
| 3 — Capa Cognitiva (Tutor/Student) | `backend/ai/` | Tutor define prompts, Student genera Lua |
| 4 — Hermes / LLMs externos | `backend/tooling/` | Tooling solamente |

---

## Modelo de Confianza

1. El sandbox determinista es la única verdad
2. Los LLMs pueden asistir con docs, tests, wrappers, refactors
3. Los LLMs NO pueden modificar leyes físicas, invariantes ni el compilador DSL
4. Los modelos neuronales se validan contra invariantes simbólicos antes de aceptarse
5. Parches sensibles requieren aprobación humana explícita (`CPT_HERMES_HUMAN_APPROVAL=1`)

---

## Flujo de Verificación

1. Compilar DSL a Lua seguro (o StudentEngine propone Lua directamente)
2. Ejecutar en sandbox hardened (Docker, sin red, read-only, `--cap-drop ALL`)
3. Recolectar traza (`collect_trace=True`)
4. Ejecutar invariantes (`verify_simulation(trace, invariant_set)`)
5. Comparar salidas neuronales vs salidas del sandbox
6. Rechazar modelos que excedan el umbral de violaciones

---

## Invariantes Disponibles

| Nombre | Archivo | Qué verifica |
|:---|:---|:---|
| `energy_conservation` | `invariants/energy.py` | KE + PE constante en la traza |
| `momentum_conservation` | `invariants/momentum.py` | Momentum total conservado |
| `logic_basic` | `invariants/logic.py` | Transiciones de estado siguen axiomas lógicos |

---

## Pipeline de Datasets

Los datasets se generan exclusivamente desde trazas del sandbox. Cada registro incluye:

- `question`
- `structured_state`
- `reasoning_trace`
- `answer`

---

## Permisos de Hermes

**Permitidos**: analyze_logs, suggest_refactor, generate_tests, verbalize_dataset, create_wrapper, detect_regressions, optimize_dataset, debugging_help

**Denegados**: modify_core_truth, modify_verifiers, modify_invariants, modify_dsl_compiler, approve_rules, merge

**Requieren aprobación humana**: apply_patch, merge, patch_core

El sistema solicita aprobación interactiva vía Telegram con botones. El orquestador pausa y espera **[Aprobar]** antes de ejecutar parches en archivos sensibles.

---

## Roadmap Arquitectónico

### Fase 1 (Activa)
- Sandbox hardening ✅
- Sistema de invariantes (3) ✅
- DSL base 🟡
- Generador de datasets sistemático 🟡
- Completar 6 módulos Lua pendientes ⏳

### Fase 2
- Dataset sintético (CPT como oráculo)
- Modularización neural

### Fase 3
- Fine-tuning transformer diminuto (LoRA/SFT 1B-3B)
- Export ONNX/TFLite/GGUF
- Suite de benchmarks
