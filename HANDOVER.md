# CPT Cognitive Engine v2.5 — Handover

> **Propósito**: Transferir el contexto completo del proyecto a cualquier IA o desarrollador que lo retome.

---

## Estado Actual

- **Módulos**: 43 total — 3 confirmados (7%), 37 pendientes (86%), 3 rechazados (7%)
- **Fase**: 1 — Regenerando módulos con tracking de `generated_by`
- **Repositorio**: [antillanca/cpt_simulator_v5](https://github.com/antillanca/cpt_simulator_v5) (Público)
- **Siguiente paso**: Orquestador regenerando módulos automáticamente

**Nota**: El 2026-05-15 se resetearon 37 módulos confirmed→pending para regenerar con el nuevo campo `generated_by` que registra qué modelo (Ollama/OpenRouter) produjo cada código Lua.

| Nivel | Módulo | Problema |
|:---:|:---|:---|
| 13 | `waves_oscillation` | Rechazado (4 intentos) |
| 14 | `magnetism_lorentz_force` | Rechazado (6 intentos) — bloqueo pedagógico |
| 23 | `lagrangian_mechanics` | Rechazado (4 intentos) |
| 34 | `quantum_double_slit_logic` | Sin código Lua |

**Recientes confirmados**: `energy_potential` y `energy_conservation` (nivel 12, owl-alpha). `waves_oscillation` también confirmado.

---

## ¿Qué es?

CPT es un **motor de razonamiento neuro-simbólico** construido desde cero. No es un LLM ajustado. Construye conocimiento incrementalmente: parte de axiomas matemáticos y físicos verificables, y los combina con redes neuronales diminutas para producir intuiciones rápidas y seguras.

**El problema que resuelve**: Los LLMs grandes alucinan porque internalizaron estadística del lenguaje, no física ni lógica verificable. CPT resuelve esto al revés:
1. Primero construye la física (sandbox determinista ejecuta leyes reales)
2. Luego entrena intuiciones encima (redes neuronales aprenden de simulaciones)
3. La IA solo razona dentro de lo que la física permite

**Objetivo final**: Modelo exportable en `.gguf` (llama.cpp, teléfonos, embebido) que no alucina porque su conocimiento proviene de simulaciones matemáticas exactas.

---

## Arquitectura

```
┌──────────────────────────────────────────────────┐
│  4. Hermes / LLMs externos   (tooling solamente) │  ← Sugiere, nunca aprueba
├──────────────────────────────────────────────────┤
│  3. StudentEngine / Planificador A*              │  ← Razona dentro de límites
├──────────────────────────────────────────────────┤
│  2. Filtros Neuronales Tabulares (PyTorch .pt)   │  ← Intuición rápida, validada
├──────────────────────────────────────────────────┤
│  1. Verifiers / Invariantes Simbólicos           │  ← Árbitro matemático
├──────────────────────────────────────────────────┤
│  0. Core Truth / Sandbox Lua  ← FUENTE DE VERDAD │  ← Absolutamente determinista
└──────────────────────────────────────────────────┘
```

- **Capa 0** (`backend/core_truth/`): Simulador determinista en Lua 5.4, Docker sin red. **PROTEGIDO.**
- **Capa 1** (`backend/verifiers/`): Invariantes físicos (energía, momentum, lógica). **PROTEGIDO.**
- **Capa 2** (`models/*.pt`): Redes PyTorch mínimas, una por dominio. Entrenadas vía Kaggle.
- **Capa 3**: StudentEngine (Qwen3 local) propone código Lua. Planificador A* usa filtros como heurística.
- **Hermes**: Tooling asistido con aprobación interactiva por Telegram. NO puede modificar capas 0-1.

---

## Flujo Principal

```
Leer modules.json → ¿pending?
  → tabular: fábrica DPO → Kaggle GPU → descargar .pt → verificar invariantes → confirmar
  → lua: StudentEngine genera código → sandbox → verificar invariants → confirmar
  → error: Hermes propone parche → pausa → aprobación humana Telegram → continuar
```

---

## Cómo Operar

```bash
# Arrancar entrenamiento
cd /home/john/www/cpt_simulator_v5
nohup python3 scripts/training_orchestrator.py > training_full.log 2>&1 &

# Monitorear
tail -f training_full.log

# Ver estado del currículo
cat docs/ESTADO_CURRICULO.md

# Revisar intervenciones Hermes
cat docs/CHANGELOG_HERMES.md
```

---

## Mapa de Archivos

```
cpt_simulator_v5/
├── HANDOVER.md                       ← este archivo
├── backend/
│   ├── core_truth/                   ← 🔒 PROTEGIDO
│   │   ├── modules.json              ← currículo (43 módulos)
│   │   └── sandbox.py                ← API del sandbox
│   ├── verifiers/                    ← 🔒 PROTEGIDO
│   ├── tooling/                      ← Hermes + permisos
│   └── ai/
│       ├── student_engine.py         ← Qwen3 local
│       └── tutor_engine.py           ← LLMs OpenRouter
├── scripts/training_orchestrator.py  ← orquestador maestro
├── planner/*_automation.py           ← fábricas DPO
├── models/*.pt                       ← filtros neuronales
├── sandbox/lua/sandbox_runner.lua    ← intérprete Lua hardened
└── docs/
    ├── ARCHITECTURE.md               ← detalle técnico de capas
    ├── CLICK_ONDA.md                 ← flujo de desarrollo
    ├── CONTEXTO.md                   ← glosario + decisiones
    ├── ESTADO_CURRICULO.md           ← detalle de los 43 módulos
    ├── CHANGELOG_HERMES.md           ← intervenciones Hermes
    ├── GUIA_AGENTE_SYLLABUS.md       ← guía para agentes
    └── HANDOVER_COMPLETO.md          ← handover anterior completo
```

---

## Configuración

```bash
KAGGLE_USERNAME=...
KAGGLE_KEY=...
OPENROUTER_API_KEY=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
CPT_HERMES_HUMAN_APPROVAL=0   # cambiar a "1" para autorizar parches
```

---

## Roadmap

- **Fase 1** (actual): Completar 9 módulos Lua pendientes
- **Fase 1.5**: Expandir currículo (programación, electrónica)
- **Fase 2**: Dataset sintético (CPT como oráculo)
- **Fase 3**: Destilación a `.gguf` (LoRA/SFT 1B-3B)

---

## Principios de Diseño (NO negociables)

1. El Sandbox es la única verdad
2. Los LLMs son herramientas, no autoridades
3. Todo modelo neural debe superar invariantes antes de confirmarse
4. Parches de IA a código sensible requieren aprobación humana
5. El entrenamiento es incremental y dependiente

---

## Ver También

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — detalle técnico de capas, invariantes, permisos Hermes
- [docs/CLICK_ONDA.md](docs/CLICK_ONDA.md) — flujo de desarrollo Onda/Click
- [docs/CONTEXTO.md](docs/CONTEXTO.md) — glosario y decisiones de diseño
- [docs/ESTADO_CURRICULO.md](docs/ESTADO_CURRICULO.md) — detalle de los 43 módulos
