# CPT Cognitive Engine v2.5

Motor de razonamiento neuro-simbólico. Aprende física y matemática desde cero vía simulaciones deterministas + redes neuronales.

## Repositorio Oficial

→ **[GitHub: antillanca/cpt_simulator_v5](https://github.com/antillanca/cpt_simulator_v5)**


## Documento Maestro

→ **[HANDOVER.md](HANDOVER.md)** — Contexto completo (arquitectura, estado, flujo, configuración, roadmap)

## Estado Rápido

- **Módulos**: 43 total — 43 confirmados (100%), 0 pendientes (0%), 0 rechazados (0%)
- **Fase**: 2 — Currículo completo. Iniciando fase de distillación y expansión.
- **Stack**: FastAPI + Lua sandbox + PyTorch + Ollama (Qwen3) + Kaggle GPUs
- **Nuevo**: Campo `generated_by` registra qué modelo produjo cada código Lua

## Estructura

```
cpt_simulator_v5/
├── HANDOVER.md              ← contexto completo
├── backend/
│   ├── core_truth/          ← sandbox Lua + currículo (PROTEGIDO)
│   ├── verifiers/           ← invariantes físicos (PROTEGIDO)
│   ├── tooling/             ← Hermes + permisos
│   └── ai/                  ← StudentEngine + TutorEngine
├── scripts/                 ← training_orchestrator.py
├── planner/                 ← fábricas DPO
├── models/                  ← filtros neuronales (.pt)
├── sandbox/lua/             ← intérprete Lua hardened
└── docs/                    ← documentación de detalle
```

## Ver también

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — detalle técnico de capas
- [docs/CLICK_ONDA.md](docs/CLICK_ONDA.md) — flujo de desarrollo Onda/Click
- [docs/CONTEXTO.md](docs/CONTEXTO.md) — glosario y decisiones de diseño
- [docs/ESTADO_CURRICULO.md](docs/ESTADO_CURRICULO.md) — detalle de los 43 módulos
