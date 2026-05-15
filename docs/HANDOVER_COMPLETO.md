# 🛡️ CPT Cognitive Engine v2.5 — Documento de Contexto Maestro

> **Propósito de este documento**: Transferir el contexto completo del proyecto a cualquier IA o desarrollador que lo retome. Al leerlo, debe ser posible entender qué hace el sistema, por qué se diseñó así, cuál es su estado actual y cuál es el siguiente paso.

---

## 🎯 ¿Qué es el CPT Cognitive Engine?

CPT es un **motor de razonamiento neuro-simbólico** construido desde cero. No es un LLM ajustado. Es un sistema que construye conocimiento de manera incremental, partiendo de axiomas matemáticos y físicos verificables, y los combina con redes neuronales diminutas para producir intuiciones rápidas y seguras.

### El Problema que Resuelve

Los LLMs grandes alucina porque internalizaron estadística del lenguaje, no física ni lógica verificable. CPT resuelve esto al revés:

1. **Primero construye la física**: Un sandbox determinista ejecuta las leyes reales del universo.
2. **Luego entrena intuiciones encima**: Redes neuronales pequeñas aprenden patrones de las simulaciones físicas.
3. **La IA solo puede razonar dentro de lo que la física permite**: Si la red sugiere algo físicamente imposible, el verificador lo rechaza antes de que ocurra.

### El Objetivo Final

Crear un modelo exportable en formato **`.gguf`** (compatible con llama.cpp, ejecutable en teléfonos y hardware embebido) que no alucine porque su conocimiento proviene de simulaciones matemáticas exactas, no de texto de internet.

---

## 🏗️ Arquitectura en Capas (CPT v2.5)

El sistema tiene una jerarquía estricta de confianza. Las capas inferiores son absolutamente autoritativas:

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

### Capa 0: Core Truth (`backend/core_truth/`)
- **Qué es**: Motor de simulación determinista escrito en Lua 5.4, ejecutado dentro de contenedores Docker sin red ni permisos de escritura.
- **Qué contiene**: El simulador de partículas, el currículo de conocimiento (`modules.json`), y el punto de entrada para ejecutar reglas físicas verificadas.
- **Regla clave**: **Nada por encima puede modificar esta capa sin revisión humana.**
- **Archivos clave**: `backend/core_truth/sandbox.py`, `backend/core_truth/modules.json`, `sandbox/lua/sandbox_runner.lua`

### Capa 1: Verifiers (`backend/verifiers/`)
- **Qué es**: Sistema de invariantes físicos. Verifica que cualquier traza de simulación cumple con leyes como la conservación de energía o del momentum.
- **Invariantes disponibles**: `energy_conservation`, `momentum_conservation`, `logic_basic`
- **Uso**: `verify_simulation(trace, invariant_set)` → devuelve `{passed, violations, metrics}`
- **Regla clave**: Ningún modelo neural o regla Lua es aceptado sin pasar primero por aquí.

### Capa 2: Filtros Neuronales Tabulares (`models/*.pt`)
- **Qué son**: Redes neuronales mínimas de PyTorch (arquitectura `input → 32 → 16 → 1`), una por dominio de conocimiento.
- **Cómo se entrenan**: El sandbox genera millones de escenarios. Los datos se suben a Kaggle para entrenamiento en la nube. El `.pt` resultante se descarga.
- **Modelos ya entrenados**:
  - `logic_tabular_filter.pt` — Principios lógicos aristotélicos
  - `geometry_tabular_filter.pt` — Geometría euclidiana
  - `numeric_tabular_filter.pt` — Aritmética y representación numérica
  - `proportion_tabular_filter.pt` — Ratios y porcentajes
  - `algebra_tabular_filter.pt` — Álgebra básica
  - `function_tabular_filter.pt` — Funciones matemáticas
  - `vector_tabular_filter.pt` — Vectores y descomposición
  - `trig_tabular_filter.pt` — Trigonometría
  - `newton_tabular_filter.pt` — Mecánica newtoniana
  - `action_tabular_filter.pt` — Navegación y movimiento básico

### Capa 3: Planificador A* y StudentEngine
- **Planificador A***: Usa los filtros neurales como heurística de poda. Si el filtro considera una acción física inviable, se descarta antes de explorarla.
- **StudentEngine** (`backend/ai/student_engine.py`): Un agente local (Qwen3 via Ollama) que propone código Lua para módulos de alta abstracción. Usa un ciclo de intento-error-reintento (máx 5 intentos). **Toda propuesta de código pasa por el Sandbox y los Verifiers antes de aceptarse.**

### 🤖 El Papel de Hermes (Tooling Asistido e Interactivo)
Hermes actúa como asistente de tooling experto. Para garantizar la seguridad sin sacrificar la agilidad, hemos implementado un sistema de **Aprobación Interactiva por Telegram**:
- Si Hermes detecta un error y propone un parche en un archivo protegido, el orquestador enviará una notificación con botones de **[Aprobar]** y **[Rechazar]**.
- El orquestador esperará tu respuesta (polling de 60s) antes de proceder.
- Una vez aprobado desde Telegram, Hermes aplica el parche y el entrenamiento continúa automáticamente.
- Hermes NO puede modificar `core_truth`, `verifiers` ni `invariants`. Tampoco puede hacer merge automático.
- `backend/tooling/permissions.py` implementa una lista explícita de acciones permitidas y denegadas. Los parches que afectan rutas protegidas requieren la variable de entorno `CPT_HERMES_HUMAN_APPROVAL=1`.
- **Modelo usado**: `meta-llama/llama-3.3-70b-instruct:free` via OpenRouter (gratuito).

---

## 🔄 El Orquestador de Entrenamiento (`scripts/training_orchestrator.py`)

El entrenamiento es completamente autónomo. El orquestador sigue este ciclo:

```
┌─ Leer modules.json ──────────────────────────────────────────┐
│                                                               │
│  ¿Hay módulos "pending"?  →  NO  →  🏁 CURRRÍCULO COMPLETO   │
│           │                                                   │
│           ↓ SÍ                                               │
│                                                               │
│  ¿engine_type == "tabular"?                                  │
│    → Ejecutar fábrica DPO (planner/*_automation.py)          │
│    → Subir dataset a Kaggle, entrenar, descargar .pt         │
│    → Correr verificación de invariantes en Sandbox           │
│    → ✅ CONFIRMAR módulo  |  ❌ Llamar a Hermes              │
│                                                               │
│  ¿engine_type == "lua"?                                      │
│    → StudentEngine genera código Lua (hasta 5 intentos)      │
│    → Ejecutar en Sandbox, recolectar traza                   │
│    → Verificar contra invariants[] del módulo                │
│    → ✅ CONFIRMAR módulo  |  ❌ Llamar a Hermes              │
│                                                               │
│  Hermes: propone parche → PAUSA → Espera aprobación humana   │
└───────────────────────────────────────────────────────────────┘
```

**Notificaciones**: Cada evento relevante (inicio, confirmación, error, intervención de Hermes) se envía como alerta a Telegram via `backend/notifier.py`.

---

## 📊 Estado Actual del Currículo (May 2025)

- **Total de módulos**: 43
- **Confirmados** ✅: 36 (84%)
- **Pendientes** ⏳: 6 (14%)

### Módulos pendientes (bloqueando el avance):
|| Nivel | Módulo | Materia | Problema conocido ||
||:---:|:---|:---|:---||
|| 13 | `waves_frequency_amplitude` | Ondas | Sin código Lua ||
|| 14 | `magnetism_lorentz_force` | Magnetismo | Fuerza de Lorentz incorrecta (1 rechazo) ||
|| 23 | `lagrangian_mechanics` | Lagrangiano | Sin código Lua ||
|| 25 | `electromagnetism_maxwell` | EM | Error de compilación Lua (1 rechazo) ||
|| 28 | `quantum_mechanics_wavefunction` | Cuántica | Sin código Lua ||
|| 34 | `quantum_double_slit_logic` | Cuántica | Sin código Lua ||

**Recientes**: `energy_potential`, `energy_conservation` (nivel 12) y `waves_oscillation` (nivel 13) confirmados por owl-alpha.

---

## 🗺️ ROADMAP

### Fase actual — Fase 1: Completar el Motor de Verdad
Completar los 9 módulos Lua pendientes. Una vez que `energy_conservation` pase los verificadores, el agente dominará desde lógica aristotélica hasta mecánica ondulatoria.

### Fase 1.5 — Expansión del Currículo
Ampliar `modules.json` con nuevas disciplinas que heredan de las bases existentes:
- **Programación**: Variables, bucles, algoritmos (hereda de Álgebra e Inglés)
- **Electrónica**: Compuertas, transistores, circuitos (hereda de Electromagnetismo y Lógica)

### Fase 2 — Generación de Dataset Sintético
El motor CPT actúa como "oráculo": un LLM le hace preguntas condicionales y CPT responde ejecutando simulaciones. Se genera un dataset masivo de razonamiento paso a paso con alta pureza factual.

### Fase 3 — Destilación a `.gguf`
Fine-tuning (LoRA/SFT) de un modelo base de 1B–3B parámetros usando el dataset de la Fase 2. Exportar en formato `.gguf` para correr en llama.cpp, teléfonos o hardware embebido. El modelo final tiene el razonamiento físico internalizado en sus pesos, sin necesidad de un simulador en tiempo real.

---

## 📂 Mapa de Archivos Críticos

```
cpt_simulator_v5/
│
├── backend/
│   ├── core_truth/              ← 🔒 PROTEGIDO — Fuente de verdad
│   │   ├── modules.json         ← Currículo completo (43 módulos)
│   │   └── sandbox.py           ← API Python del sandbox Docker/Lua
│   ├── verifiers/               ← 🔒 PROTEGIDO — Árbitros matemáticos
│   │   ├── simulation.py        ← verify_simulation(trace, invariant_set)
│   │   └── invariants/          ← energy, momentum, logic
│   ├── tooling/                 ← Hermes y permisos
│   │   ├── hermes.py            ← IntelligentToolingAssistant
│   │   └── permissions.py       ← PermissionPolicy (allowlist/denylist)
│   ├── ai/
│   │   ├── student_engine.py    ← Qwen3 local → propone Lua → valida
│   │   └── tutor_engine.py      ← Cascada de LLMs avanzados (OpenRouter)
│   └── notifier.py              ← Alertas a Telegram
│
├── scripts/
│   └── training_orchestrator.py ← 🎛️ Director maestro del entrenamiento
│
├── planner/
│   └── *_automation.py          ← Fábricas DPO por dominio (Kaggle)
│
├── models/
│   └── *.pt                     ← Filtros neurales entrenados
│
├── sandbox/lua/
│   └── sandbox_runner.lua       ← Intérprete Lua hardened (Docker)
│
└── docs/
    ├── ARCHITECTURE_V25.md      ← Separación de capas y modelo de confianza
    ├── CHANGELOG_HERMES.md      ← Registro de intervenciones de Hermes
    └── ESTADO_CURRICULO.md      ← Estado de los 43 módulos
```

---

## ⚙️ Variables de Entorno Requeridas

```bash
# Credenciales de Kaggle para entrenar modelos en la nube
KAGGLE_USERNAME=...
KAGGLE_KEY=...

# API de OpenRouter para Hermes y el TutorEngine (usar modelos free primero)
OPENROUTER_API_KEY=...

# Bot de Telegram para notificaciones
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# Para aprobar parches sensibles de Hermes manualmente
CPT_HERMES_HUMAN_APPROVAL=0   # Cambiar a "1" para autorizar un parche
```

---

## 🚦 Cómo Continuar el Entrenamiento

```bash
# Desde el directorio raíz del proyecto
cd /home/john/www/cpt_simulator_v5

# Ejecutar en background con logs
nohup python3 scripts/training_orchestrator.py > training_full.log 2>&1 &

# Monitorear en tiempo real
tail -f training_full.log

# Ver el estado del currículo
cat docs/ESTADO_CURRICULO.md

# Revisar intervenciones de Hermes
cat docs/CHANGELOG_HERMES.md
```

---

## 🧠 Principios de Diseño (NO negociables)

1. **El Sandbox es la única verdad**. Si el sandbox dice que algo viola la física, se rechaza sin excepción.
2. **Los LLMs son herramientas, no autoridades**. Proponen, el verificador matemático aprueba.
3. **Todo modelo neural debe superar invariantes antes de ser confirmado**. No existe "funciona porque no crasheó".
4. **Los parches de IA a código sensible requieren aprobación humana explícita**. Sin merge automático.
5. **El entrenamiento es incremental y dependiente**. No se puede aprender mecánica cuántica sin dominar antes la conservación de energía.
