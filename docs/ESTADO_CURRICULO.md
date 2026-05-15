# Estado del Currículo — CPT v2.5

> **Resumen**: 43 módulos totales — 3 confirmados (7%), 37 pendientes (86%), 3 rechazados (7%). Regeneración en curso con tracking de `generated_by`.

---

## Estadísticas Globales

| Métrica | Valor |
|:---|:---:|
| Total de módulos | **43** |
| Confirmados | **3** (7%) |
| Pendientes | **37** (86%) |
| Rechazados | **3** (7%) |

Fuente de verdad: `backend/core_truth/modules.json`

**Nota**: Los 37 módulos fueron reseteados de confirmed→pending el 2026-05-15 para regenerar con el nuevo campo `generated_by` que registra qué modelo Ollama produjo cada código Lua.

---

## Módulos Confirmados (regenerados)

| Nivel | Módulo | confirmed_by | generated_by |
|:---:|:---|:---|:---|
| L00 | layer_00_existence | hermes_override | hermes_override |
| L00 | logic_02_non_contradiction | hermes_override | hermes_override |
| L00 | logic_03_excluded_middle | kaggle_orchestrator | kaggle_orchestrator |

## Módulos Rechazados

| Nivel | Módulo | rejection_count |
|:---:|:---|:---:|
| L13 | waves_oscillation | 4 |
| L14 | magnetism_lorentz_force | 6 |
| L23 | lagrangian_mechanics | 4 |

---

## Estado de Capas de Arquitectura

| Capa | Directorio | Estado |
|:---|:---|:---:|
| Core Truth (Sandbox Lua) | `backend/core_truth/` | ✅ Activo |
| Verificadores de Invariantes | `backend/verifiers/` | ✅ Activo (3 invariantes) |
| DSL (YAML→Lua) | `backend/dsl/` | 🟡 Parcial |
| Generación de datasets | `backend/datasets/` | 🟡 Parcial |
| Tooling Hermes | `backend/tooling/` | ✅ Activo |

---

## Currículo por Dominio

| Materia | Nivel | Motor | Estado | Descripción |
|:---|:---:|:---|:---:|:---|
| **Lógica Aristotélica** | 0 | .pt | ✅ | Identidad, No-Contradicción, Tercero Excluido |
| **Conteo** | 1 | .pt | ✅ | Representación de cantidades |
| **Operaciones** | 2 | .pt | ✅ | +, -, ×, ÷ |
| **Números** | 3 | .pt | ✅ | Enteros, fracciones, potencias |
| **Proporciones** | 4 | .pt | ✅ | Ratios, escalas, porcentajes |
| **Álgebra** | 5 | .pt | ✅ | Variables y ecuaciones |
| **Funciones** | 6 | .pt | ✅ | Transformaciones deterministas |
| **Geometría** | 7 | .pt | ✅ | Distancia euclidiana |
| **Vectores** | 8 | .pt | ✅ | Descomposición vectorial |
| **Trigonometría** | 9 | .pt | ✅ | sin, cos, ángulos |
| **Cinemática** | 10 | .pt | ✅ | Movimiento, velocidad, posición |
| **Dinámica / Newton** | 11 | .pt | ✅ | F=ma, segunda ley |
| **Energía Cinética** | 12 | Lua | ✅ | KE = ½mv² |
|| **Energía Potencial** | 12 | Lua | ✅ | mgh — confirmado por owl-alpha |
|| **Conservación de Energía** | 12 | Lua | ✅ | KE+PE = constante — confirmado por owl-alpha |
|| **Oscilación** | 13 | Lua | ✅ | x = A·sin(t) — confirmado |
|| **Frecuencia / Amplitud** | 13 | Lua | ⏳ | Sin código Lua |
| **Ley de Ohm** | 14 | Lua | ✅ | V=IR, P=VI |
|| **Fuerza de Lorentz** | 14 | Lua | ⏳ | F=qvB — física incorrecta (1 rechazo) |
| **Termodinámica** | 15 | Lua | ✅ | ΔT = Q/(mc) |
| **Entropía** | 15 | Lua | ✅ | ΔS = Q/T |
| **Probabilidad** | 16 | Lua | ✅ | Media, distribuciones |
| **Oscilador Amortiguado** | 17 | Lua | ✅ | Modelado avanzado |
| **Derivada Numérica** | 18 | Lua | ✅ | dx/dt |
| **Integral Numérica** | 19 | Lua | ✅ | Acumulación de velocidad |
| **EDO (Euler)** | 20 | Lua | ✅ | dx/dt = -0.5x |
| **Álgebra Lineal** | 21 | Lua | ✅ | Rotación 2D, matrices |
| **Análisis Numérico** | 22 | Lua | ✅ | Caída libre, Euler |
| **Lagrangiano** | 23 | Lua | ⏳ | L = KE - PE — sin código |
| **Hamiltoniano** | 24 | Lua | ✅ | H = KE + PE |
|| **Maxwell / EM** | 25 | Lua | ⏳ | Error de compilación Lua (1 rechazo) |
| **Relatividad Especial** | 26 | Lua | ✅ | Factor de Lorentz γ |
| **Relatividad General** | 27 | Lua | ✅ | Intervalo espacio-tiempo |
| **Función de Onda** | 28 | Lua | ⏳ | |ψ|² — sin código |
| **Teoría Cuántica de Campo** | 29 | Lua | ✅ | Oscilador cuántico armónico |
| **Expansión Cosmológica** | 30 | Lua | ✅ | Ley de Hubble |
| **Caos / Logístico** | 31 | Lua | ✅ | Mapa logístico |
| **Conocimiento Frontera** | 32 | Lua | ✅ | Unidades de Planck |
| **Doble Rendija** | 34 | Lua | ⏳ | Superposición — sin código |

---

## Capacidades Adquiridas

- [x] Poda Lógica: descartar acciones físicamente imposibles
- [x] Heurística Geométrica: evitar colisiones por posición relativa
- [x] Aritmética y Álgebra: cálculos numéricos exactos
- [x] Dinámica Newtoniana: fuerzas, inercia y trayectorias
- [x] Verificación por Invariantes: ningún modelo se acepta sin validación
- [ ] Conservación de Energía (en progreso — bloquea niveles 13+)
- [ ] Electromagnetismo completo (bloqueado por nivel 12-13)

---

## Herramientas Principales

| Herramienta | Archivo | Función |
|:---|:---|:---|
| Orquestador | `scripts/training_orchestrator.py` | Director del ciclo de entrenamiento |
| Fábricas DPO | `planner/*_automation.py` | Generan datasets y suben a Kaggle |
| StudentEngine | `backend/ai/student_engine.py` | Qwen3 local genera código Lua |
| TutorEngine | `backend/ai/tutor_engine.py` | Cascada LLM avanzado (OpenRouter) |
| Verificadores | `backend/verifiers/simulation.py` | Árbitro matemático de invariantes |
| Sandbox | `backend/core_truth/sandbox.py` | Ejecuta Lua en Docker aislado |
| Hermes | `backend/tooling/hermes.py` | Asistente de tooling (permisos restringidos) |
| Notificador | `backend/notifier.py` | Alertas a Telegram |
