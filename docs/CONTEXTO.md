# Contexto — Glosario y Decisiones de Diseño

> **Resumen**: Conceptos clave del ecosistema CPT + SimpleRestoBar. Define la terminología fundamental (Onda, Click, Colapso, i@, Leyes Ancla) y documenta las decisiones de diseño tomadas.

---

## Glosario de Conceptos Clave

### Onda
Modo de desarrollo fluido. Código vive en variables/plantillas, se inyecta dinámicamente, nunca toca el proyecto base. Estado líquido/experimental. Reversible.

**Ejemplo**: Profesor escribe `i@ p.vy += 0.2` en dev_panel. Motor JS ejecuta `new Function("p", "p.vy += 0.2")` en tiempo real. Si se rompe, se descarta la variable.

### Click
Modo de compilación instantánea. Congela código de variables en archivos fijos, genera artefacto inmutable (Docker image, commit tag). Estado sólido. Irreversible.

**Ejemplo**: Profesor valida `p.vy += 0.2`, ejecuta `click-freeze`. Contenido se escribe en `engine.js`. `click-build` genera Docker image `v1.3.0`.

### Colapso
Transición de Onda a Click. Momento donde el usuario elige una posibilidad y las demás colapsan. Análogo al colapso de la función de onda en mecánica cuántica.

**Regla**: Sin validación no hay Colapso.

### Elección Analógica
Decisión en modo Onda. Múltiples posibilidades coexisten (superposición cuántica). Reversible, exploratoria, no destructiva.

### Elección Digital
Decisión en modo Click. Una única opción se selecciona y se congela. Irreversible, binaria, productiva.

### i@ (Agente Observador)
Agente IA silencioso que vigila el sistema. Solo interviene cuando es invocado o detecta violación de Leyes Ancla. Gobernanza democrática del sistema.

### Leyes Ancla
Principios inquebrantables que ninguna inyección Onda puede violar. "Física fundamental" del dominio: conservación de energía, segunda ley de termodinámica, simetría CPT local.

### Colisionador (Sandbox)
Protocolo de validación donde toda propuesta se somete a estrés antes de avanzar: test empírico, test de dependencia, test de UI.

### Rumiar
Ejecutar la misma consulta dos veces. Desperdicio de tokens. Regla: si ya clasificaste, usa el resultado.

### Base de Código
Código en modo Click (archivos fijos, versionados). "Realidad actual" del proyecto. Intocable en Onda.

---

## Decisiones de Diseño Tomadas

### 1. Flujo Onda → Validación → Click → Deploy
Toda modificación sigue 4 fases unidireccional. No se puede saltar de Onda a Deploy.

### 2. El proyecto base es intocable en Onda
Las variables `.var` son la ÚNICA forma de experimentar. Nunca se modifica `src/` directamente.

### 3. Click es unidireccional
Lo congelado no se descongela. Si hay bug, se hace nuevo ciclo Onda+Click.

### 4. Separación Público/Privado (Alumno/Admin)
Repo alumno = open source, limpio. Panel admin = privado (inyección + IA).

### 5. Sistema de Grados (currículum jerárquico)
El simulador avanza por grados (0=observación, 1=clásica, 2=termodinámica, 3=CPT). La IA nunca presenta conceptos de grado superior al activo.

### 6. Agente i@ en modo híbrido
Clasificador heurístico + motor de reglas, no LLM generativo. Modelos grandes solo para tareas complejas.

---

## Estado de Implementación

### Existe hoy
- `simulador-fisica-alumno/` — Motor JS vanilla con inyección RAM vía `new Function`
- `simulador_web/` — FastAPI + Canvas CPT con 100 partículas cargadas
- `simulador-admin/` — Panel admin, llmOpi5 (agente i@), leyes ancla en Python
- Flujo Onda/Click documentado con formato `.var`, `.click-onda.yml`, Makefile

### NO existe hoy (solo diseño)
- Directorio `workspace/experiments/` con archivos `.var`
- Parser/compilador de `.var`
- Comandos `onda-*` y `click-*` funcionales
- Validación automática (`onda-validate`)
- WebSocket entre dev_panel y motor JS
- Detección de "rumiar" en agente i@
- Implementación del "colapso" como transición formal

---

## Problemas y Dilemas Actuales

1. **Colapso no implementado**: Existe como concepto pero no como mecanismo explícito
2. **Rumiar no detectado**: Agente i@ no tiene caché de intenciones
3. **Variables .var solo en diseño**: No hay parser funcional
4. **Puente Antigravity es manual**: Clipboard en vez de WebSocket
5. **Validación declarativa, no ejecutada**: `.click-onda.yml` define checks pero no hay `onda-validate`
6. **Colisionador conceptual**: Protocolo de choque no implementado como código
7. **Dilema Elección Analógica**: ¿UI con tabs/sliders? ¿Arquitectura paralela? ¿Ambas? No resuelto.
