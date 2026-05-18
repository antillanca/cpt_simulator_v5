# Flujo de Desarrollo — Click y Onda

> **Resumen**: Dos modos de trabajo opuestos y complementarios. **Onda** = código experimental en variables, no toca el proyecto base. **Click** = congelamiento en archivos fijos, genera artefacto inmutable. El flujo es: Onda → Validación → Click → Deploy. Click es unidireccional.

---

## Concepto

| Modo | Metáfora | Qué hace | Protege |
|:---|:---|:---|:---|
| **Onda** | Desarrollo fluido | Código vive en variables/plantillas, se inyecta dinámicamente | El proyecto base queda intacto |
| **Click** | Compilación instantánea | Congela código en archivos fijos, genera artefacto inmutable | Reproducibilidad y trazabilidad |

---

## Fases del Flujo

### 1. ONDA — Desarrollo Rápido

```
[variable/template] → [entorno pruebas Docker] → [validación en vivo]
```

- Código nuevo se almacena en **variables** (archivos `.var` o `workspace.yml`)
- El entorno monta variables como volumen o las inyecta via WebSocket
- Cambios se reflejan al instante sin reconstruir imagen
- Si algo se rompe, se descarta la variable sin tocar el proyecto base

**Regla**: Nada en modo Onda se considera estable.

### 2. VALIDACIÓN — Gate de Calidad

```
[codigo en variable] → [tests automáticos] → [checks manuales] → [aprobación]
```

- Tests unitarios y de integración corren contra las variables
- Si pasa: avanza a Click. Si falla: vuelve a Onda.

**Regla**: Sin validación no hay Click.

### 3. CLICK — Congelamiento y Despliegue

```
[variables aprobadas] → [compilación a código fijo] → [artefacto inmutable] → [deploy]
```

- Variables se "compilan": contenido se escribe en archivos definitivos
- Se genera artefacto inmutable (Docker image taggeada, commit con tag semántico)
- Deploy desde el artefacto, nunca desde variables

**Regla**: Click es unidireccional. Lo congelado no se descongela — se itera.

---

## Diagrama del Ciclo

```
  +--------+     +------------+     +-------+     +--------+
  |  ONDA  | --> | VALIDACION | --> | CLICK | --> | DEPLOY |
  | (var)  |     | (tests)    |     | (fijo)|     | (prod) |
  +--------+     +------------+     +-------+     +--------+
      ^                                  |
      |          fallo en validacion     |
      +----------------------------------+
      ^                                  |
      |         nuevo feature/fix        |
      +----------------------------------+
```

---

## Estructura de Archivos

```
proyecto/
├── workspace/              # ZONA ONDA — no se commitea
│   ├── experiments/        # Variables con código en prueba
│   │   ├── feature-x.var
│   │   └── fix-y.var
│   ├── workspace.yml       # Índice de variables activas
│   └── sandbox/            # Entorno Docker de desarrollo
├── src/                    # ZONA CLICK — código fijo, versionado
├── Makefile                # Comandos click-y-onda
└── .click-onda.yml         # Configuración del flujo
```

---

## Formato de Variable (.var)

```yaml
# workspace/experiments/feature-x.var
name: feature-x
target: src/app/Http/Controllers/SaleController.php
mode: replace-block
marker: // @var:feature-x
status: testing
created: 2026-04-27
---

public function calculateTotal($items)
{
    return collect($items)->sum(fn($i) => $i['price'] * $i['qty']);
}
```

Campos: `name`, `target`, `mode` (replace-block | append | new-file), `marker`, `status` (testing | validated | frozen)

---

## Comandos (Makefile)

```makefile
# MODO ONDA
onda-up:       # Levantar entorno de desarrollo
onda-inject:   # Inyectar variable .var al entorno en vivo
onda-list:     # Listar variables activas
onda-diff:     # Ver diff entre variable y archivo destino
onda-reset:    # Descartar todas las variables

# VALIDACION
onda-test:     # Correr tests contra variables activas
onda-validate: # Validar checks requeridos

# MODO CLICK
click-freeze:  # Congelar variables validadas en código fijo
click-build:   # Construir artefacto inmutable
click-tag:     # Crear tag semántico
click-deploy:  # Desplegar artefacto

# FLUJO COMPLETO
release:       # onda-validate + click-freeze + click-build + click-tag + click-deploy
```

---

## Configuración por Proyecto (.click-onda.yml)


### Simulador CPT
```yaml
project: simulador-cpt
stack: python-fastapi-canvas
onda:
  dev_command: docker compose -f docker-compose.dev.yml up
  dev_inject: true
click:
  build_command: make release
  deploy_target: vps
```

---

## Reglas de Oro

1. El proyecto base es intocable en Onda
2. Sin validación no hay Click
3. Click es unidireccional
4. Cada variable tiene un único target
5. El workspace/ nunca se mergea a main
6. El .click-onda.yml es el contrato

---

## Integración con Agentes de IA

- **En Onda**: crean/editan archivos `.var`, no tocan `src/`
- **En Validación**: ejecutan `onda-test` y verifican checks
- **En Click**: ejecutan `click-freeze` + `click-build`
- **Nunca**: se saltan validación para hacer Click directo
