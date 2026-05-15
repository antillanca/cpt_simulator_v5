# Changelog de Intervenciones — Hermes

> **Resumen**: Registro de reparaciones automáticas realizadas por el agente Hermes para mantener el ciclo de entrenamiento activo.

---

## CPT v2.5

- Hermes pasa a ser un asistente de tooling, no una autoridad de merge
- No puede modificar `backend/core_truth/`, `backend/verifiers/` ni `backend/dsl/`
- Las acciones sensibles requieren aprobación humana explícita
- Uso principal: análisis de logs, sugerencias de parche, borradores de PR y diagnóstico
