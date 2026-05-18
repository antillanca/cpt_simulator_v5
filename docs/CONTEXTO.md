# Contexto — Glosario y Decisiones de Diseño (V2.9F)

> **Resumen**: Conceptos clave del ecosistema **CPT Simulator v5**. Define la terminología fundamental (Leyes Ancla, Oráculo, Subrogado Neuronal, Proyección Física) y documenta las decisiones de diseño arquitectónicas actuales.

---

## Glosario de Conceptos Clave

### Leyes Ancla (Invariantes Físicos)
Principios inquebrantables que ninguna red neuronal puede violar en su estado final. Constituyen la "Física fundamental" del dominio:
- **KCL (Kirchhoff's Current Law)**: Conservación de carga en cada nodo.
- **KVL (Kirchhoff's Voltage Law)**: Conservación de energía en ciclos cerrados.
- **Power Conservation**: Equilibrio entre potencia suministrada y disipada.

### Oráculo (Ground Truth)
El solver analítico tradicional (`dc_solver.py` usando MNA). Es matemáticamente perfecto pero lento ($O(N^3)$). Se usa exclusivamente para generar los datos de entrenamiento y validar la exactitud final del sistema. No se usa en inferencia rápida.

### Subrogado Neuronal (Neural Surrogate)
Una Red Neuronal de Grafos (`EdgeAwareCircuitGNN`) que aprende a imitar al oráculo. Proporciona predicciones ultrarrápidas ($<1$ ms) de los voltajes de un circuito. Por su naturaleza probabilística, sus salidas puras contienen violaciones a las Leyes Ancla.

### Proyección Física (Physics Projection)
Capa iterativa determinista (tipo Jacobi) que toma la predicción del subrogado neuronal y la ajusta para forzar matemáticamente el cumplimiento de KCL y KVL.

### True Global Virtual Node
Un nodo matemático inyectado únicamente durante la Proyección Física. Promedia el error residual de toda la red y lo redistribuye instantáneamente. Su propósito es romper el cuello de botella del diámetro del grafo en topologías largas (como cadenas radiales), permitiendo convergencia global rápida.

### Warm-Start Híbrido
El cambio de paradigma de V2.9F. Consiste en usar al Subrogado Neuronal no como la respuesta final, sino como el punto de partida ideal (pre-condicionador) para el solver de Proyección Física, reduciendo drásticamente las iteraciones necesarias para encontrar la solución perfecta.

### Hermes (Agente de Supervisión)
El agente IA que vigila el sistema. Su rol es analizar las métricas de rendimiento (ej. fallos en ciertas topologías) y orquestar re-entrenamientos autónomos. No modifica las Leyes Ancla.

---

## Decisiones de Diseño Tomadas

### 1. Separación Neuro-Simbólica
El sistema divide estrictamente la intuición (red neuronal) del razonamiento riguroso (proyección física determinista). La red adivina rápido, la proyección corrige matemáticamente.

### 2. Currículo Topológico
El entrenamiento no se hace alimentando circuitos aleatorios. Se sigue una progresión estructurada (`topology_curriculum.py`) desde lo más fácil (Trivial/Árboles) hasta lo más difícil (Mallas Densas), garantizando que la red neuronal adquiera "conceptos" incrementales.

### 3. Normalización Logarítmica
Para poder manejar resistencias que varían en rangos extremos (OOD, desde $0.1\Omega$ hasta $1M\Omega$), los features numéricos en los grafos se normalizan logarítmicamente. Esto evita la explosión de gradientes y la inestabilidad numérica.

### 4. Diagnóstico Basado en Causa Raíz
Cuando una inferencia falla, el error no se reporta solo como un número (MSE). El sistema analiza la topología y clasifica el error estructuralmente (`cycle_drift_failure`, `dense_mesh_leakage`, `bridge_node_instability`) facilitando la corrección algorítmica.

---

## Estado de Implementación (Fase V2.9F)

### Existe y es funcional
- `backend/circuits/dc_solver.py` — Oráculo MNA perfecto.
- `backend/neural/models/circuit_gnn.py` — Subrogado GNN (EdgeAwareCircuitGNN).
- `backend/circuits/physics_projection.py` — Corrector iterativo con el **True Global Virtual Node**.
- `backend/circuits/topology_curriculum.py` — Scheduler de dificultad topológica.
- `scripts/train_circuit_gnn.py` — Pipeline de entrenamiento PINN.
- `scripts/run_circuit_arena.py` — Evaluador científico segregado por familias de circuitos.

### Próximos Desafíos
1.  **Refinamiento Diferenciable**: Mover la corrección de Newton/Jacobi hacia adentro del grafo computacional de PyTorch durante el entrenamiento, para que la red neuronal aprenda a predecir derivadas exactas.
2.  **Autonomía Total**: Conectar el output taxonómico de fallos con el input de generación de Hermes para crear un bucle cerrado de aprendizaje activo infinito (Active Learning).
