# CPT v2.9F — Arquitectura Core: Solver Iterativo Híbrido Neuro-Simbólico

> **Resumen**: En la versión V2.9F, el sistema CPT Simulator ha evolucionado de un simple regresor neuronal a un **Solver Iterativo Híbrido**. La arquitectura combina la velocidad de pre-condicionamiento de las Redes Neuronales de Grafos (GNN) con la precisión absoluta de los solvers matemáticos tradicionales (Jacobi-style Physics Projection) mediante el innovador **True Global Virtual Node**.

---

## 🏗️ Pila Arquitectónica Híbrida (Neuro-Simbólica)

La arquitectura se divide en tres componentes principales que actúan en cascada para garantizar velocidad sin comprometer las leyes físicas.

### 1. El Oráculo Analítico (Ground Truth)
- **Componente**: `backend/circuits/dc_solver.py`
- **Rol**: La fuente absoluta de la verdad. Utiliza Análisis Nodal Modificado (MNA) matemático exacto para resolver los circuitos y generar las etiquetas de entrenamiento.
- **Característica**: $O(N^3)$, seguro, pero computacionalmente lento para grafos masivos.

### 2. El Subrogado Neuronal (Pre-condicionador)
- **Componente**: `backend/neural/models/circuit_gnn.py` (EdgeAwareCircuitGNN)
- **Rol**: Una Graph Neural Network (GNN) entrenada bajo restricciones físicas (PINN) que predice un estado de voltaje inicial en fracciones de milisegundo.
- **Característica**: En el paradigma V2.9F, esta red actúa como un **Warm-Start** hiper-rápido, no como la respuesta final. Maneja features topológicos dinámicos y resistencias log-normalizadas para evitar explosión de gradientes en OOD.

### 3. La Proyección Física (El "Corrector" Determinista)
- **Componente**: `backend/circuits/physics_projection.py`
- **Rol**: Una capa iterativa (estilo Jacobi) determinista que toma la predicción "sucia" de la GNN y la ajusta matemáticamente para forzar el cumplimiento estricto de las leyes de conservación de carga (KCL) y voltaje (KVL).
- **Innovación Core (V2.9F)**: **True Global Virtual Node**. Agrega y redistribuye el residual de error globalmente en cada paso de proyección, evadiendo el cuello de botella del radio espectral y permitiendo la convergencia instantánea en grafos de alto diámetro (ej. cadenas radiales y escaleras largas).

---

## 🧠 Flujo de Datos y Diagnóstico de Fallos

### 1. Currículo Topológico (`topology_curriculum.py`)
Los grafos de entrenamiento no se presentan aleatoriamente. Pasan por un orquestador matemático que los segmenta por dificultad:
- **Trivial**: Árboles cortos.
- **Simple**: Lazos únicos.
- **Medium**: Múltiples ciclos interconectados.
- **Dense**: Mallas complejas densamente conectadas (que, contraintuitivamente, convergen más rápido debido a sus propias restricciones topológicas).

### 2. Taxonomía Estructural (`failure_analysis.py`)
Cuando un error residual sobrevive a la proyección, el sistema no solo reporta el MSE, sino que clasifica la anomalía físicamente:
- `cycle_drift_failure` (Falla de balance KCL en ciclos cerrados).
- `dense_mesh_leakage` (Fuga de gradientes en alta interconectividad).
- `bridge_node_instability` (Drift de cuellos de botella en árboles).

---

## 🛠️ Entorno de Desarrollo (Tooling)

*   **Entrenamiento**: `scripts/train_circuit_gnn.py` (Maneja las iteraciones, el scheduler del currículo y las funciones de pérdida PINN penalizadas por KCL/KVL/Power).
*   **Evaluación**: `scripts/run_circuit_arena.py` (Benchmarking científico, segmenta el MAE y los slopes de convergencia según la familia topológica).
*   **Pruebas (Tests)**: Suite completa bajo `pytest` en `tests/test_v29f_*.py` y `test_v29e_*.py` verificando la estabilidad numérica OOD, monotonía residual y determinismo.

---

## 🚀 Roadmap Arquitectónico (Próximos Pasos Post-V2.9F)

1.  **Refinamiento de Newton-Loss**: Embeber cabezales auto-correctivos físicos directamente en las capas residuales de la GNN para forzar el cumplimiento analítico *durante* el forward-pass de entrenamiento.
2.  **Escalamiento de Receptividad Temporal**: Integrar nodos virtuales globales pero a nivel de grafo de la propia GNN (no solo en la capa de proyección) para manejar redes en escalera de $>100$ etapas.
3.  **Active Learning Autónomo (Hermes)**: Acoplar la salida del `failure_analysis.py` con el pipeline de generación, para que el sistema automáticamente cree y entrene más circuitos de las familias específicas en las que detecte debilidad, cerrando el bucle de auto-aprendizaje.
