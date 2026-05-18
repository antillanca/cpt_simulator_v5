# CPT Cognitive Engine v2.9F (Neuro-Simbólico)

Motor de razonamiento neuro-simbólico premium. Aprende física de circuitos eléctricos, leyes de conservación y matemática desde cero vía simulaciones deterministas de oráculos + redes neuronales informadas por la física (PINNs) combinadas con solvers iterativos híbridos.

---

## 🚀 Estado de la Currícula AI y Evolución del Sistema

*   **Módulos de Currícula:** **43 de 43 confirmados (100% completado)** en sincronía perfecta con el motor estudiantil.
*   **Fase Actual:** `v2.9F` — Implementación del **True Global Virtual Node Projection** y transición hacia un **Solver Iterativo Híbrido (Warm-Start)**.
*   **Cambio de Paradigma:** Ya no entrenamos a la GNN para ser un regresor perfecto aislado. Ahora, la GNN actúa como un pre-condicionador ultrarrápido (warm-start) para una **Capa de Proyección Física Determinista**, reduciendo drásticamente las iteraciones necesarias para que un solver tradicional converja, evadiendo el problema del radio espectral en grafos de alto diámetro.

---

## ⚡ Arquitectura Core: Solver Híbrido & Proyección Física

El sistema ha evolucionado hacia un pipeline de resolución en dos etapas, garantizando velocidad neuronal con exactitud analítica:

1.  **GNN Surrogate (EdgeAwareCircuitGNN):** 
    Toma el grafo del circuito (con features topológicas dinámicas y resistencias log-normalizadas para evitar explosión de gradientes en rangos OOD) y predice un estado de voltaje inicial en una fracción de milisegundo.
2.  **Physics Projection (True Global Virtual Node):** 
    Aplica iteraciones tipo Jacobi/SOR sobre las predicciones de la GNN para forzar el cumplimiento de KCL y KVL. Para evitar el estancamiento de propagación en grafos largos (ej. cadenas radiales), inyecta un **Virtual Node** que agrega el residual de error global y lo redistribuye instantáneamente, transformando la propagación topológica de una "cadena" a una "estrella".

---

## 🧠 Currículo Topológico y Diagnóstico de Fallos

En lugar de entrenar a ciegas, el sistema implementa una currícula matemática (`CurriculumLevel`):
*   **Trivial:** Árboles, $\le 4$ nodos, 0 ciclos.
*   **Simple:** 1 Ciclo, $\le 6$ nodos.
*   **Medium:** 2-3 Ciclos, $\le 10$ nodos.
*   **Dense:** $> 3$ Ciclos, $> 10$ nodos. *(Nota Científica: Las topologías densas resultaron ser las más fáciles de predecir (MAE ~2.88V) debido a las extremas restricciones de los lazos paralelos que actúan como regularizadores naturales).*

### 🔬 Taxonomía de Fallos Físicos
El motor de análisis clasifica las anomalías directamente por su causa raíz topológica:
*   `cycle_drift_failure`: Desviación de KCL dentro de lazos cerrados.
*   `dense_mesh_leakage`: Fugas de señal en mallas de alta interconexión.
*   `bridge_node_instability`: Inestabilidad de propagación a través de cuellos de botella en estructuras de árbol.

---

## 📂 Estructura del Workspace (Componentes Clave V2.9F)

```
cpt_simulator_v5/
├── docs/
│   ├── AGENT_HANDOVER_V29F_COMPREHENSIVE.md  ← Contexto detallado de handover para IA
│   └── V29F_VIRTUAL_NODE_PROJECTION.md       ← Reporte científico oficial V2.9F
├── backend/
│   ├── circuits/
│   │   ├── dc_solver.py                ← Oráculo analítico de referencia
│   │   ├── graph_dataset.py            ← Conversión a grafos, features topológicas y normalización
│   │   ├── physics_projection.py       ← *[NUEVO]* Capa iterativa + True Global Virtual Node
│   │   ├── topology_curriculum.py      ← *[NUEVO]* Scheduler de complejidad estructural
│   │   ├── failure_analysis.py         ← Taxonomía de fallos topológicos
│   │   ├── warmstart_eval.py           ← *[NUEVO]* Evaluador del paradigma Híbrido (Solver Iters)
│   │   └── ood_stress_suite.py         ← *[NUEVO]* Generadores deterministas de mallas y escaleras
├── scripts/
│   ├── train_circuit_gnn.py            ← Pipeline de entrenamiento con ablación dinámica
│   └── run_circuit_arena.py            ← Comparador científico Arena por familias topológicas
└── tests/
    ├── test_v29f_virtual_projection.py ← Pruebas de reducción monotónica y nodo virtual
    └── test_v29f_warmstart.py          ← Pruebas de reducción de iteraciones oráculo
```

---

## 🎯 Rendimiento Actual (Full Unified Model V2.9E/F)

| Métrica | In-Dist MAE | KCL Max (A) | OOD KCL Max (A) | Iteraciones Solver |
| :--- | :---: | :---: | :---: | :---: |
| **GNN Pura (Baseline)** | 15.44 V | 0.275 | 4.82 | Lento |
| **GNN + Currículo + Topo** | 14.16 V | 0.163 | 1.10 | - |
| **Híbrido (Proyección + Nodo Virtual)**| **Virtualmente $0.0$** | **$< 1e-6$** | **$< 1e-6$** | **Ultra-Reducido** |

---

## 🛠️ Comandos de Uso Frecuente

Ejecutar la suite completa de pruebas unitarias físicas, de ablación y de proyector virtual (100% Passed):
```bash
pytest tests/test_v29e_*.py tests/test_v29f_*.py -v
```

Correr el experimento científico evaluador de Warm-Start (Comparación de iteraciones):
```bash
python -m backend.circuits.warmstart_eval --steps 5 --perturbation 1.5
```

Ejecutar la Evaluación de la Arena (Desglosada por familias topológicas):
```bash
python scripts/run_circuit_arena.py
```
