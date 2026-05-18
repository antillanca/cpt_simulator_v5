# CPT Cognitive Engine v2.9F — Master Handover

> **Current Status (2026-05-18)**: 🚀 **MILESTONE ACHIEVED: V2.9F True Global Virtual Node Projection.**
> El sistema ha transicionado de un regresor aislado a un **Solver Iterativo Híbrido** neuro-simbólico.

---

## 🏗️ Architecture: The Hybrid Neuro-Symbolic Stack

CPT es un sistema de inteligencia estructurado en capas que ahora fusiona la velocidad de las Graph Neural Networks (GNN) con la exactitud analítica de los solvers iterativos.

### 1. Layer 0: Core Truth (Oráculo Analítico)
- **Path**: `backend/circuits/dc_solver.py`
- **Logic**: Oráculo tradicional basado en Modified Nodal Analysis (MNA). Resuelve los circuitos usando ecuaciones matemáticas exactas para generar el *Ground Truth*.

### 2. Layer 1: GNN Surrogate (Pre-condicionador)
- **Path**: `scripts/train_circuit_gnn.py` & `backend/neural/models/circuit_gnn.py`
- **Logic**: Una red neuronal informada por la física (PINN) que predice un estado de voltaje inicial en fracciones de milisegundo. En V2.9F, esta red actúa como un **Warm-Start** para el solver físico.

### 3. Layer 2: Physics Projection (El "Corrector")
- **Path**: `backend/circuits/physics_projection.py`
- **Logic**: Capa iterativa (estilo Jacobi) que toma la predicción de la GNN y la ajusta para forzar el cumplimiento estricto de las leyes KCL y KVL.
- **Innovación V2.9F**: Implementa el **True Global Virtual Node**, un nodo virtual que agrega y redistribuye el error global instantáneamente, solucionando los cuellos de botella de convergencia en grafos muy largos (cadenas radiales).

---

## 📈 Progression & Metrics (V2.9F)

- **Currículo Topológico:** Los circuitos se entrenan en orden de dificultad (Trivial, Simple, Medium, Dense).
- **Taxonomía de Fallos:** El motor ahora diagnostica anomalías físicas (`cycle_drift_failure`, `dense_mesh_leakage`, `bridge_node_instability`).
- **OOD Stress Suite:** Generadores deterministas de mallas masivas y redes en escalera para estresar los límites de la red (`ood_stress_suite.py`).

| Métrica | Híbrido (Proyección + Nodo Virtual) |
|:---|:---:|
| **In-Dist MAE** | Virtualmente $0.0$ |
| **KCL Max (A)** | $< 1e-6$ |
| **Iteraciones Solver** | Ultra-Reducidas |

---

## 🔮 Next Steps: El Camino a Seguir

Para los agentes entrantes, el trabajo debe centrarse en:

1. **Refinamiento de Pérdida de Newton (Newton-Physics Loss)**: 
   Integrar cabezales de corrección física auto-correctivos dentro de las capas residuales durante el entrenamiento para forzar el cumplimiento de KCL de forma analítica en el forward pass.
2. **Escalamiento de Receptividad Temporal**:
   Investigar cómo usar nodos virtuales a nivel de grafo en la GNN para evitar la atenuación de señal en redes en escalera extremadamente largas (>$100$ etapas).
3. **Delegación Autónoma (Agente Hermes)**:
   Configurar a Hermes para monitorear las métricas de la Arena por familia topológica y orquestar sesiones de re-entrenamiento enfocadas en las familias con menor precisión.

---

## 📂 Documentos de Contexto Integral

Para un entendimiento técnico profundo desde cero hasta la fase actual, **DEBES** leer:
- 📖 [Guía Comprensiva de Handover para IA (V2.9F)](docs/AGENT_HANDOVER_V29F_COMPREHENSIVE.md)
- 🔬 [Reporte Científico Oficial V2.9F (Virtual Node)](docs/V29F_VIRTUAL_NODE_PROJECTION.md)
- 📊 [Reporte Científico V2.9E (Ablación y Topología)](docs/V29E_TOPOLOGY_AWARE_SURROGATE.md)
