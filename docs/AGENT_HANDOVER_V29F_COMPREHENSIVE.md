# 🧠 AGENT HANDOVER: CPT Simulator v5 (V2.9F Comprehensive Guide)

**Date of Handover:** May 17, 2026
**Target Audience:** Next AI Agent taking over development.

¡Hola, Agente! Este documento está diseñado para darte **todo el contexto desde cero** sobre qué es este proyecto, cómo funciona internamente, dónde están los archivos clave y en qué punto exacto debes continuar trabajando. Lee esto y estarás listo para codificar sin tener que escanear todo el repositorio.

---

## 1. 🚀 ¿Qué es el CPT Simulator v5?

El objetivo principal del proyecto es crear un **Subrogado Neuronal (Neural Surrogate)** para simulaciones de circuitos eléctricos de corriente continua (DC). 

**El problema:** Los simuladores tradicionales (oráculos como SPICE) resuelven grandes sistemas de matrices (Modified Nodal Analysis - MNA). Esto es lento y computacionalmente costoso $O(N^3)$.
**La solución:** Entrenar una Graph Neural Network (GNN) que tome un circuito (nodos y componentes) y prediga casi instantáneamente los voltajes de los nodos.

Sin embargo, las redes neuronales a menudo violan las leyes físicas elementales. Por lo tanto, nuestra arquitectura impone restricciones físicas: **Ley de Corrientes de Kirchhoff (KCL)**, **Ley de Voltajes de Kirchhoff (KVL)** y **Conservación de la Energía (Power)**.

---

## 2. 🧬 La Evolución hasta V2.9F: El Cambio de Paradigma

Hemos pasado por muchas fases de ablación y ajuste. Lo más importante que debes saber de la fase actual (**V2.9F**) es que **hemos cambiado nuestra filosofía científica**.

Ya NO estamos intentando que la GNN prediga voltajes perfectos por sí sola (regresión pura). Ahora concebimos el sistema como un **Solver Iterativo Híbrido (Warm-Start Solver)**.
1. La **GNN** hace una predicción inicial rápida.
2. Una capa determinista de **Proyección Física** ajusta esos voltajes iterativamente para forzar el cumplimiento de KCL y KVL.
3. Descubrimiento V2.9F: Las iteraciones locales (tipo Jacobi/SOR) son muy lentas para propagar correcciones en grafos muy largos (ej. cadenas radiales). Para solucionar esto, inventamos el **True Global Virtual Node**, un nodo virtual en la capa de proyección que promedia el error global y lo redistribuye instantáneamente, reduciendo el "diámetro de comunicación" del grafo.

---

## 3. 📂 Arquitectura del Código y Rutas Clave (Dónde está qué)

El repositorio está altamente modularizado. Aquí tienes el mapa mental exacto:

### 🧠 Modelos Core & Grafo
*   **`backend/circuits/models.py`**: Define las clases puras de Python (`Circuit`, `Resistor`, `VoltageSource`, `CurrentSource`). Los nodos de tierra siempre se normalizan a `"0"`.
*   **`backend/circuits/graph_dataset.py`**: Convierte los circuitos en grafos de PyTorch Geometric (`CircuitGraph`). Extrae dimensiones de nodos/edges dinámicamente y aplica normalización logarítmica a las resistencias extremas.
*   **`backend/circuits/dc_solver.py`**: El "Oráculo" tradicional. Resuelve los circuitos usando ecuaciones matemáticas exactas para generar el "Ground Truth" (los voltajes reales a aprender).

### 🎓 Entrenamiento y Currículo (V2.9E)
*   **`scripts/train_circuit_gnn.py`**: El script principal de entrenamiento. Instancia la `EdgeAwareCircuitGNN`.
*   **`backend/circuits/topology_curriculum.py`**: Clasifica los circuitos por dificultad (Trivial, Simple, Medium, Dense). El entrenamiento empieza con circuitos fáciles y desbloquea los difíciles progresivamente.
*   **`backend/circuits/losses.py`** & **`physics_loss.py`**: Definen las funciones de pérdida que penalizan a la red neuronal si sus predicciones violan KCL o KVL.

### 🔬 Proyección Física y Solver (Lo nuevo en V2.9F)
*   **`backend/circuits/physics_projection.py`**: **CRÍTICO**. Contiene la lógica del solver iterativo (tipo Jacobi) post-GNN y la implementación del `VirtualNodeProjection` que soluciona los cuellos de botella en cadenas radiales.
*   **`backend/circuits/warmstart_eval.py`**: Un experimento científico que prueba que usar las predicciones de la GNN reduce drásticamente el número de iteraciones necesarias para que un oráculo tradicional converja.

### 🧪 Evaluación y Diagnósticos
*   **`scripts/run_circuit_arena.py`**: El "Arena" de evaluación. Compara modelos base contra la GNN en métricas In-Distribution (IID) y Out-Of-Distribution (OOD). Produce reportes segmentados por familia topológica.
*   **`backend/circuits/failure_analysis.py`** & **`structural_failure_analysis.py`**: Diagnostican exactamente por qué falló una predicción (ej. `cycle_drift_failure`, `dense_mesh_leakage`).
*   **`backend/circuits/ood_stress_suite.py`**: Generadores deterministas de circuitos pesadillescos (mallas masivas, escaleras larguísimas) para estresar los límites de la red.

---

## 4. 🛠️ Comandos Esenciales de Trabajo

Como agente, querrás validar constantemente que tus cambios no rompen la física ni la lógica implementada. Todo está bajo cobertura de tests:

**Para correr toda la suite de pruebas V2.9E y V2.9F (Lo más importante):**
```bash
pytest tests/test_v29e_*.py tests/test_v29f_*.py -v
```

**Para correr el experimento de Warm-Start (reducción de iteraciones):**
```bash
python -m backend.circuits.warmstart_eval --steps 5 --perturbation 1.5
```

**Para correr la evaluación completa (Arena):**
```bash
python scripts/run_circuit_arena.py
```

---

## 5. 🎯 ¿Qué sigue? (Tu Misión / Próximos Pasos)

El entorno V2.9F está **completamente implementado, testeado y estable**. Tu misión, al tomar el control, debe orientarse a las siguientes tareas estratégicas:

1.  **Refinamiento de Pérdida de Newton (Newton-Physics Loss)**:
    Actualmente el proyector de física (`physics_projection.py`) es determinista y ocurre *después* de la predicción de la GNN. El siguiente paso es crear "Cabezales de Corrección Física" auto-correctivos (capas residuales) que intenten forzar KCL analíticamente **dentro del propio pipeline de entrenamiento** (durante el forward pass) de forma diferenciable.
2.  **Escalamiento de Receptividad Temporal para Escaleras Extremas**:
    Las redes en escalera muy largas (ej. $>100$ etapas) aún sufren atenuación de señal en el paso de mensajes de la GNN. Podrías investigar cómo introducir "nodos virtuales" (virtual nodes) **a nivel de grafo en la GNN** (no solo en la capa de proyección) o conexiones *highway* para que la información fluya extremo a extremo en menos capas.
3.  **Delegación Autónoma (Agente Hermes)**:
    El sistema ya cuenta con reportes taxonómicos de fallo súper ricos (`failure_analysis.py`). Se debe configurar al agente "Hermes" para que lea esos fallos en tiempo real y orqueste sesiones automáticas de re-entrenamiento enfocadas *únicamente* en las familias de circuitos donde el subrogado está fallando (Active Learning automatizado).

**¡Bienvenido a bordo! El estado actual es prístino y científicamente validado. Confía en las pruebas automatizadas y en este documento como tu única fuente de verdad funcional.**
