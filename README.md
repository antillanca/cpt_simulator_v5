# CPT Cognitive Engine v2.9D (Neuro-Simbólico)

Motor de razonamiento neuro-simbólico premium. Aprende física de circuitos eléctricos, leyes de conservación y matemática desde cero vía simulaciones deterministas de oráculos + redes neuronales informadas por la física (PINNs).

---

## 🚀 Estado de la Currícula AI (100% Completado)

*   **Módulos de Currícula:** **43 de 43 confirmados (100% completado)** en sincronía perfecta con el motor estudiantil.
*   **Fase Actual:** `v2.9D` — Entrenamiento de alta fidelidad del surrogate GNN en el dataset completo (`train_10k.jsonl`) con endurecimiento de invariantes físicos y empaquetamiento kernel-ready para Kaggle.
*   **Aceleración Científica:** Sustitución de resolvedores matriciales tradicionales por un surrogate basado en **Graph Neural Networks (GNN)** con pérdidas físicas acopladas directamente en PyTorch.

---

## ⚡ GNN Surrogate & Pérdida Informada por la Física (PINN)

El modelo de circuitos implementa una red neuronal de paso de mensajes condicionada por bordes (**EdgeAwareCircuitGNN**) que respeta estrictamente los principios de conservación física mediante la clase `PhysicsInformedLoss`:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}(V) + \lambda_{\text{KCL}} \mathcal{L}_{\text{KCL}} + \lambda_{\text{KVL}} \mathcal{L}_{\text{KVL}} + \lambda_{\text{Power}} \mathcal{L}_{\text{Power}}$$

*   **Leyes de Corrientes de Kirchhoff (KCL):** Penalización estricta de la divergencia de corriente neta en cada nodo, calculada a partir de las admitancias de los componentes y fuentes de corriente.
*   **Leyes de Voltajes de Kirchhoff (KVL):** Penalización de la suma de caídas de tensión a lo largo de ciclos fundamentales independientes generados deterministamente mediante matrices de ciclos de grafos.
*   **Balance de Potencia (Power Invariant):** Penalización de la discrepancia entre la potencia total suministrada por las fuentes de tensión/corriente y la potencia disipada por los resistores.

---

## 📂 Estructura del Workspace

```
cpt_simulator_v5/
├── HANDOVER.md              ← contexto completo de desarrollo
├── README.md                ← esta guía rápida
├── backend/
│   ├── core_truth/          ← currículo + base de conocimiento (PROTEGIDO)
│   ├── circuits/            ← resolvedor DC, grafos, pérdida PINN y taxonomía de fallas
│   │   ├── dc_solver.py      ← oráculo analítico de referencia (Norton/Nodal)
│   │   ├── graph_dataset.py  ← conversión de netlists a grafos PyG deterministas
│   │   ├── physics_loss.py   ← cálculo de KCL, KVL en lazo y balance de potencia
│   │   └── failure_analysis.py ← taxonomía y clasificación de anomalías físicas
│   └── neural/
│       ├── models/
│       │   └── circuit_gnn.py ← arquitectura EdgeAwareCircuitGNN (<250k params)
│       └── training_snapshot.py ← firmas e instrumentación determinista de checkpoints
├── configs/
│   └── training/
│       └── kaggle_v29d.yaml ← hiperparámetros de entrenamiento de alta fidelidad
├── scripts/
│   ├── train_circuit_gnn.py  ← pipeline determinista de entrenamiento
│   ├── run_circuit_arena.py  ← comparador científico Arena (Oracle vs GNN)
│   ├── analyze_v29c_failures.py ← taxonomía de anomalías OOD
│   ├── generate_v29d_report.py  ← generador de reportes científicos comparativos
│   └── kaggle_prepare_v29d.py  ← empaquetador del bundle de exportación reproducible
└── tests/
    └── test_v29d_physics_loss.py ← suite de regresión y determinismo físico (16 passed)
```

---

## 🎯 Objetivos de Éxito Científico (v2.9D)

| Métrica | Meta | v2.9D Subset | v2.9D Full (En Proceso) |
| :--- | :--- | :--- | :--- |
| **IID MAE** | `< 5.0 V` | `6.77 V` | *Próximamente* |
| **OOD MAE** | `< 50.0 V` | `144.88 V` | *Próximamente* |
| **KCL Max Violation** | `< 1e-3` | `3.3e+07` | *Próximamente* |
| **KVL Max Violation** | `< 1e-3` | `5.58 V` | *Próximamente* |
| **Parámetros GNN** | `< 250,000` | **82,113** (Cumplido) | **82,113** (Cumplido) |

---

## 🛠️ Comandos de Validación Rápida

Ejecutar la suite completa de pruebas unitarias físicas y deterministas:
```bash
pytest tests/test_v29d_physics_loss.py -v
```

Generar el reporte de comparación y empaquetar el bundle reproducible:
```bash
python scripts/kaggle_prepare_v29d.py --config configs/training/kaggle_v29d.yaml
```
