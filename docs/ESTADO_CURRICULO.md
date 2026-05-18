# Estado del Currículo — CPT Simulator v5

> **Resumen Ejecutivo**: El currículo del sistema CPT Simulator v5 se divide en dos fases principales. La **Fase 1 (Dominio Simbólico)** está 100% completa. Actualmente el sistema se encuentra cursando la **Fase 2 (Física Estructural y Topológica)**, dominando la resolución híbrida de grafos eléctricos.

---

## 🎓 FASE 1: Currículo de Dominio Simbólico (V2.5) — ✅ COMPLETADO

Esta fase consistió en enseñar al motor base (vía generación de código Lua validado) las leyes fundamentales del universo físico y matemático.

| Métrica | Valor | Estado |
|:---|:---:|:---:|
| Total de módulos teóricos | **43** | ✅ Asimilado |
| Confirmados por Invariantes | **43** (100%) | ✅ Verificado |
| Pendientes | **0** (0%) | - |

### Dominios Adquiridos (Resumen)
- **Matemáticas Clásicas**: Aritmética, Álgebra, Geometría Euclidiana, Trigonometría, Álgebra Lineal, Derivadas/Integrales numéricas.
- **Física Clásica**: Cinemática, Dinámica de Newton, Osciladores, Conservación de Energía (KE+PE).
- **Electromagnetismo**: Ley de Ohm, Fuerza de Lorentz, Ecuaciones de Maxwell.
- **Física Moderna**: Relatividad (Especial/General), Mecánica Cuántica (Función de onda, Doble Rendija), QFT, Expansión Cosmológica.
- **Sistemas**: Caos y Sistemas Dinámicos, Termodinámica y Entropía.

*Todos los módulos base residen en `backend/core_truth/` y fueron validados por el motor analítico sin intervención de LLMs.*

---

## ⚡ FASE 2: Currículo Topológico y Estructural (V2.9F) — 🔄 ACTIVO

Habiendo asimilado las leyes de Ohm y Kirchhoff teóricamente, el sistema ahora entrena un **Subrogado Neuronal (GNN)** acoplado a una capa de **Proyección Física** para resolver circuitos complejos en tiempo real. 

Para evitar el colapso del aprendizaje, los circuitos no se presentan aleatoriamente, sino a través de un riguroso `CurriculumLevel` topológico controlado por `topology_curriculum.py`.

### Estado de Niveles Topológicos

| Nivel | Nombre | Definición Estructural | Estado de Dominio GNN | Comportamiento del Solver Híbrido |
|:---:|:---|:---|:---:|:---|
| **L0** | **Trivial** | Estructuras de árbol, $\le 4$ nodos, $0$ ciclos. | ✅ **Dominado** | Convergencia instantánea. |
| **L1** | **Simple** | 1 Ciclo independiente, $\le 6$ nodos. | ✅ **Dominado** | El GNN acierta con alta precisión inicial. |
| **L2** | **Medium** | 2-3 Ciclos, $\le 10$ nodos. | 🟡 **Avanzado** | Estable, la GNN sirve de excelente pre-condicionador. |
| **L3** | **Dense** | $>3$ Ciclos, $>10$ nodos (Mallas complejas). | 🟢 **Excelente** | Contraintuitivamente, las densas restricciones del grafo regularizan la red, logrando MAEs bajísimos. |
| **L4** | **Extremo (OOD)**| Cadenas radiales de $+50$ nodos, resistores $1M\Omega$. | 🟡 **Activo** | El **True Global Virtual Node** implementado en V2.9F es crítico aquí para lograr convergencia matemática sorteando la atenuación espectral. |

### Hitos de la Fase 2 (V2.9F)
- [x] Transición de regresión pura a **Solver Iterativo Híbrido**.
- [x] Proyección Física determinista post-GNN.
- [x] Inyección del **Virtual Node** para reducir el diámetro de comunicación en redes largas.
- [x] Diagnóstico automático por Taxonomía de Fallos Estructurales.

---

## 🔮 FASE 3: Currículo Autónomo (Roadmap)

La próxima etapa buscará delegar la gestión del currículo topológico al agente supervisor (Hermes). 
- El agente analizará los resultados de la *Circuit Arena*.
- Identificará familias estructurales débiles (ej. Redes en Escalera).
- Generará automáticamente lotes sintéticos específicos para esa familia y los inyectará en el pipeline de entrenamiento (Active Learning).
