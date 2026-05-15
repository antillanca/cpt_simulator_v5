# Guía de Contexto: CPT Simulator v5 (Para el Agente Creador de Currículum)

¡Hola, colega IA! Tu misión es ayudar al usuario a diseñar un **Plan de Estudios (Syllabus)** en formato JSON para el **CPT Simulator v5**. 

Para que puedas generar las materias y su orden correctamente, necesitas entender la filosofía del sistema. No somos un tutor tradicional; somos una Inteligencia Artificial General en miniatura que aprende las leyes del universo paso a paso en un **Sandbox estricto (Lua)**.

---

## 1. Terminología Fundamental

Para evitar ambigüedades, definamos los tres conceptos clave del sistema:

1.  **Capa (Layer)**: Nivel conceptual abstracto (del 0 al 34). Representa un salto evolutivo en la comprensión del universo (ej. pasar de "Aritmética" a "Álgebra").
2.  **Nodo (Node)**: Un bloque de conocimiento reusable. Una capa puede tener múltiples nodos. (Ej. dentro de la Capa Álgebra, un nodo es "Variables" y otro es "Ecuaciones").
3.  **Ejercicio (Exercise)**: La prueba de validación concreta que el modelo debe pasar para dominar un Nodo. Consiste en escribir código Lua para alcanzar un estado físico exacto.

---

## 2. El Plan Maestro (Macro-Estructura MoE)

El objetivo final del simulador no es solo aprender física, sino desarrollar una Inteligencia Artificial General (AGI) a través de un currículo evolutivo. El sistema consolidará su conocimiento siguiendo esta progresión fundacional:

1. **Realidad**: Comprensión del sandbox y las leyes fundamentales de existencia.
2. **Física**: Dinámica, cinemática, energía y comportamiento del universo simulado.
3. **Matemática**: Abstracción de las interacciones físicas en ecuaciones puras.
4. **Representación**: Creación de modelos y estructuras de datos para estados complejos.
5. **Símbolos**: Asignación de tokens abstractos a representaciones lógicas.
6. **Inglés**: Procesamiento sintáctico y gramatical usando lógica simbólica.
7. **Traducción**: Mapeo conceptual entre el lenguaje y la lógica física/matemática.
8. **Abstracción Avanzada**: Razonamiento puro de alto nivel desvinculado de la simulación básica.

---

## 3. Arquitectura de Capas de Aprendizaje

**El modelo no aprende “temas aislados”.** Aprende representaciones del universo mediante reglas matemáticas progresivas y **filtros neuronales duales** (Lógica y Acción). 
El flujo de cognición es estricto y ahora jerárquico: `Currículo` → `Tutor (Genera Prompt)` → `Estudiante (Genera Lua)` → `Sandbox` → `Validación`.
El Tutor actúa como el arquitecto pedagógico, definiendo **qué** aprender, mientras que el Estudiante se enfoca en **cómo** resolverlo mediante código.

### Regla Estructural Inviolable
El sistema **nunca** debe intentar aprender un Nodo si los Nodos previos (sus dependencias) no están consolidados. Cada nuevo conocimiento se construye inyectando el código Lua aprendido en los Nodos anteriores.

### Las 35 Capas del Universo (Syllabus Base)
Tu propuesta de Ejercicios debe mapearse estrictamente a estas capas:
*   **0. Existencia y Diferencia**: Distinguir `A != B`. Detectar variación y cambio de estado.
*   **1. Conteo Discreto**: Incremento, decremento, cantidad (`1+1=2`).
*   **2. Operaciones Fundamentales**: Suma, resta, multiplicación, división (`a+b=c`).
*   **3. Representación Numérica**: Enteros, negativos, fracciones, potencias, raíces.
*   **4. Relaciones y Proporciones**: Razón, escala, porcentaje. *Si A aumenta, B cambia.*
*   **5. Variables y Álgebra**: Abstracción, incógnitas, inferencia algebraica.
*   **6. Funciones**: Transformación determinística. Entrada → Proceso → Salida.
*   **7. Geometría**: Punto, línea, plano, distancia, orientación.
*   **8. Vectores**: Movimiento dirigido. Magnitud y dirección.
*   **9. Trigonometría**: Seno, coseno, rotaciones, oscilaciones.
*   **10. Tiempo y Cinemática**: Posición, velocidad, aceleración. *El cambio temporal.*
*   **11. Dinámica**: Relacionar movimiento con causas. Fuerza, masa, inercia (`F = m * a`).
*   **12. Energía**: Conservación. Cinética, potencial, trabajo.
*   **13. Oscilaciones y Ondas**: Comportamiento periódico, frecuencia, amplitud.
*   **14. Electricidad y Magnetismo**: Interacción de campos, cargas, inducción.
*   **15. Termodinámica**: Sistemas estadísticos, calor, entropía, equilibrio.
*   **16. Probabilidad y Estadística**: Modelar incertidumbre, error.
*   **17. Modelado Avanzado**: Sistemas dinámicos complejos.
*   **18. Cálculo Diferencial**: Tasas de cambio continuo.
*   **19. Cálculo Integral**: Acumulación, áreas bajo la curva.
*   **20. Ecuaciones Diferenciales**: Modelado de cambio a lo largo del tiempo.
*   **21. Álgebra Lineal**: Transformaciones de matrices.
*   **22. Análisis Numérico**: Métodos computacionales como Euler/Runge-Kutta.
*   **23. Mecánica Lagrangiana**: Derivación de movimiento a través de la energía.
*   **24. Mecánica Hamiltoniana**: Estado dinámico y espacio de fase.
*   **25. Electromagnetismo Avanzado**: Ecuaciones de Maxwell, ondas electromagnéticas.
*   **26. Relatividad Especial**: Dilatación del tiempo, contracción de longitud.
*   **27. Relatividad General**: Geodésicas, intervalo de espaciotiempo.
*   **28. Mecánica Cuántica**: Función de onda, probabilidad de estados.
*   **29. Teoría Cuántica de Campos**: Osciladores armónicos cuánticos, excitación de vacío.
*   **30. Cosmología**: Expansión del universo, Ley de Hubble.
*   **31. Teoría del Caos**: Dinámicas no lineales, atractores.
*   **32. Frontera del Conocimiento**: Modelos teóricos abstractos y unidades naturales.
*   **33. [Reservado]**: Transición topológica.
*   **34. Lógica Cuántica Aplicada**: Experimento de doble rendija y superposición.

---

## 3. Esquema de Salida Requerido (JSON)

Para que el simulador pueda ingerir tu propuesta, debes generar una lista plana de Nodos/Ejercicios. Cada ítem debe seguir **estrictamente** este esquema JSON.

> [!IMPORTANT]
> **Consistencia de Idioma:** Dado que el agente local procesará este archivo, los campos `id`, `title` y `objective` **DEBEN estar en Inglés**.

```json
[
  {
    "id": "kinematics_constant_velocity",
    "title": "Constant Velocity Motion",
    "layer": 10,
    "prerequisites": ["math_vectors_addition", "math_algebra_variables"],
    "objective": "Move the particle to the right at a constant horizontal speed of 5.0. No vertical movement.",
    "target_state": {"vx": 5.0, "vy": 0.0},
    "tolerance": 0.5,
    "simulation_frames": 1,
    "order": 100
  }
]
```

### Reglas de Validación de los Campos:
1.  **`id`**: Cadena única, en inglés, en formato *snake_case*.
2.  **`layer`**: Número entero (0 a 34) correspondiente a la lista filosófica de arriba.
3.  **`prerequisites`**: Lista de `id`s de nodos anteriores indispensables para este ejercicio. ¡Crucial para armar el árbol de dependencias!
4.  **`objective`**: Instrucción clara y técnica en **inglés**. La IA del simulador es monolingüe. No uses definiciones abstractas ("Entender la velocidad"); usa comandos accionables ("Set the horizontal velocity of the particle to X").
5.  **`target_state`**: El estado matemático a evaluar. El sistema solo puede evaluar 4 variables de la partícula: `x`, `y`, `vx`, `vy`. Puedes incluir una, varias o todas.
6.  **`tolerance`**: El margen de error permitido. Usualmente `0.5` a `2.0` dependiendo de la precisión esperada.
7.  **`simulation_frames`**: Cuántos ciclos (ticks) correrá la simulación antes de evaluar el `target_state`. Usa `1` para comprobaciones instantáneas de álgebra/velocidad, y un número mayor (ej. `60`) si deseas que el objetivo sea alcanzar una posición `x, y` tras cierto tiempo.
8.  **`order`**: Un identificador de secuencia para ordenar la lista plana (ej. sumando de a 10: 10, 20, 30...).

¡Procede a generar los nodos respetando absolutamente esta filosofía y formato!
