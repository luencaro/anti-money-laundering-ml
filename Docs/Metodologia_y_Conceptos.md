# Metodología y Conceptos Previos

Antes de adentrarnos en la evaluación de los modelos benchmark, es fundamental establecer el marco de trabajo, las métricas elegidas y las técnicas empleadas para abordar tanto el desbalanceo de los datos como la optimización de los hiperparámetros. En problemas de detección de anomalías o lavado de dinero, las clases suelen estar fuertemente desbalanceadas, lo que requiere un tratamiento especial.

---

## Métricas de Evaluación

En datasets donde la clase positiva (fraude/lavado) representa una fracción muy pequeña del total, métricas tradicionales como la exactitud (*Accuracy*) pueden resultar engañosas. Un modelo que clasifique todo como la clase mayoritaria obtendría una puntuación alta sin detectar un solo caso de fraude. Para entender las métricas que sí son útiles, conviene partir de la **Matriz de Confusión**:

| | Predicho Positivo | Predicho Negativo |
|---|---|---|
| **Real Positivo** | Verdadero Positivo (TP) | Falso Negativo (FN) |
| **Real Negativo** | Falso Positivo (FP) | Verdadero Negativo (TN) |

### F1-Score de la Clase Minoritaria (Métrica Principal)

El **F1-Score** es la media armónica entre la Precisión y la Exhaustividad (*Recall*), equilibrando el costo de generar falsas alarmas (baja Precisión) con el de dejar escapar fraudes reales (bajo Recall):

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

Donde:
- **Precision** $= \frac{TP}{TP + FP}$: fracción de alertas que son fraude real.
- **Recall** $= \frac{TP}{TP + FN}$: fracción de fraudes reales que se detectaron.

La media armónica penaliza con fuerza cuando cualquiera de los dos componentes es muy bajo, obligando al modelo a mantener un desempeño razonable en ambas dimensiones a la vez.

#### Justificación en el contexto de lavado de dinero

Esta elección de métrica no es arbitraria. Egressy et al. (2024), en su trabajo *"Provably Powerful Graph Neural Networks for Directed Multigraphs"* presentado en **AAAI 2024**, justifican explícitamente que, dado el fuerte desbalance de sus datasets AML (ratios de ilícitos de apenas 0.05%–0.11%), la exactitud y otras métricas populares resultan inadecuadas, adoptando el F1 de la clase minoritaria como métrica de referencia por ser la que bancos y reguladores utilizan en escenarios reales. Dado que uno de los objetivos de este benchmark es comparar nuestros modelos con los resultados reportados en dicho trabajo —donde los mejores modelos alcanzan F1-scores de entre 22% y 68% según el dataset— adoptamos la misma métrica para situarnos en el mismo marco de referencia.

### PR-AUC (Area Under the Precision-Recall Curve)

El **PR-AUC** resume la curva de Precisión-Exhaustividad calculando el área bajo la misma. A diferencia del ROC-AUC, que puede inflarse artificialmente por la abundancia de verdaderos negativos en datasets desbalanceados, el PR-AUC se enfoca exclusivamente en el rendimiento sobre la clase positiva, siendo la métrica recomendada cuando el desbalance es extremo.

---

## Técnicas de Manejo de Clases Desbalanceadas

Para evitar que el modelo se sesgue hacia la clase mayoritaria, se emplean estrategias a nivel de datos y a nivel de algoritmo.

### SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE genera **muestras sintéticas** interpolando linealmente entre un ejemplo de la clase minoritaria y uno de sus $k$ vecinos más cercanos de la misma clase:

$$x_{nuevo} = x_i + \lambda \cdot (x_{zi} - x_i), \quad \lambda \sim \text{Uniforme}[0,1]$$

Esto enriquece la región de decisión sin limitarse a repetir los mismos puntos, reduciendo el riesgo de sobreajuste que produce la simple duplicación.

### ADASYN (Adaptive Synthetic Sampling)

ADASYN opera con la misma lógica de interpolación que SMOTE, pero con un criterio adaptativo: **concentra la generación de muestras en los ejemplos minoritarios más difíciles de aprender**, es decir, los que tienen mayor proporción de vecinos de la clase mayoritaria en su entorno $k$-NN. Esto desplaza el límite de decisión hacia las zonas de frontera más conflictivas.

### `class_weight='balanced'`

Esta técnica opera a **nivel algorítmico**, sin modificar el dataset. Asigna a cada clase un peso inversamente proporcional a su frecuencia:

$$w_c = \frac{N}{K \cdot N_c}$$

lo que penaliza con mayor severidad los errores sobre la clase minoritaria durante el entrenamiento. Es más rápida que SMOTE/ADASYN y no introduce riesgo de ruido por interpolación.

---

## Técnicas de Optimización de Hiperparámetros

Encontrar la configuración óptima del modelo es vital para maximizar su capacidad predictiva. La superficie de error en el espacio de hiperparámetros suele ser no convexa y costosa de evaluar, lo que convierte este proceso en un problema de optimización por sí mismo.

### GridSearchCV

Búsqueda exhaustiva (*fuerza bruta*) que evalúa **todas las combinaciones posibles** de un espacio de hiperparámetros definido explícitamente, usando validación cruzada. Es riguroso y garantiza encontrar el óptimo dentro de la cuadrícula, pero su costo crece exponencialmente con el número de hiperparámetros.

### RandomizedSearchCV

En lugar de probar todas las combinaciones, **muestrea aleatoriamente** un número fijo de configuraciones a partir de distribuciones estadísticas. Con el mismo presupuesto de evaluaciones que GridSearch, explora un espacio mucho más amplio, logrando resultados cercanos al óptimo en una fracción del tiempo.

### Optuna

Framework de **optimización bayesiana** basado en el algoritmo *Tree-structured Parzen Estimator* (TPE). A diferencia de las búsquedas aleatorias, Optuna aprende del historial de evaluaciones para guiar la búsqueda hacia regiones prometedoras del espacio de hiperparámetros. Incluye además **poda temprana** (*pruning*) de ensayos con bajo rendimiento, lo que reduce significativamente el costo computacional.

### DEAP (Distributed Evolutionary Algorithms in Python)

Optimización mediante **Algoritmos Genéticos**. Mantiene una población de configuraciones y aplica iterativamente tres operaciones: *selección* de las más aptas, *cruce* entre progenitores para generar descendientes, y *mutación* aleatoria para preservar diversidad. Favorece la exploración global y es especialmente útil en superficies de error altamente no convexas o irregulares.

**Nota Importante sobre la Temporalidad:** Es fundamental destacar que, en el contexto de prevención de lavado de activos para este dataset, respetar la estricta cronología de las transacciones es indispensable. Cualquier validación, entrenamiento o ajuste de hiperparámetros debe garantizar que el aprendizaje provenga exclusivamente de datos pasados para predecir eventos futuros. Incurrir en un descuido de la temporalidad conduciría irremediablemente a la fuga de información (*data leakage*), lo cual inflaría el rendimiento del modelo de manera artificial, haciéndolo parecer efectivo en validación pero inutilizable en un escenario real y productivo.