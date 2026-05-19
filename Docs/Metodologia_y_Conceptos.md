# Metodología y Conceptos Previos

Antes de adentrarnos en la evaluación de los modelos benchmark, es fundamental establecer el marco de trabajo, las métricas elegidas y las técnicas empleadas para abordar tanto el desbalanceo de los datos como la optimización de los hiperparámetros. En problemas de detección de anomalías o lavado de dinero, las clases suelen estar fuertemente desbalanceadas, lo que requiere un tratamiento especial.

## Métricas de Evaluación

En datasets donde la clase positiva (fraude/lavado) representa una fracción muy pequeña del total, métricas tradicionales como la exactitud (*Accuracy*) pueden resultar engañosas, ya que un modelo que clasifique todo como la clase mayoritaria obtendría una puntuación alta sin detectar un solo caso de fraude.

### F1-Score de la Clase Minoritaria (Métrica Principal)
El **F1-Score** es la media armónica entre la Precisión y el Exhaustividad (*Recall*). Se utiliza como métrica principal porque busca un equilibrio entre no generar demasiadas falsas alarmas (Precisión) y no dejar escapar transacciones fraudulentas reales (Recall).

$$ F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} $$

Donde:
* **Precision:** Proporción de predicciones positivas que fueron realmente correctas ($ \frac{TP}{TP + FP} $).
* **Recall:** Proporción de casos positivos reales que fueron identificados correctamente ($ \frac{TP}{TP + FN} $).

### PR-AUC (Area Under the Precision-Recall Curve)
El **PR-AUC** resume la curva de Precisión-Exhaustividad (Precision-Recall) calculando el área bajo la misma. A diferencia del ROC-AUC (que incluye los verdaderos negativos de la clase mayoritaria), el PR-AUC se enfoca exclusivamente en el rendimiento sobre la clase positiva. Es la métrica recomendada para evaluar conjuntos de datos extremadamente desbalanceados.

## Técnicas de Manejo de Clases Desbalanceadas

Para evitar que el modelo se sesgue hacia la clase mayoritaria, se emplean estrategias a nivel de datos y a nivel de algoritmo.

### SMOTE (Synthetic Minority Over-sampling Technique)
SMOTE es una técnica de sobremuestreo probabilístico. En lugar de simplemente duplicar las muestras minoritarias existentes, SMOTE genera **muestras sintéticas** interpolando valores entre los ejemplos de la clase minoritaria y sus "k" vecinos más cercanos ($k$-NN). Esto enriquece la región de decisión sin provocar un sobreajuste severo.

### ADASYN (Adaptive Synthetic Sampling)
ADASYN opera con un enfoque similar a SMOTE, pero con un criterio adaptativo: **genera más datos sintéticos en aquellas observaciones de la clase minoritaria que son más difíciles de aprender** (es decir, las que están más rodeadas por ejemplos de la clase mayoritaria). Esto desplaza de forma efectiva el límite de decisión hacia los ejemplos difíciles.

### `class_weight='balanced'`
Esta técnica opera a **nivel algorítmico**. En lugar de alterar el conjunto de datos de entrenamiento, modifica la función de pérdida del modelo durante su ajuste, penalizando con mayor severidad los errores cometidos en la clase minoritaria. Los pesos son inversamente proporcionales a las frecuencias de las clases en los datos de entrenamiento.

## Técnicas de Optimización de Hiperparámetros

Encontrar la configuración óptima del modelo es vital para maximizar su capacidad predictiva.

### GridSearchCV
Es un método de búsqueda exhaustiva (*fuerza bruta*). Se define un espacio de hiperparámetros cerrado y GridSearchCV evalúa el rendimiento del modelo utilizando validación cruzada para **todas las combinaciones posibles**. Es riguroso, pero computacionalmente costoso para espacios muy grandes.

### RandomizedSearchCV
A diferencia de GridSearchCV, este enfoque no prueba todas las combinaciones, sino que **muestrea aleatoriamente** un número definido de configuraciones a partir de distribuciones estadísticas especificadas para cada hiperparámetro. Permite explorar un espacio de búsqueda mucho más amplio en una fracción del tiempo, logrando a menudo resultados cercanos al óptimo.

### Optuna
Optuna es un framework moderno fundamentado en **optimización bayesiana** (principalmente usando el algoritmo *Tree-structured Parzen Estimator* o TPE). A diferencia de las búsquedas ciegas o aleatorias, Optuna "aprende" del historial de evaluaciones pasadas para guiar inteligentemente la búsqueda hacia las regiones del espacio de hiperparámetros que son más prometedoras, siendo altamente eficiente.

### DEAP (Distributed Evolutionary Algorithms in Python)
DEAP se utiliza para la optimización mediante **Algoritmos Genéticos**. Inspirado en la evolución biológica, este enfoque genera una población inicial de hiperparámetros y aplica operaciones como *selección*, *cruce* (crossover) y *mutación* a lo largo de varias generaciones. Favorece la exploración global y es ideal para superficies de error altamente no convexas o irregulares.