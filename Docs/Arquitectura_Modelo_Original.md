# Arquitectura y Metodología del Modelo Original: CascadeBoost

Este documento describe los fundamentos y el funcionamiento del modelo propuesto para la detección de lavado de dinero, denominado **CascadeBoost**. La arquitectura combina dos clasificadores en secuencia, un esquema de validación temporal con estratificación, y un proceso de optimización de hiperparámetros en dos fases.

---

## 1. Validación Temporal: StratifiedTimeSeriesSplit

La detección de lavado de dinero está intrínsecamente ligada al tiempo: usar una validación cruzada estándar (K-Fold) implicaría entrenar con datos del futuro para predecir el pasado (*data leakage*). Para evitarlo, se implementó un validador cruzado personalizado — `StratifiedTimeSeriesSplit` — que combina dos propiedades esenciales:

**Expanding window temporal.** El conjunto de entrenamiento siempre crece hacia adelante en el tiempo. Si los datos se dividen en $n\_splits + 1$ bloques cronológicos, el fold $k$ entrena con los bloques $0 \dots k$ y valida en el bloque $k+1$. Nunca se ve información futura durante el entrenamiento.

**Estratificación por positivos.** Los datos de lavado de dinero tienen una prevalencia extremadamente baja (típicamente < 0.1%). Los bloques no se definen por tamaño fijo en número de muestras, sino por **cantidad de ejemplos positivos**: cada bloque contiene aproximadamente la misma cantidad de casos de lavado, garantizando que ningún fold de validación quede vacío de positivos. Formalmente, si hay $N_{pos}$ muestras positivas en total, cada uno de los $n\_splits + 1$ bloques contiene $\lfloor N_{pos} / (n\_splits + 1) \rfloor$ positivos.

El resultado es una validación cruzada robusta ante el desbalanceo extremo y respetuosa con la causalidad temporal, que es el estándar que utilizan bancos y reguladores en producción.

---

## 2. Arquitectura en Cascada: CascadeAMLClassifier

El modelo propuesto implementa una **arquitectura en cascada de dos etapas**, donde cada etapa cumple un rol distinto y complementario en el proceso de detección.

### Etapa 1 — RUSBoost (Filtro de alto Recall)

El primer clasificador es un **RUSBoost** (Random Under-Sampling + AdaBoost), cuyo objetivo es actuar como un filtro de alta sensibilidad: debe capturar prácticamente todos los casos de lavado, aunque eso implique dejar pasar también una cantidad de falsos positivos. Su lógica interna combina dos mecanismos:

- **Random Under-Sampling (RUS):** En cada iteración de boosting se construye un subconjunto de entrenamiento $S'_t$ donde la clase mayoritaria se submuestrea aleatoriamente hasta alcanzar la proporción definida por `sampling_strategy`. Esto equilibra el entrenamiento de cada clasificador débil sin necesidad de generar datos sintéticos.
- **AdaBoost (SAMME):** Los clasificadores débiles sucesivos se entrenan prestando mayor atención a los ejemplos difíciles. El peso de actualización es:

$$w_{t+1}(i) = w_t(i) \times \begin{cases} \alpha_t & \text{si } h_t(x_i) = y_i \\ 1 & \text{si } h_t(x_i) \neq y_i \end{cases}, \quad \alpha_t = \frac{\epsilon_t}{1 - \epsilon_t}$$

donde $\epsilon_t$ es el error ponderado del clasificador débil $h_t$. Los clasificadores bien ajustados reducen el peso de los ejemplos que ya dominan, concentrando la atención futura en los casos más difíciles.

La predicción de esta etapa es una **probabilidad de sospecha** $\hat{p}_1(x)$. Al inferir, solo las transacciones con $\hat{p}_1(x) \geq \theta_1$ pasan a la Etapa 2. El umbral $\theta_1$ se selecciona automáticamente sobre las predicciones *out-of-fold* (OOF) para garantizar un Recall $\geq 99\%$ sin data leakage.

**Predicciones OOF y leakage.** Para entrenar la Etapa 2 sin filtraciones de información, las probabilidades de Etapa 1 que alimentan a Etapa 2 durante el entrenamiento se obtienen como predicciones *out-of-fold*: cada muestra es puntuada por un modelo de Etapa 1 que nunca la vio durante su propio entrenamiento.

### Etapa 2 — XGBoost con balance dinámico (Filtro de alta Precisión)

Las transacciones filtradas por Etapa 1 son recibidas por un **XGBoost** (`DynamicBalanceXGB`), cuyo objetivo es depurar la lista de sospechosos: separar los verdaderos casos de lavado de los falsos positivos producidos por la primera etapa.

XGBoost opera con Gradient Boosting, construyendo árboles que minimizan el gradiente del error residual en cada iteración. Para manejar el desbalanceo residual presente en el subconjunto filtrado, se calcula `scale_pos_weight` **dentro de cada fold** (no sobre el dataset completo) para evitar leakage:

$$\text{scale\_pos\_weight} = \frac{N_{neg}}{N_{pos}} \bigg|_{\text{fold de entrenamiento}}$$

Esto equivale a asignar mayor peso a cada error sobre un positivo, sin modificar el dataset en sí.

### Modos de operación: Hard y Soft

La cascada puede operar en dos modos:

- **Modo Hard:** Solo las muestras con $\hat{p}_1(x) \geq \theta_1$ son enviadas a Etapa 2. XGBoost se entrena y predice exclusivamente sobre ese subconjunto. Las transacciones descartadas en Etapa 1 reciben automáticamente probabilidad 0.
- **Modo Soft:** Todas las muestras pasan a Etapa 2, pero la probabilidad de Etapa 1 ($\hat{p}_1$) se añade como una característica adicional al vector de entrada de XGBoost. Esto permite que Etapa 2 use la señal de sospecha de Etapa 1 sin descartar ninguna muestra del entrenamiento.

### Predicción final

La predicción final del modelo combina ambas etapas: se obtiene $\hat{p}_2(x)$ de Etapa 2 y se aplica un segundo umbral $\theta_2$ (por defecto 0.5) para producir la etiqueta binaria:

$$\hat{y} = \mathbb{1}\left[\hat{p}_2(x) \geq \theta_2\right]$$

---

## 3. Optimización de Hiperparámetros en dos fases

El proceso de ajuste de hiperparámetros respeta la arquitectura en cascada: primero se optimiza Etapa 1 de forma independiente, y luego se optimiza Etapa 2 sobre el subconjunto filtrado por los parámetros elegidos en Etapa 1.

### Fase 1 — Etapa 1 (RUSBoost)

Los hiperparámetros a optimizar son: `n_estimators` [50–500], `learning_rate` [0.01–2.0], `sampling_strategy` [0.1–0.9] y `max_depth` del árbol base [1–5]. Se implementaron cuatro estrategias de búsqueda comparables en presupuesto (30 evaluaciones × 3 folds):

| Método | Estrategia |
|---|---|
| **RandomizedSearchCV** | 30 configuraciones muestreadas de distribuciones estadísticas |
| **GridSearchCV** | Cuadrícula fija de 81 combinaciones discretas |
| **Optuna (TPE)** | 30 trials guiados por optimización bayesiana |
| **DEAP** | 10 individuos × 15 generaciones mediante algoritmo genético |

La métrica de selección es el **F1-Score** de la clase minoritaria promediado sobre los folds del `StratifiedTimeSeriesSplit`.

### Fase 2 — Etapa 2 (XGBoost)

Una vez fijados los parámetros de Etapa 1 y obtenidas las predicciones OOF, se construye el subconjunto de entrenamiento para Etapa 2 (muestras con $\hat{p}_1 \geq \theta_1$) y se repite el mismo proceso de búsqueda sobre los hiperparámetros de XGBoost: `learning_rate`, `max_depth` [3–10], `n_estimators` [50–300], `subsample` [0.5–1.0] y `colsample_bytree` [0.5–1.0].

Esta separación en dos fases reduce drásticamente el espacio de búsqueda conjunto y evita que la interacción entre parámetros de ambas etapas haga el problema de optimización intractable.

---

## 4. Experimentos de Optimización: Greedy vs. CV Estratificado

Para evaluar el impacto del esquema de validación sobre la calidad de los hiperparámetros encontrados, se realizaron dos experimentos con los mismos métodos de búsqueda (RandomSearch, GridSearch, Optuna y DEAP) pero con estrategias de validación interna distintas.

**Experimento 1 — Búsqueda Greedy.** En este enfoque se utiliza una validación cruzada estándar (sin restricción temporal ni estratificación), tomando decisiones localmente óptimas sobre el espacio de hiperparámetros sin considerar la naturaleza secuencial de los datos. Sirve como línea base para cuantificar cuánto aporta una validación bien especificada. Al ignorar el orden temporal, este experimento es propenso a seleccionar configuraciones que se benefician implícitamente de data leakage y que no necesariamente generalizan al período de prueba.

**Experimento 2 — Búsqueda con StratifiedTimeSeriesSplit.** Se reemplaza el validador interno de cada método de búsqueda por el `StratifiedTimeSeriesSplit` propuesto, de modo que cada evaluación de hiperparámetros respeta la causalidad temporal y garantiza representación de la clase minoritaria en cada fold de validación. Este experimento refleja las condiciones reales de despliegue del modelo y produce hiperparámetros más robustos frente a la deriva temporal del comportamiento transaccional.

La comparación entre ambos experimentos permite aislar la contribución del esquema de validación al rendimiento final del modelo, independientemente del método de búsqueda utilizado.

# Arquitectura y Metodología del Modelo Original: CascadeBoost

Este documento describe los fundamentos y el funcionamiento del modelo propuesto para la detección de lavado de dinero, denominado **CascadeBoost**. La arquitectura combina dos clasificadores en secuencia, un esquema de validación temporal con estratificación, y un proceso de optimización de hiperparámetros en dos fases.

---

## 1. Validación Temporal: StratifiedTimeSeriesSplit

La detección de lavado de dinero está intrínsecamente ligada al tiempo: usar una validación cruzada estándar (K-Fold) implicaría entrenar con datos del futuro para predecir el pasado (*data leakage*). Para evitarlo, se implementó un validador cruzado personalizado — `StratifiedTimeSeriesSplit` — que combina dos propiedades esenciales:

**Expanding window temporal.** El conjunto de entrenamiento siempre crece hacia adelante en el tiempo. Si los datos se dividen en $n\_splits + 1$ bloques cronológicos, el fold $k$ entrena con los bloques $0 \dots k$ y valida en el bloque $k+1$. Nunca se ve información futura durante el entrenamiento.

**Estratificación por positivos.** Los datos de lavado de dinero tienen una prevalencia extremadamente baja (típicamente < 0.1%). Los bloques no se definen por tamaño fijo en número de muestras, sino por **cantidad de ejemplos positivos**: cada bloque contiene aproximadamente la misma cantidad de casos de lavado, garantizando que ningún fold de validación quede vacío de positivos. Formalmente, si hay $N_{pos}$ muestras positivas en total, cada uno de los $n\_splits + 1$ bloques contiene $\lfloor N_{pos} / (n\_splits + 1) \rfloor$ positivos.

El resultado es una validación cruzada robusta ante el desbalanceo extremo y respetuosa con la causalidad temporal, que es el estándar que utilizan bancos y reguladores en producción.

---

## 2. Arquitectura en Cascada: CascadeAMLClassifier

El modelo propuesto implementa una **arquitectura en cascada de dos etapas**, donde cada etapa cumple un rol distinto y complementario en el proceso de detección.

### Etapa 1 — RUSBoost (Filtro de alto Recall)

El primer clasificador es un **RUSBoost** (Random Under-Sampling + AdaBoost), cuyo objetivo es actuar como un filtro de alta sensibilidad: debe capturar prácticamente todos los casos de lavado, aunque eso implique dejar pasar también una cantidad de falsos positivos. Su lógica interna combina dos mecanismos:

- **Random Under-Sampling (RUS):** En cada iteración de boosting se construye un subconjunto de entrenamiento $S'_t$ donde la clase mayoritaria se submuestrea aleatoriamente hasta alcanzar la proporción definida por `sampling_strategy`. Esto equilibra el entrenamiento de cada clasificador débil sin necesidad de generar datos sintéticos.
- **AdaBoost (SAMME):** Los clasificadores débiles sucesivos se entrenan prestando mayor atención a los ejemplos difíciles. El peso de actualización es:

$$w_{t+1}(i) = w_t(i) \times \begin{cases} \alpha_t & \text{si } h_t(x_i) = y_i \\ 1 & \text{si } h_t(x_i) \neq y_i \end{cases}, \quad \alpha_t = \frac{\epsilon_t}{1 - \epsilon_t}$$

donde $\epsilon_t$ es el error ponderado del clasificador débil $h_t$. Los clasificadores bien ajustados reducen el peso de los ejemplos que ya dominan, concentrando la atención futura en los casos más difíciles.

La predicción de esta etapa es una **probabilidad de sospecha** $\hat{p}_1(x)$. Al inferir, solo las transacciones con $\hat{p}_1(x) \geq \theta_1$ pasan a la Etapa 2. El umbral $\theta_1$ se selecciona automáticamente sobre las predicciones *out-of-fold* (OOF) para garantizar un Recall $\geq 99\%$ sin data leakage.

**Predicciones OOF y leakage.** Para entrenar la Etapa 2 sin filtraciones de información, las probabilidades de Etapa 1 que alimentan a Etapa 2 durante el entrenamiento se obtienen como predicciones *out-of-fold*: cada muestra es puntuada por un modelo de Etapa 1 que nunca la vio durante su propio entrenamiento.

### Etapa 2 — XGBoost con balance dinámico (Filtro de alta Precisión)

Las transacciones filtradas por Etapa 1 son recibidas por un **XGBoost** (`DynamicBalanceXGB`), cuyo objetivo es depurar la lista de sospechosos: separar los verdaderos casos de lavado de los falsos positivos producidos por la primera etapa.

XGBoost opera con Gradient Boosting, construyendo árboles que minimizan el gradiente del error residual en cada iteración. Para manejar el desbalanceo residual presente en el subconjunto filtrado, se calcula `scale_pos_weight` **dentro de cada fold** (no sobre el dataset completo) para evitar leakage:

$$\text{scale\_pos\_weight} = \frac{N_{neg}}{N_{pos}} \bigg|_{\text{fold de entrenamiento}}$$

Esto equivale a asignar mayor peso a cada error sobre un positivo, sin modificar el dataset en sí.

### Modos de operación: Hard y Soft

La cascada puede operar en dos modos:

- **Modo Hard:** Solo las muestras con $\hat{p}_1(x) \geq \theta_1$ son enviadas a Etapa 2. XGBoost se entrena y predice exclusivamente sobre ese subconjunto. Las transacciones descartadas en Etapa 1 reciben automáticamente probabilidad 0.
- **Modo Soft:** Todas las muestras pasan a Etapa 2, pero la probabilidad de Etapa 1 ($\hat{p}_1$) se añade como una característica adicional al vector de entrada de XGBoost. Esto permite que Etapa 2 use la señal de sospecha de Etapa 1 sin descartar ninguna muestra del entrenamiento.

### Predicción final

La predicción final del modelo combina ambas etapas: se obtiene $\hat{p}_2(x)$ de Etapa 2 y se aplica un segundo umbral $\theta_2$ (por defecto 0.5) para producir la etiqueta binaria:

$$\hat{y} = \mathbb{1}\left[\hat{p}_2(x) \geq \theta_2\right]$$

---

## 3. Experimentos de Optimización: Greedy vs. CV Global

Para encontrar los hiperparámetros óptimos del modelo se diseñaron dos experimentos con filosofías distintas sobre cómo tratar la cascada durante la búsqueda.

**Experimento 1 — Optimización Greedy (secuencial con OOF).** Cada etapa se optimiza de forma independiente y en orden: primero se buscan los mejores parámetros de RUSBoost (Etapa 1) mediante `StratifiedTimeSeriesSplit`, luego se generan las predicciones OOF con esos parámetros fijados, y finalmente se optimiza XGBoost (Etapa 2) sobre el subconjunto filtrado por $\theta_1$. Los cuatro métodos de búsqueda (RandomSearch, GridSearch, Optuna y DEAP) se ejecutan bajo este esquema, cada uno produciendo su propia configuración. Se denomina *greedy* porque toma decisiones localmente óptimas etapa por etapa sin explorar cómo interactúan los hiperparámetros de ambas etapas entre sí.

**Experimento 2 — Optimización CV Global (conjunta).** En lugar de optimizar las etapas por separado, se implementa un clasificador unificado (`CascadeClassifier_CV`) que encapsula toda la cascada —incluyendo la generación interna de OOF y el filtrado por $\theta_1$— como un único estimador compatible con `cross_val_score`. Esto permite a Optuna buscar simultáneamente sobre los hiperparámetros de Etapa 1, Etapa 2 y el propio umbral $\theta_1$, evaluando cada combinación con `StratifiedTimeSeriesSplit` sobre el pipeline completo. La búsqueda conjunta captura las interacciones entre ambas etapas que el enfoque greedy ignora, a costa de un mayor costo computacional por evaluación.
