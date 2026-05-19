# Detección de Lavado de Activos con Machine Learning

**Integrantes:** Luis Cabarcas · Natalia Frías · Luis Cantillo

---

## Descripción

Proyecto de clasificación binaria para detectar transacciones de lavado de activos
utilizando el dataset público
[IBM Transactions for Anti Money Laundering (AML) — HI-Small](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
El objetivo es construir un sistema capaz de identificar la clase minoritaria (lavado)
con alta precisión bajo condiciones de desbalance extremo (~0.18% de prevalencia),
evaluando múltiples arquitecturas desde modelos baseline hasta un clasificador en
cascada propuesto (CascadeBoost).

---

## Estructura del Proyecto

El proyecto se desarrolla en dos etapas principales:

### Etapa 1 — EDA y Baseline
1. **Análisis Exploratorio de Datos (EDA):** distribución de clases, análisis de
   montos, formatos de pago, patrones temporales y tipos de entidad.
2. **Limpieza de Datos:** verificación de nulos, duplicados, montos negativos y
   consistencia de cuentas.
3. **Ingeniería de Características:** creación de features transaccionales y de
   comportamiento de red relevantes para el modelado.
4. **Modelo Baseline — Regresión Logística:** evaluación con Cross-Validation
   estratificado (5-Fold) usando PR-AUC como métrica principal, estableciendo el
   piso de rendimiento y motivando el uso de técnicas avanzadas.

### Etapa 2 — Modelado Avanzado y Comparativa
1. **CascadeBoost:** arquitectura en dos etapas (RUSBoost → XGBoost) con
   validación temporal estratificada (`StratifiedTimeSeriesSplit`) y búsqueda de
   hiperparámetros mediante Random Search, GridSearch, Optuna y DEAP.
2. **Benchmarking:** comparativa de siete modelos propios (Naive Bayes, Decision
   Tree, Ridge, LogReg, XGBoost, SVM, CascadeBoost) con test de DeLong, bootstrap
   IC 95% y posicionamiento frente al estado del arte (Egressy et al., AAAI 2024).

---

## Hallazgos Principales

- Desbalance extremo de clases (~0.18% lavado vs ~99.82% legítimo), lo que hace
  del PR-AUC la métrica de referencia frente al ROC-AUC.
- La regresión logística sin ajuste no detecta lavado; con `class_weight='balanced'`
  mejora recall pero genera una cantidad inaceptable de falsos positivos.
- **CascadeBoost** es el único modelo que equilibra precisión (0.865) y recall
  (0.503), alcanzando F1 = 0.636 y PR-AUC = 0.635, superando a todos los modelos
  propios y posicionándose a solo 4.6 puntos del SOTA de GNNs del paper de
  referencia (68.2%), sin utilizar información topológica de red.


## Datos

Descarga el dataset desde
[Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
y colócalo en la carpeta `Data/` antes de ejecutar los notebooks.

---

## Referencia

Egressy et al. *"Provably Powerful Graph Neural Networks for Directed Multigraphs"*,
AAAI 2024. [arXiv:2306.11586](https://arxiv.org/abs/2306.11586)

