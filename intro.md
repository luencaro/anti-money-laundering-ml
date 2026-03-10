# Detección de Lavado de Dinero con Machine Learning

```{image} Docs/logo.png
:alt: Logo Universidad
:width: 200px
:align: center
```

---

## Información del Curso

|               |                                                  |
|---------------|--------------------------------------------------|
| **Curso**     | Machine Learning — NRC 1627                      |
| **Docente**   | Lihki Rubio                                      |
| **Integrantes** | Luis Cabarcas · Natalia Frias · Luis Cantillo  |

---

## Descripción

Proyecto de clasificación binaria para detectar transacciones de lavado de dinero utilizando el dataset
[IBM Transactions for Anti Money Laundering (AML) — HI-Small](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).

El dataset presenta un **desbalance extremo de clases** (~0.1% lavado vs ~99.9% legítimo), lo que
requiere técnicas especializadas de evaluación y modelado.

## Contenido del Proyecto (Etapa 1): EDA y Baseline

1. **Análisis Exploratorio de Datos (EDA):** distribución de clases, análisis de montos, formatos de pago, patrones temporales y tipos de entidad.
2. **Limpieza de Datos:** verificación de nulos, duplicados, montos negativos y consistencia de cuentas.
3. **Ingeniería de Características:** creación de features relevantes para el modelado.
4. **Modelo Baseline — Regresión Logística:** evaluación con Cross-Validation estratificado (5-Fold) usando PR-AUC como métrica principal.

## Hallazgos Principales

- Desbalance extremo de clases (~0.1% lavado vs ~99.9% legítimo).
- La regresión logística sin ajuste no detecta lavado; con `class_weight='balanced'` mejora recall pero genera muchos falsos positivos.
- Se requieren técnicas avanzadas: resampling, modelos ensemble y feature engineering de grafos.

## Datos

El dataset no se incluye en el repositorio por su tamaño. Puede descargarlo desde
[Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
y colocarlo en la carpeta `Data/`.
