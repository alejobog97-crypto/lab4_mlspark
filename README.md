# Taller 4: Machine Learning Escalable con Spark ML

**Diplomado en Gestion de Datos - Universidad Santo Tomas**

---

## Descripcion

Taller practico basado en **retos y ejercicios** para aprender Machine Learning distribuido con Spark ML y MLOps con MLflow. Usaras datos reales de contratos publicos de **Colombia Compra Eficiente (SECOP II)**.

Cada notebook presenta los conceptos teoricos y luego plantea retos con `TODO` que debes completar. Las soluciones sugeridas estan comentadas como referencia.

**Secciones evaluadas**:
- **Seccion 13**: Spark ML - Pipelines y Feature Engineering
- **Seccion 14**: Modelos de Regresion y Regularizacion
- **Seccion 15**: Optimizacion de Hiperparametros y Validacion Cruzada
- **Seccion 16**: MLOps - Tracking, Registro y Despliegue con MLflow

---

## Objetivos de Aprendizaje

1. Escalar algoritmos de ML a datasets masivos usando Spark MLlib
2. Construir Pipelines reproducibles con Transformers y Estimators
3. Implementar regresion lineal, logistica y regularizacion
4. Aplicar Grid Search y validacion cruzada para tuning
5. Gestionar el ciclo de vida de modelos con MLflow

---

## Dataset: SECOP II - Contratos Electronicos

**Fuente**: [Datos Abiertos Colombia - SECOP II](https://www.datos.gov.co/Gastos-Gubernamentales/SECOP-II-Contratos-Electr-nicos/jbjy-vk9h/data)

Contratos electronicos del Sistema Electronico para la Contratacion Publica de Colombia.

**Campos clave**: Referencia del Contrato, Precio Base, Departamento, Tipo de Contrato, Fecha de Firma, Plazo de Ejecucion, Proveedor Adjudicado, Estado del Contrato.

---

## Estructura del Proyecto

```
lab4_mlspark/
├── README.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_ingesta_datos.py
│   ├── 02_exploracion_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_transformaciones.py
│   ├── 05_regresion_lineal.py
│   ├── 06_regresion_logistica.py
│   ├── 07_regularizacion.py
│   ├── 08_validacion_cruzada.py
│   ├── 09_optimizacion_hiperparametros.py
│   ├── 10_mlflow_tracking.py
│   ├── 11_model_registry.py
│   └── 12_inferencia_produccion.py
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_utils.py
│   └── model_trainer.py
│
└── mlruns/
```

---

## Guia de Notebooks y Retos

### Fase 1: Ingesta y Exploracion

#### Notebook 01: Ingesta de Datos
**Archivo**: [notebooks/01_ingesta_datos.py](notebooks/01_ingesta_datos.py)

| Reto | Descripcion |
|------|-------------|
| Reto 1 | Configurar SparkSession conectada al cluster |
| Reto 2 | Descargar datos desde la API de Datos Abiertos Colombia |
| Reto 3 | Cargar datos en Spark y explorar el esquema |
| Reto 4 | Seleccionar columnas clave para ML |
| Reto 5 | Guardar en formato Parquet optimizado |

#### Notebook 02: Analisis Exploratorio (EDA)
**Archivo**: [notebooks/02_exploracion_eda.py](notebooks/02_exploracion_eda.py)

| Reto | Descripcion |
|------|-------------|
| Reto 1 | Calcular estadisticas descriptivas |
| Reto 2 | Analizar valores nulos y decidir estrategia |
| Reto 3 | Explorar la variable objetivo (valor de contratos) |
| Reto 4 | Analizar distribucion por departamento |
| Reto 5 | Explorar tipo de contrato y estado |
| Reto 6 | Detectar outliers con metodo IQR |
| Bonus | Analisis temporal de contratos |

---

### Fase 2: Feature Engineering (Seccion 13)

#### Notebook 03: Pipelines y Feature Engineering
**Archivo**: [notebooks/03_feature_engineering.py](notebooks/03_feature_engineering.py)

| Reto | Descripcion |
|------|-------------|
| Reto 1 | Seleccionar features categoricas y numericas |
| Reto 2 | Implementar estrategia de limpieza de datos |
| Reto 3 | Crear VectorAssembler para combinar features |
| Reto 4 | Construir Pipeline completo (orden correcto de stages) |
| Bonus 1 | Calcular dimension total de features post-encoding |
| Bonus 2 | Analisis de varianza de features |

#### Notebook 04: Transformaciones Avanzadas
**Archivo**: [notebooks/04_transformaciones.py](notebooks/04_transformaciones.py)

| Reto | Descripcion |
|------|-------------|
| Reto 1 | Analizar por que normalizar (examinar escalas) |
| Reto 2 | Comparar antes y despues de StandardScaler |
| Reto 3 | Configurar PCA y elegir numero de componentes |
| Reto 4 | Analizar varianza explicada por componente |
| Reto 5 | Integrar todo en un Pipeline completo |
| Bonus | Experimentar con diferentes valores de k en PCA |

---

### Fase 3: Modelos de Regresion (Seccion 14)

#### Notebook 05: Regresion Lineal
**Archivo**: [notebooks/05_regresion_lineal.py](notebooks/05_regresion_lineal.py)

| Reto | Descripcion |
|------|-------------|
| Reto 1 | Definir estrategia de train/test split |
| Reto 2 | Configurar modelo de LinearRegression |
| Reto 3 | Interpretar R² del modelo |
| Reto 4 | Analizar calidad de predicciones y errores |
| Reto 5 | Comparar train vs test para detectar overfitting |
| Reto 6 | Analizar coeficientes del modelo |
| Bonus 1 | Analisis de distribucion de residuos |
| Bonus 2 | Feature importance aproximado |

#### Notebook 06: Regresion Logistica
**Archivo**: [notebooks/06_regresion_logistica.py](notebooks/06_regresion_logistica.py)

| Reto | Descripcion |
|------|-------------|
| Reto 1 | Crear variable objetivo binaria (definir criterio de riesgo) |
| Reto 2 | Analizar balance de clases |
| Reto 3 | Entender diferencia con regresion lineal |
| Reto 4 | Configurar modelo con threshold apropiado |
| Reto 5 | Interpretar probabilidades de prediccion |
| Reto 6 | Evaluar con AUC-ROC, Precision, Recall, F1 |
| Reto 7 | Construir e interpretar matriz de confusion |
| Bonus 1 | Experimentar con diferentes thresholds |
| Bonus 2 | Implementar curva ROC |

#### Notebook 07: Regularizacion
**Archivo**: [notebooks/07_regularizacion.py](notebooks/07_regularizacion.py)

| Reto | Descripcion |
|------|-------------|
| Reto 1 | Entender cuando y por que usar regularizacion |
| Reto 2 | Configurar evaluador de modelos |
| Reto 3 | Entrenar modelos con multiples combinaciones L1/L2/ElasticNet |
| Reto 4 | Analizar resultados y encontrar mejor modelo |
| Reto 5 | Comparar overfitting entre tipos de regularizacion |
| Reto 6 | Entrenar y guardar modelo final |
| Bonus | Visualizar efecto de lambda en coeficientes Lasso |

---

### Fase 4: Optimizacion de Hiperparametros (Seccion 15)

#### Notebook 08: Validacion Cruzada
**Archivo**: [notebooks/08_validacion_cruzada.py](notebooks/08_validacion_cruzada.py)

| Reto | Descripcion |
|------|-------------|
| Reto 1 | Entender concepto de K-Fold (diagrama mental) |
| Reto 2 | Crear modelo base y evaluador |
| Reto 3 | Construir ParamGrid de hiperparametros |
| Reto 4 | Configurar CrossValidator (elegir K) |
| Reto 5 | Ejecutar CV y analizar metricas por configuracion |
| Reto 6 | Comparar CV vs simple train/test split |
| Bonus | Experimentar con K=3, K=5, K=10 |

#### Notebook 09: Optimizacion de Hiperparametros
**Archivo**: [notebooks/09_optimizacion_hiperparametros.py](notebooks/09_optimizacion_hiperparametros.py)

| Reto | Descripcion |
|------|-------------|
| Reto 1 | Disenar grid de hiperparametros (escala logaritmica) |
| Reto 2 | Implementar Grid Search + Cross-Validation |
| Reto 3 | Implementar Train-Validation Split |
| Reto 4 | Comparar ambas estrategias (rendimiento vs velocidad) |
| Reto 5 | Seleccionar y guardar modelo final con hiperparametros |
| Bonus | Refinar grid alrededor de mejores valores |

---

### Fase 5: MLOps y Produccion (Seccion 16)

#### Notebook 10: MLflow Tracking
**Archivo**: [notebooks/10_mlflow_tracking.py](notebooks/10_mlflow_tracking.py)

| Reto | Descripcion |
|------|-------------|
| Reto 1 | Configurar MLflow tracking server y experimento |
| Reto 2 | Registrar experimento baseline con log_param/log_metric |
| Reto 3 | Registrar multiples modelos (Ridge, Lasso, ElasticNet) |
| Reto 4 | Explorar y comparar runs en MLflow UI |
| Reto 5 | Agregar artefactos personalizados (reportes, graficos) |

#### Notebook 11: Model Registry
**Archivo**: [notebooks/11_model_registry.py](notebooks/11_model_registry.py)

| Reto | Descripcion |
|------|-------------|
| Reto 1 | Configurar MLflow y MlflowClient |
| Reto 2 | Entrenar y registrar modelo v1 (baseline) |
| Reto 3 | Entrenar y registrar modelo v2 (mejorado) |
| Reto 4 | Gestionar stages: None → Staging → Production → Archived |
| Reto 5 | Agregar metadata y descripcion al modelo |
| Reto 6 | Cargar modelo desde Registry para prediccion |

#### Notebook 12: Inferencia en Produccion
**Archivo**: [notebooks/12_inferencia_produccion.py](notebooks/12_inferencia_produccion.py)

| Reto | Descripcion |
|------|-------------|
| Reto 1 | Cargar modelo en Production desde MLflow Registry |
| Reto 2 | Preparar datos nuevos para prediccion |
| Reto 3 | Generar predicciones batch con timestamp |
| Reto 4 | Monitorear predicciones (estadisticas, anomalias, rangos) |
| Reto 5 | Guardar resultados en Parquet y CSV |
| Reto 6 | Disenar pipeline de produccion automatizado |
| Bonus | Simulacion de scoring continuo por lotes |

---

## Como Trabajar con los Notebooks

1. **Lee la introduccion** de cada notebook para entender los conceptos
2. **Completa los `TODO`** con tu codigo
3. **Responde las preguntas** de reflexion en los comentarios
4. **Descomenta las soluciones sugeridas** si te bloqueas (estan como referencia)
5. **Ejecuta celda por celda** para verificar resultados

---

## Prerrequisitos

- **Docker Desktop** instalado y corriendo
- **Git** para clonar el repositorio
- **Al menos 8 GB de RAM** disponible para Docker
- **10 GB de espacio en disco**

---

## Inicio Rapido

```bash
# 1. Clonar repositorio
git clone <url-del-repositorio>
cd lab4_mlspark

# 2. Levantar cluster
docker-compose up --build -d

# 3. Verificar servicios
# Jupyter Lab:    http://localhost:8888 (password: spark)
# Spark Master:   http://localhost:8080
# MLflow UI:      http://localhost:5000
```

---

## Evaluacion

| Criterio | Peso | Notebooks |
|----------|------|-----------|
| **Feature Engineering** | 25% | 03, 04 |
| **Modelos de ML** | 25% | 05, 06, 07 |
| **Optimizacion** | 20% | 08, 09 |
| **MLOps** | 20% | 10, 11, 12 |
| **Reflexion y analisis** | 10% | Preguntas en todos los notebooks |

---

## Limpieza del Entorno

```bash
# Detener servicios
docker-compose down

# Eliminar datos y logs
docker-compose down -v

# Reinicio completo
rm -rf data/processed/* mlruns/*
docker-compose up --build -d
```

---

## Recursos

- [Spark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [SECOP II - Datos Abiertos](https://www.datos.gov.co/Gastos-Gubernamentales/SECOP-II-Contratos-Electr-nicos/jbjy-vk9h/data)
- [Seccion 13: Spark ML](https://ustatisticaldatapulse.github.io/diplomado-gestion-datos/seccion_13.html)
- [Seccion 14: Regresion](https://ustatisticaldatapulse.github.io/diplomado-gestion-datos/seccion_14.html)
- [Seccion 15: Tuning](https://ustatisticaldatapulse.github.io/diplomado-gestion-datos/seccion_15.html)
- [Seccion 16: MLOps](https://ustatisticaldatapulse.github.io/diplomado-gestion-datos/seccion_16.html)

---

**Diplomado en Gestion de Datos - Universidad Santo Tomas**
