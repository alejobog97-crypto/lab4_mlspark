# %% [markdown]
# # Notebook 11: Model Registry con MLflow
#
# **Sección 16 - MLOps**: Versionamiento y gestión del ciclo de vida
#
# **Objetivo**: Registrar modelos, crear versiones y promover a producción
#
# ## Conceptos clave:
# - **Model Registry**: Catálogo centralizado de modelos
# - **Versioning**: Cada modelo puede tener múltiples versiones (v1, v2, etc.)
# - **Stages**: Ciclo de vida: None -> Staging -> Production -> Archived
# - **MlflowClient**: API programática para gestionar el registry
#
# ## Actividades:
# 1. Registrar modelo en MLflow Model Registry
# 2. Crear versiones (v1, v2, etc.)
# 3. Transicionar entre stages: None -> Staging -> Production
# 4. Cargar modelo desde Registry

# %%
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import when, col
from pyspark.sql.functions import abs as spark_abs, col
from pyspark.ml.feature import StandardScaler, PCA, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import ParamGridBuilder
from delta import configure_spark_with_delta_pip
from pyspark.ml.feature import VectorAssembler
import numpy as np
import mlflow
import mlflow.spark

# %%
# %%
# Configurar SparkSession
builder = (
    SparkSession.builder
    .appName("SECOP_EDA")
    .master("spark://spark-master:7077")
    .config("spark.executor.memory", "2g")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()

print(f"Spark Version: {spark.version}")

# %%
# RETO 1: Configurar MLflow y el Model Registry

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator
from mlflow.models.signature import infer_signature


# 1. Configurar la URI del Tracking Server
mlflow.set_tracking_uri("http://mlflow:5000")

# 2. Crear cliente para interactuar con el Model Registry
client = MlflowClient()

# 3. Definir nombre descriptivo del modelo en el registry
model_name = "secop_prediccion_contratos"

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Nombre del modelo en el Registry: {model_name}")

# %%
# Cargar datos
df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")

# Renombrar columnas para ML
df = (
    df.withColumnRenamed("valor_del_contrato_num", "label")
      .withColumnRenamed("features_pca", "features")
      .filter(col("label").isNotNull())
)

# Split train / test
train, test = df.randomSplit([0.8, 0.2], seed=42)

print(f"Train: {train.count():,}")
print(f"Test:  {test.count():,}")

# Evaluador
evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)


# %%
# RETO 2: Entrenar y Registrar Modelo v1 (Baseline)

import mlflow
from pyspark.ml.regression import LinearRegression

# 1. Configurar el experimento en MLflow
# =========================================
# Imports
# =========================================
import mlflow
from mlflow.tracking import MlflowClient

# =========================================
# CONFIGURACIÓN MLFLOW (AQUÍ 👇)
# =========================================
mlflow.set_tracking_uri("http://mlflow:5000")

client = MlflowClient()

print("MLflow Tracking URI:", mlflow.get_tracking_uri())

mlflow.set_experiment("/SECOP_Model_Registry")


# 2. Entrenar y registrar el modelo baseline (sin regularización)
with mlflow.start_run(run_name="model_v1_baseline") as run:

    # Entrenar modelo baseline
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        regParam=0.0,     # Sin regularización
        elasticNetParam=0.0,
        maxIter=100
    )
    model_v1 = lr.fit(train)

    # Evaluar en test
    predictions = model_v1.transform(test)
    rmse_v1 = evaluator.evaluate(predictions)

    # Log de parámetros
    mlflow.log_param("version", "1.0")
    mlflow.log_param("model_type", "baseline")
    mlflow.log_param("regParam", 0.0)
    mlflow.log_param("elasticNetParam", 0.0)
    mlflow.log_param("maxIter", 100)

    # Log de métricas
    mlflow.log_metric("rmse", rmse_v1)

    # Registrar el modelo en el Model Registry (v1)
    
    mlflow.spark.log_model(
        spark_model=model_v1,
        artifact_path="model",
        registered_model_name=model_name,
        pip_requirements=[
        "pyspark==3.5.0",
        "mlflow>=2.9.0",
        "numpy",
        "pandas"
    ]
        )
    run_id_v1 = run.info.run_id
    print(f"✓ Modelo v1 registrado correctamente")
    print(f"  Run ID: {run_id_v1}")
    print(f"  RMSE Test: ${rmse_v1:,.2f}")


# %%
# RETO 3: Entrenar y Registrar Modelo v2 (Mejorado)

import mlflow
from pyspark.ml.regression import LinearRegression

# Asumimos que:
# - model_name ya está definido
# - rmse_v1 viene del RETO 2 (modelo baseline)
# - train, test y evaluator ya existen

with mlflow.start_run(run_name="model_v2_regularized") as run:

    # 1. Entrenar modelo con regularización (hiperparámetros óptimos)
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        regParam=0.1,          # λ óptimo (ejemplo)
        elasticNetParam=0.5,   # ElasticNet
        maxIter=100
    )
    model_v2 = lr.fit(train)

    # 2. Evaluar en test
    predictions_v2 = model_v2.transform(test)
    rmse_v2 = evaluator.evaluate(predictions_v2)

    # 3. Log de parámetros
    mlflow.log_param("version", "2.0")
    mlflow.log_param("model_type", "regularized")
    mlflow.log_param("regParam", 0.1)
    mlflow.log_param("elasticNetParam", 0.5)
    mlflow.log_param("maxIter", 100)

    # 4. Log de métricas
    mlflow.log_metric("rmse", rmse_v2)

    # 5. Registrar el modelo como nueva versión en el Model Registry
    mlflow.spark.log_model(
        spark_model=model_v2,
        artifact_path="model",
        registered_model_name=model_name
    )

    run_id_v2 = run.info.run_id
    print(f"✓ Modelo v2 registrado correctamente")
    print(f"  Run ID: {run_id_v2}")
    print(f"  RMSE Test: ${rmse_v2:,.2f}")

# %%
# Comparación v1 vs v2
print("\n=== COMPARACIÓN DE MODELOS ===")
print(f"RMSE v1 (baseline):   ${rmse_v1:,.2f}")
print(f"RMSE v2 (regularizado): ${rmse_v2:,.2f}")
print(f"Mejor modelo: {'v2' if rmse_v2 < rmse_v1 else 'v1'}")


# %%
# RETO 4: Gestionar Versiones y Stages en MLflow Model Registry

from mlflow.tracking import MlflowClient

# Asumimos que:
# - client ya está creado
# - model_name está definido
# - rmse_v1 y rmse_v2 existen (de los retos anteriores)

# 1. Listar versiones registradas del modelo
model_versions = client.search_model_versions(f"name='{model_name}'")

print(f"\n=== VERSIONES DEL MODELO '{model_name}' ===")
for mv in model_versions:
    print(
        f"Versión {mv.version} | "
        f"Stage: {mv.current_stage} | "
        f"Run ID: {mv.run_id[:8]}"
    )

# %%
# 2. Promover versión 1 a Staging (ejemplo inicial)
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"
)
print("\n✓ Versión 1 promovida a STAGING")

# %%
# 3. Promover la mejor versión a Production
if rmse_v2 < rmse_v1:
    client.transition_model_version_stage(
        name=model_name,
        version=2,
        stage="Production"
    )
    print("✓ Versión 2 promovida a PRODUCTION (mejor desempeño)")

    # 4. Archivar versión anterior
    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage="Archived"
    )
    print("✓ Versión 1 archivada")
else:
    print("ℹ️ La versión 1 sigue siendo mejor; no se promueve v2")

# %%
# Reflexión (responder en markdown del notebook):
#
# ¿Por qué pasar por Staging antes de Production?
# - Permite validar el modelo en un entorno controlado
# - Facilita pruebas de integración y monitoreo inicial
# - Reduce el riesgo de impactar sistemas productivos
# - Asegura gobernanza y trazabilidad del ciclo de vida del modelo


# %%
# RETO 5: Agregar Metadata al Modelo en MLflow Model Registry

from datetime import datetime

# Determinar la mejor versión según RMSE
best_version = 2 if rmse_v2 < rmse_v1 else 1
best_rmse = min(rmse_v1, rmse_v2)

# Construir descripción detallada del modelo
descripcion_modelo = f"""
Modelo de predicción de valor de contratos SECOP

Detalles:
- Versión: {best_version}
- RMSE (test): ${best_rmse:,.2f}
- Fecha de registro: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Autor: Equipo de Analítica / Ciencia de Datos
- Dataset: secop_ml_ready.parquet
- Features: PCA + variables numéricas normalizadas
- Tipo de modelo: Regresión Lineal (Spark ML)
"""

# Actualizar metadata de la versión en el registry
client.update_model_version(
    name=model_name,
    version=best_version,
    description=descripcion_modelo
)

print(f"✓ Metadata agregada correctamente a la versión {best_version}")

# %%
# Reflexión (responder en markdown del notebook):
#
# ¿Qué información mínima debería tener cada versión de modelo?
# - Objetivo del modelo
# - Dataset y features utilizados
# - Métricas clave (RMSE, MAE, R², etc.)
# - Fecha y autor
# - Hiperparámetros principales
# - Estado en el ciclo de vida (Staging / Production)


# %%
# RETO 6: Cargar Modelo desde el MLflow Model Registry y Validar





#Cargar el modelo desde el Registry
import mlflow

# Configurar el tracking URI (esto ya lo tienes)
mlflow.set_tracking_uri("http://mlflow:5000")

model_name = "secop_prediccion_contratos"
model_uri = f"models:/{model_name}/Production"

# Spark necesita saber dónde están los archivos temporales para la descarga
loaded_model = mlflow.spark.load_model(model_uri)

print(f"✓ Modelo cargado desde el Registry")
print(f"URI: {model_uri}")
print(f"Tipo de modelo: {type(loaded_model)}")

# %%
# Verificar que el modelo funciona correctamente

# Generar predicciones sobre el set de test
test_predictions = loaded_model.transform(test)

# Evaluar RMSE
test_rmse = evaluator.evaluate(test_predictions)

print("\n=== VALIDACIÓN DEL MODELO EN PRODUCCIÓN ===")
print(f"RMSE esperado (registro): ${min(rmse_v1, rmse_v2):,.2f}")
print(f"RMSE verificación:        ${test_rmse:,.2f}")

# %%
# Reflexión (responder en markdown del notebook):
#
# - Si el RMSE coincide (o es muy cercano), el modelo cargado es correcto
# - Si difiere mucho, puede indicar:
#   • Cambio en datos de entrada
#   • Error en el pipeline previo
#   • Versión incorrecta en Production
#
# Ventaja clave:
# 👉 Cambiar el modelo en Production NO requiere cambiar este código


# %% [markdown]
# ## Preguntas de Reflexión
#
# 1. **¿Cómo harías rollback si el modelo en Production falla?**
#    **Respuesta:**
#    Haría el rollback directamente desde el **MLflow Model Registry**, sin modificar código.
#    Simplemente se promueve una versión anterior estable al stage **Production** y se mueve
#    la versión fallida a **Archived**. Como las aplicaciones cargan el modelo por
#    `models:/nombre_modelo/Production`, el cambio es inmediato y sin downtime.
#
# 2. **¿Qué criterios usarías para promover un modelo de Staging a Production?**
#    **Respuesta:**
#    - Métricas técnicas mejores o equivalentes (RMSE, MAE, R²) frente al modelo actual.
#    - Validación correcta en dataset de test o validación reciente.
#    - Ausencia de overfitting, sesgos críticos o errores funcionales.
#    - Impacto positivo esperado en el negocio.
#    - Modelo reproducible, con metadata y artefactos completos en MLflow.
#
# 3. **¿Cómo implementarías A/B testing con el Model Registry?**
#    **Respuesta:**
#    Mantendría dos versiones del modelo (A y B) en el registry y enrutaría el tráfico
#    de predicciones (por ejemplo 80/20) desde la aplicación. Se comparan métricas técnicas
#    y de negocio entre ambos modelos y, si B supera a A, se promueve a Production.
#    En caso contrario, se descarta o se mantiene en Staging.
#
# 4. **¿Quién debería tener permisos para promover modelos a Production?**
#    **Respuesta:**
#    Los permisos deberían estar restringidos a roles como **ML Engineer, MLOps Engineer
#    o Tech/Data Lead**. Los Data Scientists entrenan y validan modelos, pero la promoción
#    a Production debe pasar por un control de gobierno para reducir riesgos.
#
# %%
# TODO: Escribe tus respuestas arriba


# %%
print("\n" + "="*60)
print("RESUMEN MODEL REGISTRY")
print("="*60)
print("Verifica que hayas completado:")
print("  [ ] Registrado modelo v1 (baseline)")
print("  [ ] Registrado modelo v2 (mejorado)")
print("  [ ] Transicionado versiones entre stages")
print("  [ ] Agregado metadata al modelo")
print("  [ ] Cargado modelo desde Registry")
print("  [ ] Accede a Model Registry: http://localhost:5000/#/models")
print("="*60)

# %%
spark.stop()
