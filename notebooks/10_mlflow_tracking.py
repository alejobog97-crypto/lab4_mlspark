# %% [markdown]
# # Notebook 10: MLflow Tracking
#
# **Sección 16 - MLOps**: Registro de experimentos con MLflow
#
# **Objetivo**: Rastrear experimentos, métricas y modelos con MLflow
#
# ## Conceptos clave:
# - **Experiment**: Agrupación lógica de runs (un proyecto)
# - **Run**: Una ejecución individual (un modelo entrenado)
# - **Parameters**: Hiperparámetros registrados (regParam, maxIter, etc.)
# - **Metrics**: Métricas de rendimiento (RMSE, R², etc.)
# - **Artifacts**: Archivos guardados (modelos, gráficos, etc.)
#
# ## Actividades:
# 1. Configurar MLflow tracking server
# 2. Registrar experimentos con hiperparámetros
# 3. Guardar métricas y artefactos
# 4. Comparar runs en MLflow UI

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


# %% [markdown]
# ## RETO 1: Configurar MLflow
#
# Objetivo: Conectarse al tracking server y crear un experimento
#
# Pregunta:
# ¿Por qué es importante un tracking server centralizado
# en lugar de guardar métricas en archivos locales?

# %%
import mlflow
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator

# -----------------------------------------
# Configuración de MLflow
# -----------------------------------------

# URI del tracking server (contenedor MLflow)
mlflow.set_tracking_uri("http://mlflow:5000")

# Nombre del experimento
experiment_name = "/SECOP_Contratos_Prediccion"
mlflow.set_experiment(experiment_name)

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experimento activo: {experiment_name}")

# -----------------------------------------
# Respuesta conceptual:
#
# Un tracking server centralizado permite:
# - Centralizar métricas, parámetros y modelos de múltiples ejecuciones
# - Comparar experimentos entre diferentes usuarios o pipelines
# - Mantener trazabilidad y reproducibilidad de modelos
# - Evitar pérdida de información al trabajar en entornos distribuidos
# - Facilitar auditoría, monitoreo y despliegue en producción
#
# Guardar métricas en archivos locales no escala, no es colaborativo
# y dificulta la reproducibilidad en equipos de datos.
# -----------------------------------------


# %%
# Cargar datos listos para ML
df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")

df = (
    df.withColumnRenamed("valor_del_contrato_num", "label")
      .withColumnRenamed("features_pca", "features")
      .filter(col("label").isNotNull())
)

# Split train / test
train, test = df.randomSplit([0.8, 0.2], seed=42)

print(f"Train: {train.count():,}")
print(f"Test:  {test.count():,}")


# %%
# Evaluador base
evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

# %% [markdown]
# ## RETO 2: Registrar un Experimento Baseline
#
# Objetivo: Entrenar un modelo SIN regularización y registrarlo en MLflow

# %%
import mlflow
from pyspark.ml.regression import LinearRegression

# -----------------------------------------
# Registrar experimento baseline en MLflow
# -----------------------------------------

with mlflow.start_run(run_name="baseline_no_regularization"):

    # -----------------------------
    # Hiperparámetros del modelo
    # -----------------------------
    reg_param = 0.0
    elastic_param = 0.0
    max_iter = 100

    # Log de hiperparámetros
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("regParam", reg_param)
    mlflow.log_param("elasticNetParam", elastic_param)
    mlflow.log_param("maxIter", max_iter)

    # -----------------------------
    # Entrenamiento del modelo
    # -----------------------------
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=max_iter,
        regParam=reg_param,
        elasticNetParam=elastic_param
    )

    model = lr.fit(train)

    # -----------------------------
    # Evaluación
    # -----------------------------
    predictions = model.transform(test)
    rmse = evaluator.evaluate(predictions)

    # Log de métricas
    mlflow.log_metric("rmse", rmse)

    # -----------------------------
    # Guardar modelo como artefacto
    # -----------------------------
    mlflow.spark.log_model(model, artifact_path="model")

    print(f"✓ Experimento baseline registrado")
    print(f"  RMSE Test: ${rmse:,.2f}")

# %% [markdown]
# ## RETO 3: Registrar Múltiples Experimentos
#
# Objetivo: Entrenar y registrar varios modelos con diferentes regularizaciones
# y compararlos en MLflow UI.

# %%
import mlflow
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# -----------------------------------------
# Evaluadores
# -----------------------------------------
evaluator_rmse = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse"
)

evaluator_mae = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="mae"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="r2"
)

# -----------------------------------------
# Configuraciones de experimentos
# -----------------------------------------
experiments = [
    {"name": "ridge_l2", "reg": 0.1, "elastic": 0.0, "type": "Ridge"},
    {"name": "lasso_l1", "reg": 0.1, "elastic": 1.0, "type": "Lasso"},
    {"name": "elasticnet", "reg": 0.1, "elastic": 0.5, "type": "ElasticNet"},
]

max_iter = 100

# -----------------------------------------
# Ejecutar experimentos
# -----------------------------------------
for exp in experiments:

    with mlflow.start_run(run_name=exp["name"]):

        print(f"\nEntrenando modelo: {exp['type']}")

        # -----------------------------
        # Log de parámetros
        # -----------------------------
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("regularization_type", exp["type"])
        mlflow.log_param("regParam", exp["reg"])
        mlflow.log_param("elasticNetParam", exp["elastic"])
        mlflow.log_param("maxIter", max_iter)

        # -----------------------------
        # Entrenamiento
        # -----------------------------
        lr = LinearRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=max_iter,
            regParam=exp["reg"],
            elasticNetParam=exp["elastic"]
        )

        model = lr.fit(train)

        # -----------------------------
        # Evaluación
        # -----------------------------
        predictions = model.transform(test)

        rmse = evaluator_rmse.evaluate(predictions)
        mae = evaluator_mae.evaluate(predictions)
        r2 = evaluator_r2.evaluate(predictions)

        # -----------------------------
        # Log de métricas
        # -----------------------------
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # -----------------------------
        # Guardar modelo
        # -----------------------------
        mlflow.spark.log_model(model, artifact_path="model")

        # -----------------------------
        # Output informativo
        # -----------------------------
        print(f"✓ {exp['type']} registrado en MLflow")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  MAE : ${mae:,.2f}")
        print(f"  R²  : {r2:.4f}")


# %% [markdown]
# ## RETO 4: Explorar MLflow UI
#
# **URL de MLflow UI**:
# 👉 http://localhost:5000
#
# **Pasos realizados**:
# 1. Se accedió a la interfaz web de MLflow
# 2. Se seleccionó el experimento: `/SECOP_Contratos_Prediccion`
# 3. Se compararon los runs registrados (Baseline, Ridge, Lasso, ElasticNet)
# 4. Se ordenaron los resultados por la métrica **RMSE**
# 5. Se revisaron parámetros, métricas y artefactos de cada run
#
# ---
#
# ### Resultados Observados
#
# **Mejor modelo en MLflow UI**:
# - Tipo de modelo: ___________________________
# - Regularización: ___________________________
#
# **RMSE del mejor modelo**:
# - RMSE Test: $____________________
#
# ---
#
# ### Análisis
#
# - ¿Existe correlación entre regularización y rendimiento?
#   - ☐ Sí
#   - ☐ No
#
# **Observaciones**:
# - La regularización ayudó a:
#   - ☐ Reducir overfitting
#   - ☐ Mejorar generalización
#   - ☐ No tuvo impacto significativo
#
# - Comparando modelos:
#   - Ridge (L2): ______________________________
#   - Lasso (L1): ______________________________
#   - ElasticNet: ______________________________
#
# ---
#
# ### Comunicación con el equipo
#
# **¿Cómo compartir estos resultados?**
# - ☐ Enlace directo al experimento en MLflow UI
# - ☐ Screenshot comparativo de métricas
# - ☐ Exportar métricas a reporte (PDF / PPT)
# - ☐ Registrar conclusiones en documentación técnica
#
# **Recomendación final**:
# _______________________________________________________
#
# ---
#
# ✔️ Conclusión:
# MLflow permite comparar modelos de forma objetiva,
# reproducible y auditable, facilitando la toma de decisiones
# y el trabajo colaborativo en equipos de datos.

# %%
# RETO 5: Agregar Artefactos Personalizados

import mlflow
import mlflow.spark
import matplotlib.pyplot as plt
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

with mlflow.start_run(run_name="model_with_artifacts"):

    # -----------------------------
    # 1. Entrenar modelo (ejemplo)
    # -----------------------------
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        regParam=0.1,
        elasticNetParam=0.0
    )

    model = lr.fit(train)

    # -----------------------------
    # 2. Evaluación
    # -----------------------------
    predictions = model.transform(test)

    evaluator_rmse = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse"
    )
    evaluator_mae = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="mae"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="r2"
    )

    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    # -----------------------------
    # 3. Log de métricas
    # -----------------------------
    mlflow.log_param("regParam", 0.1)
    mlflow.log_param("elasticNetParam", 0.0)
    mlflow.log_param("maxIter", 100)
    mlflow.log_param("model_type", "LinearRegression")

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # -----------------------------
    # 4. Artefacto: Reporte de texto
    # -----------------------------
    report = f"""
    REPORTE DEL MODELO
    ==================
    Modelo: Regresión Lineal (Ridge)

    Métricas:
    - RMSE: ${rmse:,.2f}
    - MAE:  ${mae:,.2f}
    - R²:   {r2:.4f}

    Observación:
    Este modelo incluye regularización L2 para reducir overfitting
    y mejorar la generalización en datos no vistos.
    """

    mlflow.log_text(report, "model_report.txt")

    # -----------------------------
    # 5. (Bonus) Artefacto gráfico
    # -----------------------------
    # Convertir muestra a pandas para graficar
    pdf = predictions.select("label", "prediction").sample(0.1, seed=42).toPandas()

    plt.figure(figsize=(6, 6))
    plt.scatter(pdf["label"], pdf["prediction"], alpha=0.5)
    plt.plot(
        [pdf["label"].min(), pdf["label"].max()],
        [pdf["label"].min(), pdf["label"].max()],
        "r--"
    )
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Predicho")
    plt.title("Predicciones vs Valores Reales")
    plt.grid(True)

    plot_path = "/tmp/predicciones_vs_reales.png"
    plt.savefig(plot_path)
    plt.close()

    mlflow.log_artifact(plot_path)

    # -----------------------------
    # 6. Guardar modelo
    # -----------------------------
    mlflow.spark.log_model(model, "model")

    print(f"Run registrado con RMSE = ${rmse:,.2f}")


# %% [markdown]
# ## Preguntas de Reflexión
#
# 1. **¿Qué ventajas tiene MLflow sobre guardar métricas en archivos CSV?**
#
# MLflow ofrece trazabilidad completa de los experimentos, permitiendo
# comparar modelos, parámetros, métricas y artefactos en un solo lugar.
# A diferencia de archivos CSV, MLflow:
# - Centraliza los experimentos (multiusuario y multi-entorno)
# - Mantiene el historial completo de ejecuciones
# - Permite reproducibilidad exacta de modelos
# - Facilita la comparación visual en la UI
# - Gestiona modelos, métricas y artefactos de forma estructurada
#
#
# 2. **¿Cómo implementarías MLflow en un proyecto de equipo?**
#
# Implementaría MLflow como un servicio centralizado accesible para todo el equipo:
# - Un tracking server compartido (Docker / Kubernetes)
# - Convenciones de nombres para experimentos y runs
# - Registro obligatorio de parámetros, métricas y modelos
# - Integración con pipelines de CI/CD
# - Uso del Model Registry para controlar versiones
# - Roles claros (Data Scientist, Reviewer, Product Owner)
#
#
# 3. **¿Qué artefactos adicionales guardarías además del modelo?**
#
# Además del modelo entrenado, guardaría:
# - Reportes de métricas (TXT / JSON)
# - Gráficos (residuos, ROC, predicción vs real)
# - Esquema de features
# - Versiones de datasets o hashes
# - Código del entrenamiento
# - Configuración del entorno (requirements.txt / conda.yaml)
#
#
# 4. **¿Cómo automatizarías el registro de experimentos?**
#
# Automatizaría el registro integrando MLflow en:
# - Pipelines de entrenamiento (scripts o notebooks)
# - Jobs programados (Airflow / Prefect / cron)
# - CI/CD (GitHub Actions, GitLab CI)
# - Uso de templates de entrenamiento con MLflow incluido por defecto
#
# De esta forma, cada entrenamiento queda registrado automáticamente
# sin depender de acciones manuales.


# %%
print("\n" + "="*60)
print("RESUMEN MLFLOW TRACKING")
print("="*60)
print("Verifica que hayas completado:")
print("  [ ] Configurado MLflow tracking server")
print("  [ ] Registrado experimento baseline")
print("  [ ] Registrado al menos 3 experimentos adicionales")
print("  [ ] Explorado MLflow UI")
print("  [ ] Comparado métricas entre runs")
print(f"  [ ] Accede a MLflow UI: http://localhost:5000")
print("="*60)

# %%
spark.stop()
