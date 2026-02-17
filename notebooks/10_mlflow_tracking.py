# ============================================================
# NOTEBOOK 10: MLflow Tracking
# ====================================================

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
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Inicialización de Spark
# ------------------------------------------------------------

builder = (
    SparkSession.builder
    .appName("SECOP_Feature_Engineering")
    .master("spark://spark-master:7077")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()

print("✓ Spark inicializado correctamente")
print(f"  - Spark Version : {spark.version}")
print(f"  - Spark Master  : {spark.sparkContext.master}")


# ------------------------------------------------------------
# RETO 1: CONFIGURAR MLFLOW
# ------------------------------------------------------------

print("\n" + "="*70)
print("RETO 1: CONFIGURACIÓN DE MLFLOW")
print("="*70)

print(
    "Objetivo:\n"
    "- Conectarse a un MLflow Tracking Server centralizado\n"
    "- Crear y activar un experimento para registrar ejecuciones\n"
)

print(
    "Pregunta clave:\n"
    "¿Por qué es importante usar un tracking server centralizado\n"
    "en lugar de guardar métricas en archivos locales?\n"
)

print(
    "Respuesta:\n"
    "Un tracking server centralizado permite:\n"
    "- Centralizar métricas, parámetros y artefactos de múltiples ejecuciones\n"
    "- Comparar experimentos entre diferentes usuarios y pipelines\n"
    "- Mantener trazabilidad completa de modelos entrenados\n"
    "- Garantizar reproducibilidad en entornos distribuidos\n"
    "- Facilitar auditoría, monitoreo y despliegue en producción\n\n"
    "En contraste, guardar métricas en archivos locales:\n"
    "- No escala en equipos de trabajo\n"
    "- No es colaborativo\n"
    "- Se pierde fácilmente información histórica\n"
    "- Dificulta la reproducibilidad y el gobierno del modelo\n"
)

# ------------------------------------------------------------
# CONFIGURACIÓN DE MLFLOW
# ------------------------------------------------------------

# URI del tracking server (contenedor MLflow)
mlflow.set_tracking_uri("http://mlflow:5000")

# Nombre del experimento
experiment_name = "/SECOP_Contratos_Prediccion"
mlflow.set_experiment(experiment_name)

print("Configuración de MLflow completada:")
print(f"  • Tracking URI: {mlflow.get_tracking_uri()}")
print(f"  • Experimento activo: {experiment_name}")

print("="*70)

# ------------------------------------------------------------
# CARGA Y PREPARACIÓN DE DATOS
# ------------------------------------------------------------

print("\nCargando datos preparados para Machine Learning...")

df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")

df = (
    df.withColumnRenamed("valor_del_contrato_num", "label")
      .withColumnRenamed("features_pca", "features")
      .filter(col("label").isNotNull())
)

print("✓ Datos cargados y columnas normalizadas")
print(f"  Columnas disponibles: {df.columns}")

# Split train / test
train, test = df.randomSplit([0.8, 0.2], seed=42)

print("\nSplit Train / Test ejecutado:")
print(f"  • Train: {train.count():,} registros (80%)")
print(f"  • Test:  {test.count():,} registros (20%)")

# ------------------------------------------------------------
# EVALUADOR BASE
# ------------------------------------------------------------

evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

print("\nEvaluador configurado:")
print("  • Métrica: RMSE")

print(
    "\nJustificación de la métrica RMSE:\n"
    "- Penaliza más los errores grandes\n"
    "- Es adecuada cuando errores grandes son costosos\n"
    "  (ej. sobreestimar o subestimar contratos de alto valor)\n"
    "- Mantiene las mismas unidades del valor del contrato,\n"
    "  facilitando interpretación para negocio"
)

print("\n" + "="*70)
print("CONFIGURACIÓN INICIAL COMPLETADA")
print("="*70)


# ------------------------------------------------------------
# RETO 2: REGISTRAR UN EXPERIMENTO BASELINE EN MLFLOW
# ------------------------------------------------------------

print("\n" + "="*70)
print("RETO 2: REGISTRAR EXPERIMENTO BASELINE")
print("="*70)

print(
    "Objetivo:\n"
    "- Entrenar un modelo base SIN regularización\n"
    "- Registrarlo en MLflow como punto de referencia (baseline)\n"
)

print(
    "Justificación:\n"
    "- Un modelo baseline permite comparar mejoras posteriores\n"
    "- Ayuda a identificar si la regularización realmente aporta valor\n"
    "- Sirve como referencia mínima aceptable de desempeño\n"
)

# ------------------------------------------------------------
# REGISTRO DEL EXPERIMENTO BASELINE
# ------------------------------------------------------------

with mlflow.start_run(run_name="baseline_no_regularization"):

    # Hiperparámetros
    reg_param = 0.0
    elastic_param = 0.0
    max_iter = 100

    print("\nRegistrando hiperparámetros del modelo baseline...")
    
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("regParam", reg_param)
    mlflow.log_param("elasticNetParam", elastic_param)
    mlflow.log_param("maxIter", max_iter)

    # Entrenamiento
    print("Entrenando modelo baseline (sin regularización)...")

    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=max_iter,
        regParam=reg_param,
        elasticNetParam=elastic_param
    )

    model = lr.fit(train)

    # Evaluación
    print("Evaluando modelo baseline en test...")

    predictions = model.transform(test)
    rmse = evaluator.evaluate(predictions)

    mlflow.log_metric("rmse", rmse)

    # Guardar modelo
    mlflow.spark.log_model(model, artifact_path="model")

    print("\n✓ Experimento baseline registrado en MLflow")
    print(f"  RMSE Test: ${rmse:,.2f}")

print("\n" + "="*70)
print("BASELINE COMPLETADO")
print("="*70)


# ------------------------------------------------------------
# RETO 3: REGISTRAR MÚLTIPLES EXPERIMENTOS CON REGULARIZACIÓN
# ------------------------------------------------------------

print("\n" + "="*70)
print("RETO 3: REGISTRAR MÚLTIPLES EXPERIMENTOS")
print("="*70)

print(
    "Objetivo:\n"
    "- Entrenar varios modelos con diferentes tipos de regularización\n"
    "- Registrar métricas comparables en MLflow\n"
    "- Analizar desempeño relativo en la UI de MLflow\n"
)

print(
    "Estrategia:\n"
    "- Probar Ridge (L2), Lasso (L1) y ElasticNet\n"
    "- Mantener maxIter constante para comparabilidad\n"
    "- Evaluar con RMSE, MAE y R²\n"
)

# ------------------------------------------------------------
# EVALUADORES
# ------------------------------------------------------------

evaluator_rmse = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse"
)

evaluator_mae = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="mae"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="r2"
)

# ------------------------------------------------------------
# CONFIGURACIONES DE EXPERIMENTOS
# ------------------------------------------------------------

experiments = [
    {"name": "ridge_l2", "reg": 0.1, "elastic": 0.0, "type": "Ridge (L2)"},
    {"name": "lasso_l1", "reg": 0.1, "elastic": 1.0, "type": "Lasso (L1)"},
    {"name": "elasticnet", "reg": 0.1, "elastic": 0.5, "type": "ElasticNet"},
]

max_iter = 100

# ------------------------------------------------------------
# EJECUCIÓN DE EXPERIMENTOS
# ------------------------------------------------------------

for exp in experiments:

    with mlflow.start_run(run_name=exp["name"]):

        print(f"\nEntrenando modelo: {exp['type']}")

        # Log de parámetros
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("regularization_type", exp["type"])
        mlflow.log_param("regParam", exp["reg"])
        mlflow.log_param("elasticNetParam", exp["elastic"])
        mlflow.log_param("maxIter", max_iter)

        # Entrenamiento
        lr = LinearRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=max_iter,
            regParam=exp["reg"],
            elasticNetParam=exp["elastic"]
        )

        model = lr.fit(train)

        # Evaluación
        predictions = model.transform(test)

        rmse = evaluator_rmse.evaluate(predictions)
        mae = evaluator_mae.evaluate(predictions)
        r2 = evaluator_r2.evaluate(predictions)

        # Log de métricas
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Guardar modelo
        mlflow.spark.log_model(model, artifact_path="model")

        # Output informativo
        print(f"✓ Modelo {exp['type']} registrado en MLflow")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  MAE : ${mae:,.2f}")
        print(f"  R²  : {r2:.4f}")

print("\n" + "="*70)
print("REGISTRO DE EXPERIMENTOS COMPLETADO")
print("="*70)



print("\n" + "="*70)
print("RETO 4: EXPLORAR MLFLOW UI Y ANALIZAR RESULTADOS")
print("="*70)

print(
    "Objetivo:\n"
    "- Explorar la interfaz de MLflow UI\n"
    "- Comparar modelos entrenados y registrados\n"
    "- Identificar el mejor modelo según métricas objetivas\n"
)

print(
    "\nURL de MLflow UI:\n"
    "👉 http://localhost:5000\n"
)

print(
    "Pasos realizados en MLflow UI:\n"
    "1. Acceso a la interfaz web de MLflow\n"
    "2. Selección del experimento: /SECOP_Contratos_Prediccion\n"
    "3. Revisión de runs registrados:\n"
    "   - baseline_no_regularization\n"
    "   - ridge_l2\n"
    "   - lasso_l1\n"
    "   - elasticnet\n"
    "4. Ordenamiento de resultados por la métrica RMSE\n"
    "5. Inspección de parámetros, métricas y artefactos por run\n"
)

print("\n" + "-"*70)
print("CONTEXTO DEL EXPERIMENTO")
print("-"*70)

print(
    "Entorno de ejecución:\n"
    "- Spark Version: 3.5.0\n"
    "- MLflow Tracking URI: http://mlflow:5000\n"
    "- Experimento creado automáticamente al no existir previamente\n"
)

print(
    "Datos utilizados:\n"
    "- Registros de entrenamiento: 47,660\n"
    "- Registros de prueba:        11,666\n"
)

print("\n" + "-"*70)
print("RESULTADOS OBSERVADOS EN MLFLOW UI")
print("-"*70)

print(
    "Resumen de métricas por modelo:\n"
    "\n"
    "Baseline (sin regularización):\n"
    "  - RMSE Test: $41,776,972,478.38\n"
    "\n"
    "Ridge (L2):\n"
    "  - RMSE Test: $41,776,972,478.38\n"
    "  - MAE:       $1,009,918,363.20\n"
    "  - R²:        0.0012\n"
    "\n"
    "Lasso (L1):\n"
    "  - RMSE Test: $41,776,972,478.38\n"
    "  - MAE:       $1,009,918,363.20\n"
    "  - R²:        0.0012\n"
    "\n"
    "ElasticNet:\n"
    "  - RMSE Test: $41,776,972,478.38\n"
    "  - MAE:       $1,009,918,363.20\n"
    "  - R²:        0.0012\n"
)

print("\n" + "-"*70)
print("MEJOR MODELO IDENTIFICADO")
print("-"*70)

print(
    "Resultado:\n"
    "- No se observan diferencias significativas entre los modelos\n"
    "- Todos presentan el mismo RMSE en el set de test\n"
)

print(
    "Conclusión técnica:\n"
    "- La regularización (L1, L2, ElasticNet) NO produjo mejoras\n"
    "  medibles en este experimento específico\n"
)

print("\n" + "-"*70)
print("ANÁLISIS")
print("-"*70)

print(
    "¿Existe correlación entre regularización y rendimiento?\n"
    "→ No\n"
)

print(
    "Observaciones clave:\n"
    "- La regularización no redujo el RMSE\n"
    "- No hubo mejora en R² ni en MAE\n"
    "- El modelo parece limitado por la calidad o expresividad de las features\n"
    "- El problema no es overfitting, sino baja capacidad explicativa\n"
)

print(
    "Comparación cualitativa:\n"
    "- Ridge (L2):       Comportamiento idéntico al baseline\n"
    "- Lasso (L1):       No eliminó variables relevantes de forma efectiva\n"
    "- ElasticNet:       Sin impacto adicional frente a L1/L2\n"
)

print("\n" + "-"*70)
print("COMUNICACIÓN CON EL EQUIPO")
print("-"*70)

print(
    "Formas recomendadas de compartir resultados:\n"
    "- Enlace directo al experimento en MLflow UI\n"
    "- Screenshot comparativo de métricas (tabla de runs)\n"
    "- Registro de conclusiones en documentación técnica\n"
    "- Insumo para comité de decisión analítica\n"
)

print(
    "\nRecomendación final:\n"
    "- No avanzar a producción con este modelo\n"
    "- Priorizar mejora de features (feature engineering)\n"
    "- Explorar modelos no lineales (árboles, boosting)\n"
    "- Evaluar transformación del target (log-scale)\n"
)

print("\n" + "="*70)
print("CONCLUSIÓN GENERAL")
print("="*70)

print(
    "MLflow permitió:\n"
    "- Comparar modelos de forma objetiva\n"
    "- Garantizar trazabilidad y reproducibilidad\n"
    "- Detectar rápidamente que la regularización no era el cuello de botella\n"
    "- Evitar decisiones subjetivas o basadas en intuición\n"
)

print(
    "\nEstado del proyecto:\n"
    "✔️ Experimentos correctamente registrados\n"
    "✔️ Resultados analizados y documentados\n"
    "✔️ Decisión informada para siguientes iteraciones\n"
)

print("="*70)

# ------------------------------------------------------------
# RETO 5: Agregar Artefactos Personalizados
# ------------------------------------------------------------

print("\n" + "="*70)
print("RETO 5: AGREGAR ARTEFACTOS PERSONALIZADOS EN MLFLOW")
print("="*70)

print(
    "Objetivo:\n"
    "- Registrar no solo métricas y parámetros\n"
    "- Agregar artefactos útiles para análisis, auditoría y comunicación\n"
    "- Enriquecer el experimento más allá del modelo entrenado\n"
)

with mlflow.start_run(run_name="model_with_artifacts"):

    print("\nIniciando run con artefactos personalizados...")

    # -------------------------------------------------
    # 1. Entrenamiento del modelo
    # -------------------------------------------------
    print("\nEntrenando modelo de Regresión Lineal con regularización L2 (Ridge)...")

    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        regParam=0.1,
        elasticNetParam=0.0
    )

    model = lr.fit(train)
    print("✓ Modelo entrenado correctamente")

    # -------------------------------------------------
    # 2. Evaluación del modelo
    # -------------------------------------------------
    print("\nEvaluando modelo en el set de test...")

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

    print(f"RMSE Test: ${rmse:,.2f}")
    print(f"MAE  Test: ${mae:,.2f}")
    print(f"R²   Test: {r2:.4f}")

    # -------------------------------------------------
    # 3. Log de parámetros y métricas
    # -------------------------------------------------
    print("\nRegistrando parámetros y métricas en MLflow...")

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("regularization", "Ridge (L2)")
    mlflow.log_param("regParam", 0.1)
    mlflow.log_param("elasticNetParam", 0.0)
    mlflow.log_param("maxIter", 100)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    print("✓ Parámetros y métricas registrados")

    # -------------------------------------------------
    # 4. Artefacto: Reporte de texto
    # -------------------------------------------------
    print("\nGenerando artefacto: reporte textual del modelo...")

    report = f"""
REPORTE DEL MODELO
==================
Modelo: Regresión Lineal (Ridge - L2)

Métricas en Test:
- RMSE: ${rmse:,.2f}
- MAE:  ${mae:,.2f}
- R²:   {r2:.4f}

Observaciones:
- Se utilizó regularización L2 para controlar la magnitud de los coeficientes
- No se observaron mejoras significativas frente al baseline
- El modelo presenta baja capacidad explicativa (R² cercano a 0)

Conclusión:
La regularización no es el principal cuello de botella.
Se recomienda mejorar features o probar modelos no lineales.
"""

    mlflow.log_text(report, "model_report.txt")
    print("✓ Reporte textual registrado como artefacto")

    # -------------------------------------------------
    # 5. Artefacto gráfico: Predicción vs Valor Real
    # -------------------------------------------------
    print("\nGenerando artefacto gráfico: predicciones vs valores reales...")

    pdf = (
        predictions
        .select("label", "prediction")
        .sample(0.1, seed=42)
        .toPandas()
    )

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
    print("✓ Gráfico registrado como artefacto")

    # -------------------------------------------------
    # 6. Guardar modelo
    # -------------------------------------------------
    mlflow.spark.log_model(model, "model")
    print("✓ Modelo guardado como artefacto MLflow")

    print(f"\nRun registrado exitosamente con RMSE = ${rmse:,.2f}")

print("\n" + "="*70)
print("PREGUNTAS DE REFLEXIÓN – RESPUESTAS")
print("="*70)

print(
    "1. ¿Qué ventajas tiene MLflow frente a guardar métricas en CSV?\n"
    "- Centraliza experimentos, métricas, parámetros y modelos\n"
    "- Facilita comparación visual entre runs\n"
    "- Garantiza reproducibilidad y trazabilidad\n"
    "- Escala a equipos y entornos distribuidos\n"
)

print(
    "2. ¿Cómo implementar MLflow en un proyecto de equipo?\n"
    "- Tracking server centralizado (Docker / Kubernetes)\n"
    "- Convenciones claras de nombres\n"
    "- Registro obligatorio de métricas y modelos\n"
    "- Integración con CI/CD\n"
    "- Uso de Model Registry para control de versiones\n"
)

print(
    "3. ¿Qué artefactos adicionales son recomendables?\n"
    "- Reportes de métricas\n"
    "- Gráficos (residuos, ROC, predicción vs real)\n"
    "- Esquema de features\n"
    "- Versiones de datasets\n"
    "- Código de entrenamiento\n"
)

print(
    "4. ¿Cómo automatizar el registro de experimentos?\n"
    "- Integrando MLflow en scripts y notebooks\n"
    "- Jobs programados (Airflow, Prefect)\n"
    "- Pipelines CI/CD\n"
    "- Templates de entrenamiento con MLflow por defecto\n"
)

print("\n" + "="*60)
print("RESUMEN MLFLOW TRACKING")
print("="*60)
print("Verifica que hayas completado:")
print("  [x] Configuración del tracking server")
print("  [x] Registro de parámetros y métricas")
print("  [x] Registro de modelos")
print("  [x] Registro de artefactos (texto y gráficos)")
print("  [x] Exploración y comparación en MLflow UI")
print("  👉 Accede a MLflow UI: http://localhost:5000")
print("="*60)

spark.stop()
print("✓ SparkSession detenida correctamente")
