# %% [markdown]
# # Notebook 08: Validación Cruzada (K-Fold)
#
# **Sección 15 - Tuning**: Cross-validation para evitar overfitting
#
# **Objetivo**: Implementar K-Fold Cross-Validation
#
# ## Conceptos clave:
# - Divide datos en K folds (subconjuntos)
# - Entrena K veces, usando diferente fold como validación
# - Promedia métricas para obtener estimación robusta
#
# ## Actividades:
# 1. Entender el concepto de K-Fold
# 2. Configurar CrossValidator en Spark ML
# 3. Combinar con ParamGrid para búsqueda de hiperparámetros
# 4. Analizar resultados

# %%
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import when, col
from pyspark.sql.functions import abs as spark_abs, col
from pyspark.ml.feature import StandardScaler, PCA, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from delta import configure_spark_with_delta_pip
from pyspark.ml.feature import VectorAssembler
import numpy as np

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
# Cargar datos
df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")
df = df.withColumnRenamed("valor_del_contrato_num", "label") \
       .withColumnRenamed("features_pca", "features") \
       .filter(col("label").isNotNull())

train, test = df.randomSplit([0.8, 0.2], seed=42)

print(f"Train: {train.count():,}")
print(f"Test: {test.count():,}")

# %%
# RETO 1: Entender K-Fold Cross-Validation
#
# Supongamos K = 5

# 1. ¿En cuántos subconjuntos se dividen los datos de train?
# → En 5 subconjuntos (folds) del mismo tamaño (aprox.).

# 2. ¿Cuántos modelos se entrenan en total?
# → Se entrenan 5 modelos.
#   En cada iteración, uno de los folds actúa como validación
#   y los otros 4 como entrenamiento.

# 3. ¿Qué porcentaje de datos se usa para validación en cada iteración?
# → 1/K del total de los datos de entrenamiento.
# → Para K=5: 20% validación y 80% entrenamiento en cada iteración.

# 4. ¿Qué métrica se reporta al final?
# → El promedio (y a veces la desviación estándar) de la métrica
#   evaluada en cada fold (por ejemplo: RMSE promedio).

# ¿Por qué K-Fold es mejor que un simple train/test split?
#
# - Reduce la dependencia de una sola partición aleatoria
# - Usa todos los datos tanto para entrenamiento como para validación
# - Produce métricas más estables y confiables
# - Detecta mejor overfitting y underfitting
# - Es especialmente útil cuando el dataset no es muy grande


# %%
# RETO 2: Crear el Modelo Base y Evaluador
#
# Objetivo:
# - Definir un modelo base de regresión lineal
# - Definir un evaluador para comparar modelos

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Modelo base de Regresión Lineal (sin regularización explícita)
lr = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100
)

print("✓ Modelo base LinearRegression creado")
print(f"  featuresCol: {lr.getFeaturesCol()}")
print(f"  labelCol: {lr.getLabelCol()}")
print(f"  maxIter: {lr.getMaxIter()}")

# Evaluador del modelo
# Usamos RMSE porque penaliza más los errores grandes
evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

print("✓ Evaluador configurado")
print("  Métrica: RMSE")

# Reflexión (comentario para el notebook):
#
# - RMSE es útil cuando los errores grandes son costosos (ej. contratos de alto valor)
# - MAE podría usarse si se quiere tratar todos los errores por igual
# - R² es complementario, pero no siempre suficiente para comparar modelos

# %%
# RETO 3: Construir el ParamGrid
#
# Objetivo:
# - Definir combinaciones de hiperparámetros para Cross-Validation
# - Explorar distintos niveles y tipos de regularización

from pyspark.ml.tuning import ParamGridBuilder

# Definición del grid de hiperparámetros
param_grid = (
    ParamGridBuilder()
    # λ (lambda): fuerza de regularización
    .addGrid(lr.regParam, [0.01, 0.1, 1.0])
    
    # α (alpha): tipo de regularización
    # 0.0 = Ridge (L2)
    # 0.5 = ElasticNet
    # 1.0 = Lasso (L1)
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
    
    .build()
)

# Número de combinaciones
num_combinations = len(param_grid)
print(f"Combinaciones en el grid: {num_combinations}")

# Si usamos K-Fold Cross-Validation
K = 5
total_models = num_combinations * K
print(f"Total de modelos a entrenar: {total_models}")

# Explicación (comentario para el notebook):
#
# - 3 valores de regParam × 3 valores de elasticNetParam = 9 combinaciones
# - Con K = 5 folds:
#   👉 9 × 5 = 45 modelos entrenados en total
#
# Esto explica por qué Cross-Validation puede ser computacionalmente costoso


# %%
# RETO 4: Configurar CrossValidator
#
# Objetivo:
# - Ensamblar el proceso de Cross-Validation
# - Entrenar múltiples modelos automáticamente
# - Seleccionar el mejor según la métrica (RMSE)

from pyspark.ml.tuning import CrossValidator

# Elección de K
# K = 5 → balance clásico entre robustez y costo computacional
K = 5

# Configuración del CrossValidator
crossval = CrossValidator(
    estimator=lr,                     # Modelo base
    estimatorParamMaps=param_grid,    # Grid de hiperparámetros
    evaluator=evaluator,              # Métrica (RMSE)
    numFolds=K,                       # K-Fold Cross-Validation
    seed=42                           # Reproducibilidad
)

print(f"✓ Cross-Validation configurado con K={K} folds")
print(f"✓ Combinaciones de hiperparámetros: {len(param_grid)}")
print(f"✓ Total de modelos a entrenar: {len(param_grid) * K}")

# Explicación (comentario para el notebook):
#
# - Usamos K=5 porque:
#   ✔️ Reduce la varianza del error
#   ✔️ No es tan costoso como K=10
#   ✔️ Es estándar en problemas reales
#
# - Total de modelos entrenados:
#   combinaciones × K = {len(param_grid)} × {K}
#
# ⚠️ En datasets muy grandes, este número puede crecer rápidamente


# %%
# RETO 5: Ejecutar Cross-Validation y Analizar Resultados
#
# Objetivo:
# - Ejecutar Cross-Validation
# - Analizar métricas promedio
# - Identificar el mejor modelo
# - Evaluarlo en el set de test

print("Entrenando modelos con Cross-Validation...")
cv_model = crossval.fit(train)
print("✓ Cross-validation completada")

# %%
# Analizar métricas promedio (RMSE) por configuración
avg_metrics = cv_model.avgMetrics

# Índice del mejor modelo (menor RMSE)
best_metric_idx = avg_metrics.index(min(avg_metrics))

print("\n=== MÉTRICAS PROMEDIO POR CONFIGURACIÓN (RMSE) ===")
for i, metric in enumerate(avg_metrics):
    params = param_grid[i]
    reg = params[lr.regParam]
    elastic = params[lr.elasticNetParam]

    marker = " <-- MEJOR MODELO" if i == best_metric_idx else ""
    print(
        f"Config {i+1:02d} | "
        f"λ={reg:<5.2f} | "
        f"α={elastic:<3.1f} | "
        f"RMSE CV={metric:,.2f}"
        f"{marker}"
    )

# %%
# Obtener el mejor modelo encontrado por Cross-Validation
best_model = cv_model.bestModel

print("\n=== MEJOR MODELO SELECCIONADO ===")
print(f"regParam (λ):        {best_model.getRegParam()}")
print(f"elasticNetParam (α): {best_model.getElasticNetParam()}")

# %%
# Evaluar el mejor modelo en el set de test
predictions = best_model.transform(test)
rmse_test = evaluator.evaluate(predictions)

print("\n=== EVALUACIÓN FINAL EN TEST ===")
print(f"RMSE Test: ${rmse_test:,.2f}")

# %%
# Comentario conceptual (para el notebook):
#
# - avgMetrics contiene el RMSE promedio de cada combinación
# - El mejor modelo NO se elige por train, sino por validación cruzada
# - Esto reduce overfitting y mejora generalización
#
# ✔️ El modelo seleccionado es el que minimiza el RMSE promedio en CV


# %%
# RETO 6: Comparar Cross-Validation vs Simple Split
#
# Objetivo:
# - Entrenar un modelo SIN Cross-Validation
# - Comparar su desempeño contra el modelo seleccionado con CV
# - Analizar cuál enfoque es más confiable

from pyspark.ml.regression import LinearRegression

print("\nEntrenando modelo SIMPLE (sin Cross-Validation)...")

# Modelo simple usando los mismos hiperparámetros del mejor modelo CV
lr_simple = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=best_model.getRegParam(),
    elasticNetParam=best_model.getElasticNetParam()
)

# Entrenamiento
model_simple = lr_simple.fit(train)

# Evaluación en test
rmse_simple = evaluator.evaluate(model_simple.transform(test))

# Comparación
print("\n=== COMPARACIÓN CV vs SIMPLE SPLIT ===")
print(f"RMSE con Cross-Validation: ${rmse_test:,.2f}")
print(f"RMSE sin Cross-Validation: ${rmse_simple:,.2f}")
print(f"Diferencia absoluta:       ${abs(rmse_test - rmse_simple):,.2f}")

# %%
# Interpretación (completa como comentario en tu notebook):
#
# - El modelo con Cross-Validation es más confiable porque:
#   • Evalúa múltiples particiones del train
#   • Reduce la dependencia de un solo split aleatorio
#   • Produce una métrica más estable y robusta
#
# - El modelo sin CV puede:
#   • Verse afectado por la casualidad del split
#   • Sobreestimar o subestimar el rendimiento real
#
# Conclusión:
# ✔️ Cross-Validation ofrece una mejor estimación del desempeño real del modelo


# %%
# RETO BONUS: Experimentar con diferentes valores de K (Cross-Validation)
#
# Objetivo:
# - Comparar K=3, K=5 y K=10
# - Observar impacto en RMSE y tiempo de ejecución
# - Entender el trade-off entre robustez y costo computacional

from pyspark.ml.tuning import CrossValidator
import time

print("\n=== EXPERIMENTO CON DIFERENTES VALORES DE K ===")

resultados_k = []

for k in [3, 5, 10]:
    print(f"\nEjecutando Cross-Validation con K={k} folds...")

    cv_temp = CrossValidator(
        estimator=lr,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=k,
        seed=42
    )

    start_time = time.time()
    cv_temp_model = cv_temp.fit(train)
    elapsed_time = time.time() - start_time

    best_rmse = min(cv_temp_model.avgMetrics)

    resultados_k.append({
        "K": k,
        "best_rmse": best_rmse,
        "time_seconds": elapsed_time
    })

    print(
        f"K={k:2d} | "
        f"Mejor RMSE: ${best_rmse:,.2f} | "
        f"Tiempo: {elapsed_time:.1f} segundos"
    )

# %%
# Resumen comparativo
print("\n=== RESUMEN COMPARATIVO POR K ===")
for r in resultados_k:
    print(
        f"K={r['K']:2d} | "
        f"RMSE: ${r['best_rmse']:,.2f} | "
        f"Tiempo: {r['time_seconds']:.1f}s"
    )

# %%
# Interpretación (completa como comentario en tu notebook):
#
# - K pequeño (ej. 3):
#   • Más rápido
#   • Métrica menos estable
#
# - K intermedio (5):
#   • Buen balance entre costo y robustez
#   • Opción más común en práctica
#
# - K grande (10):
#   • Métrica más robusta
#   • Mucho más costoso computacionalmente
#
# Conclusión:
# ❌ Más folds NO siempre es mejor
# ✔️ El valor óptimo de K depende del tamaño del dataset y del costo computacional

# %%
# Guardar el mejor modelo entrenado con Cross-Validation
model_path = "/opt/spark-data/processed/cv_best_model"

# Guardar modelo
best_model.save(model_path)

print(f"✓ Modelo guardado correctamente en: {model_path}")


# %%
print("\n" + "="*60)
print("RESUMEN VALIDACIÓN CRUZADA")
print("="*60)
print("Verifica que hayas completado:")
print("  [ ] Entendido el concepto de K-Fold")
print("  [ ] Configurado ParamGrid con hiperparámetros")
print("  [ ] Ejecutado CrossValidator")
print("  [ ] Identificado el mejor modelo")
print("  [ ] Comparado con entrenamiento simple")
print("="*60)

# %%
spark.stop()
