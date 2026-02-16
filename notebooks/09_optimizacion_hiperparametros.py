# %% [markdown]
# # Notebook 09: Optimización de Hiperparámetros
#
# **Sección 15**: Grid Search y Train-Validation Split
#
# **Objetivo**: Encontrar la mejor combinación de hiperparámetros
#
# ## Estrategias:
# - **Grid Search + CV**: Búsqueda exhaustiva con cross-validation
# - **Train-Validation Split**: Alternativa más rápida (un solo split)
#
# ## Actividades:
# 1. Implementar Grid Search exhaustivo
# 2. Implementar Train-Validation Split
# 3. Comparar ambas estrategias
# 4. Seleccionar el mejor modelo global

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
# Modelo base y evaluador
lr = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100
)

evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

# %%
# RETO 1: Diseñar el Grid de Hiperparámetros

grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [
        0.01,   # Regularización baja
        0.1,    # Regularización media
        1.0     # Regularización fuerte
    ]) \
    .addGrid(lr.elasticNetParam, [
        0.0,    # Ridge (L2)
        0.5,    # ElasticNet
        1.0     # Lasso (L1)
    ]) \
    .addGrid(lr.maxIter, [
        50,
        100,
        200
    ]) \
    .build()

print(f"Combinaciones totales en el grid: {len(grid)}")

# %%
# Respuestas conceptuales (como comentarios):
#
# ¿Por qué usar escala logarítmica para regParam?
# → Porque el efecto de la regularización no es lineal.
#   Pequeños cambios en valores bajos (0.01 → 0.1) tienen
#   mucho más impacto que cambios en valores altos.
#
# ¿Cuántas combinaciones hay?
# → 3 (regParam) × 3 (elasticNetParam) × 3 (maxIter) = 27
#
# ¿Cuántos modelos se entrenan con K=3?
# → 27 × 3 = 81 modelos


# %%
import time
from pyspark.ml.tuning import CrossValidator

# RETO 2: Grid Search con Cross-Validation

# Configurar CrossValidator
cv_grid = CrossValidator(
    estimator=lr,
    estimatorParamMaps=grid,
    evaluator=evaluator,
    numFolds=3,     # K=3 → balance entre robustez y tiempo
    seed=42
)

# Ejecutar Grid Search + CV
print("Entrenando Grid Search + Cross-Validation (K=3)...")
start_time = time.time()

cv_grid_model = cv_grid.fit(train)

grid_time = time.time() - start_time
print(f"✓ Grid Search + CV completado en {grid_time:.2f} segundos")

# %%
# Obtener y evaluar el mejor modelo
best_grid_model = cv_grid_model.bestModel

predictions_grid = best_grid_model.transform(test)
rmse_grid = evaluator.evaluate(predictions_grid)

print("\n=== MEJOR MODELO (Grid Search + CV) ===")
print(f"regParam:         {best_grid_model.getRegParam()}")
print(f"elasticNetParam: {best_grid_model.getElasticNetParam()}")
print(f"maxIter:         {best_grid_model.getMaxIter()}")
print(f"RMSE en Test:    ${rmse_grid:,.2f}")

# %%
# Respuesta conceptual (como comentario):
#
# ¿Por qué K=3 y no K=5?
# → Porque el Grid Search ya es computacionalmente costoso:
#   - Modelos entrenados = combinaciones × K
#   - Con 27 combinaciones:
#       K=3  → 81 modelos
#       K=5  → 135 modelos
#   K=3 ofrece buen equilibrio entre estabilidad y tiempo de ejecución,
#   especialmente en datasets medianos o grandes.


# %%
import time
from pyspark.ml.tuning import TrainValidationSplit

# RETO 3: Train-Validation Split

# Configurar TrainValidationSplit
tvs = TrainValidationSplit(
    estimator=lr,
    estimatorParamMaps=grid,
    evaluator=evaluator,
    trainRatio=0.8,   # 80% train / 20% validation
    seed=42
)

# Ejecutar Train-Validation Split
print("Entrenando con Train-Validation Split (80/20)...")
start_time = time.time()

tvs_model = tvs.fit(train)

tvs_time = time.time() - start_time
print(f"✓ Train-Validation Split completado en {tvs_time:.2f} segundos")

# %%
# Obtener y evaluar el mejor modelo
best_tvs_model = tvs_model.bestModel

predictions_tvs = best_tvs_model.transform(test)
rmse_tvs = evaluator.evaluate(predictions_tvs)

print("\n=== MEJOR MODELO (Train-Validation Split) ===")
print(f"regParam:         {best_tvs_model.getRegParam()}")
print(f"elasticNetParam: {best_tvs_model.getElasticNetParam()}")
print(f"maxIter:         {best_tvs_model.getMaxIter()}")
print(f"RMSE en Test:    ${rmse_tvs:,.2f}")

# %%
# Comparación conceptual (como comentario):
#
# - TrainValidationSplit:
#   ✔ Mucho más rápido (cada combinación se entrena una sola vez)
#   ✘ Menos robusto (depende de un único split)
#
# - Cross-Validation:
#   ✔ Métrica más estable y confiable
#   ✘ Mucho más costoso computacionalmente
#
# En datasets grandes o exploración inicial → TVS
# En selección final de modelo → Cross-Validation


# %%
# RETO 4: Comparar Estrategias

print("\n" + "="*70)
print("COMPARACIÓN DE ESTRATEGIAS DE BÚSQUEDA DE HIPERPARÁMETROS")
print("="*70)

# Grid Search + Cross-Validation
print("\nGrid Search + Cross-Validation:")
print(f"  - Tiempo de ejecución: {grid_time:.2f} segundos")
print(f"  - RMSE en Test:        ${rmse_grid:,.2f}")
print(f"  - regParam (λ):        {best_grid_model.getRegParam()}")
print(f"  - elasticNetParam (α): {best_grid_model.getElasticNetParam()}")
print(f"  - maxIter:             {best_grid_model.getMaxIter()}")

# Train-Validation Split
print("\nTrain-Validation Split:")
print(f"  - Tiempo de ejecución: {tvs_time:.2f} segundos")
print(f"  - RMSE en Test:        ${rmse_tvs:,.2f}")
print(f"  - regParam (λ):        {best_tvs_model.getRegParam()}")
print(f"  - elasticNetParam (α): {best_tvs_model.getElasticNetParam()}")
print(f"  - maxIter:             {best_tvs_model.getMaxIter()}")

print("\n" + "="*70)

# %%
# Análisis guiado (responder en markdown):
#
# - ¿Qué estrategia obtuvo menor RMSE en test?
# - ¿Cuál fue significativamente más rápida?
# - ¿Los hiperparámetros seleccionados coinciden?
# - ¿Usarías Train-Validation Split en exploración inicial?
# - ¿Reservarías Cross-Validation para el modelo final?
#
# Conclusión esperada:
# - TVS → rápido, útil para exploración
# - CV  → más robusto, ideal para decisión final


# %%
# RETO 5: Seleccionar y Guardar Modelo Final

import json
import os

print("\n" + "="*70)
print("SELECCIÓN Y GUARDADO DEL MODELO FINAL")
print("="*70)

# Seleccionar el mejor modelo entre ambas estrategias
if rmse_grid < rmse_tvs:
    mejor_modelo = best_grid_model
    mejor_rmse = rmse_grid
    estrategia = "Grid Search + Cross-Validation"
else:
    mejor_modelo = best_tvs_model
    mejor_rmse = rmse_tvs
    estrategia = "Train-Validation Split"

print(f"\nEstrategia seleccionada: {estrategia}")
print(f"RMSE Test: ${mejor_rmse:,.2f}")

# Ruta para guardar el modelo
model_path = "/opt/spark-data/processed/tuned_model"

# Eliminar si ya existe (Spark falla si la ruta existe)
if os.path.exists(model_path):
    import shutil
    shutil.rmtree(model_path)

# Guardar el modelo
mejor_modelo.save(model_path)
print(f"✓ Modelo guardado en: {model_path}")

# %%
# Guardar hiperparámetros óptimos
hiperparametros_optimos = {
    "estrategia": estrategia,
    "regParam": float(mejor_modelo.getRegParam()),
    "elasticNetParam": float(mejor_modelo.getElasticNetParam()),
    "maxIter": int(mejor_modelo.getMaxIter()),
    "rmse_test": float(mejor_rmse)
}

params_path = "/opt/spark-data/processed/hiperparametros_optimos.json"

with open(params_path, "w") as f:
    json.dump(hiperparametros_optimos, f, indent=2)

print(f"✓ Hiperparámetros óptimos guardados en: {params_path}")

print("\nResumen final:")
for k, v in hiperparametros_optimos.items():
    print(f"  - {k}: {v}")

print("="*70)


# %%
# RETO BONUS: Grid Más Fino

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import time

print("\n" + "="*70)
print("GRID FINO DE HIPERPARÁMETROS")
print("="*70)

# Extraer mejores hiperparámetros del modelo seleccionado
best_reg = float(mejor_modelo.getRegParam())
best_elastic = float(mejor_modelo.getElasticNetParam())
best_iter = int(mejor_modelo.getMaxIter())

print(f"Mejor regParam inicial: {best_reg}")
print(f"Mejor elasticNetParam inicial: {best_elastic}")
print(f"Mejor maxIter inicial: {best_iter}")

# Definir grid fino alrededor del mejor regParam
reg_fino = sorted(set([
    best_reg * 0.5,
    best_reg * 0.8,
    best_reg,
    best_reg * 1.2,
    best_reg * 1.5
]))

# Mantener elasticNetParam fijo o explorar levemente
elastic_fino = sorted(set([
    max(0.0, best_elastic - 0.1),
    best_elastic,
    min(1.0, best_elastic + 0.1)
]))

print("\nGrid fino propuesto:")
print(f"regParam: {reg_fino}")
print(f"elasticNetParam: {elastic_fino}")

# Construir grid fino
fine_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, reg_fino) \
    .addGrid(lr.elasticNetParam, elastic_fino) \
    .addGrid(lr.maxIter, [best_iter]) \
    .build()

print(f"\nCombinaciones totales en grid fino: {len(fine_grid)}")

# Configurar CrossValidator
cv_fino = CrossValidator(
    estimator=lr,
    estimatorParamMaps=fine_grid,
    evaluator=evaluator,
    numFolds=3,
    seed=42
)

# Ejecutar CV
print("\nEntrenando Grid Fino con Cross-Validation...")
start_time = time.time()
cv_fino_model = cv_fino.fit(train)
elapsed_time = time.time() - start_time

print(f"Grid fino completado en {elapsed_time:.2f} segundos")

# Evaluar mejor modelo del grid fino
best_fine_model = cv_fino_model.bestModel
preds_fino = best_fine_model.transform(test)
rmse_fino = evaluator.evaluate(preds_fino)

print("\n=== RESULTADO GRID FINO ===")
print(f"regParam óptimo: {best_fine_model.getRegParam()}")
print(f"elasticNetParam óptimo: {best_fine_model.getElasticNetParam()}")
print(f"maxIter: {best_fine_model.getMaxIter()}")
print(f"RMSE Test (Grid Fino): ${rmse_fino:,.2f}")

# Comparación final
print("\n=== COMPARACIÓN FINAL ===")
print(f"RMSE modelo anterior: ${mejor_rmse:,.2f}")
print(f"RMSE grid fino:       ${rmse_fino:,.2f}")
print(f"Mejora absoluta:      ${mejor_rmse - rmse_fino:,.2f}")

print("="*70)

# %%
# RESPUESTAS – PREGUNTAS DE REFLEXIÓN

# 1. ¿Cuándo usarías Grid Search vs Random Search?
#
# Usaría Grid Search cuando:
# - El número de hiperparámetros es pequeño
# - El espacio de búsqueda es acotado y bien conocido
# - Quiero comparar todas las combinaciones de forma exhaustiva
# - El costo computacional es manejable
#
# Usaría Random Search cuando:
# - El espacio de hiperparámetros es grande
# - Algunos hiperparámetros son mucho más importantes que otros
# - Tengo limitaciones de tiempo o recursos
# - Busco buenas soluciones rápidamente sin evaluar todas las combinaciones
#
# En la práctica, Random Search suele encontrar modelos competitivos
# con mucho menor costo computacional que Grid Search.


# 2. ¿Por qué Train-Validation Split es más rápido que Cross-Validation?
#
# Train-Validation Split es más rápido porque:
# - Cada combinación de hiperparámetros se entrena una sola vez
# - No repite el entrenamiento K veces como en Cross-Validation
#
# En contraste, Cross-Validation:
# - Entrena cada modelo K veces (una por cada fold)
# - Tiene mayor robustez estadística
# - Es significativamente más costoso en datasets grandes
#
# Por eso, Train-Validation Split es útil para exploración rápida,
# mientras que Cross-Validation es preferible para selección final.


# 3. ¿Qué pasa si el grid es demasiado grande?
#
# Si el grid es demasiado grande:
# - El tiempo de entrenamiento crece de forma exponencial
# - Se consume una gran cantidad de recursos computacionales
# - Puede volverse impráctico o imposible de ejecutar
#
# Además:
# - Muchas combinaciones no aportan mejoras reales
# - Se corre el riesgo de overfitting al proceso de validación
#
# Por ello, es clave diseñar grids inteligentes:
# - Usar escalas logarítmicas
# - Refinar el grid progresivamente
# - Combinar con Random Search o búsqueda jerárquica


# 4. ¿Cómo implementarías Random Search en Spark ML?
#
# Spark ML no tiene Random Search nativo, pero se puede implementar de forma manual:
#
# - Definir distribuciones o rangos para cada hiperparámetro
# - Muestrear aleatoriamente combinaciones (usando Python o NumPy)
# - Construir un ParamGridBuilder con esas combinaciones aleatorias
# - Ejecutar CrossValidator o TrainValidationSplit con ese grid
#
# Alternativamente:
# - Ejecutar bucles manuales entrenando modelos con hiperparámetros aleatorios
# - Registrar métricas y seleccionar el mejor modelo
#
# Este enfoque permite explorar grandes espacios de búsqueda
# con un costo computacional mucho menor que Grid Search exhaustivo.


# %%
print("\n" + "="*60)
print("RESUMEN OPTIMIZACIÓN DE HIPERPARÁMETROS")
print("="*60)
print("Verifica que hayas completado:")
print("  [ ] Diseñado grid de hiperparámetros")
print("  [ ] Ejecutado Grid Search + CV")
print("  [ ] Ejecutado Train-Validation Split")
print("  [ ] Comparado ambas estrategias")
print("  [ ] Guardado mejor modelo y hiperparámetros")
print("="*60)

# %%
spark.stop()
