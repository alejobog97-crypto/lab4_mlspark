# ============================================================
# NOTEBOOK 09: Optimización de Hiperparámetros
# ====================================================

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import when, col
from pyspark.sql.functions import abs as spark_abs, col
from pyspark.ml.feature import StandardScaler, PCA, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from delta import configure_spark_with_delta_pip
from pyspark.ml.feature import VectorAssembler
import numpy as np
import time
import json
import os

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
# Carga de datos
# ------------------------------------------------------------

df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")
df = df.withColumnRenamed("valor_del_contrato_num", "label") \
       .withColumnRenamed("features_pca", "features") \
       .filter(col("label").isNotNull())

train, test = df.randomSplit([0.8, 0.2], seed=42)

print(f"Train: {train.count():,}")
print(f"Test: {test.count():,}")

# ------------------------------------------------------------
# MODELO BASE Y EVALUADOR
# ------------------------------------------------------------

print("\n" + "-"*60)
print("CONFIGURACIÓN DEL MODELO BASE Y EVALUADOR")
print("-"*60)

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

print("✓ Modelo base LinearRegression configurado")
print(f"  • featuresCol: {lr.getFeaturesCol()}")
print(f"  • labelCol:    {lr.getLabelCol()}")
print(f"  • maxIter:     {lr.getMaxIter()}")

print("✓ Evaluador configurado")
print("  • Métrica: RMSE")

# ------------------------------------------------------------
# RETO 1: DISEÑAR EL GRID DE HIPERPARÁMETROS
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO 1: DISEÑO DEL GRID DE HIPERPARÁMETROS")
print("-"*60)

grid = (
    ParamGridBuilder()
    .addGrid(lr.regParam, [
        0.01,   # Regularización baja
        0.1,    # Regularización media
        1.0     # Regularización fuerte
    ])
    .addGrid(lr.elasticNetParam, [
        0.0,    # Ridge (L2)
        0.5,    # ElasticNet
        1.0     # Lasso (L1)
    ])
    .addGrid(lr.maxIter, [
        50,
        100,
        200
    ])
    .build()
)

print(f"Combinaciones totales en el grid: {len(grid)}")

print(
    "\nReflexión conceptual:\n"
    "- ¿Por qué usar una escala logarítmica para regParam?\n"
    "  • Porque el efecto de la regularización NO es lineal\n"
    "  • Cambios pequeños en valores bajos (0.01 → 0.1) tienen\n"
    "    un impacto mucho mayor que cambios equivalentes en valores altos\n\n"
    "- ¿Cuántas combinaciones hay en el grid?\n"
    "  • 3 valores de regParam\n"
    "  • 3 valores de elasticNetParam\n"
    "  • 3 valores de maxIter\n"
    "  → Total: 3 × 3 × 3 = 27 combinaciones\n\n"
    "- ¿Cuántos modelos se entrenan con K=3?\n"
    "  • 27 combinaciones × 3 folds = 81 modelos"
)

# ------------------------------------------------------------
# RETO 2: GRID SEARCH CON CROSS-VALIDATION
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO 2: GRID SEARCH + CROSS-VALIDATION (K=3)")
print("-"*60)

cv_grid = CrossValidator(
    estimator=lr,
    estimatorParamMaps=grid,
    evaluator=evaluator,
    numFolds=3,   # K=3 → balance entre robustez y tiempo
    seed=42
)

print("Entrenando Grid Search + Cross-Validation...")
start_time = time.time()

cv_grid_model = cv_grid.fit(train)

grid_time = time.time() - start_time
print(f"✓ Grid Search + CV completado en {grid_time:.2f} segundos")

# ------------------------------------------------------------
# MEJOR MODELO ENCONTRADO
# ------------------------------------------------------------

best_grid_model = cv_grid_model.bestModel

predictions_grid = best_grid_model.transform(test)
rmse_grid = evaluator.evaluate(predictions_grid)

print("\n" + "="*60)
print("MEJOR MODELO (GRID SEARCH + CV)")
print("="*60)
print(f"regParam (λ):         {best_grid_model.getRegParam()}")
print(f"elasticNetParam (α):  {best_grid_model.getElasticNetParam()}")
print(f"maxIter:              {best_grid_model.getMaxIter()}")
print(f"RMSE en Test:         ${rmse_grid:,.2f}")
print("="*60)

print(
    "\nJustificación de la elección de K=3:\n"
    "- El Grid Search ya es computacionalmente costoso\n"
    "- Modelos entrenados = combinaciones × K\n\n"
    "  • K=3  → 27 × 3  = 81 modelos\n"
    "  • K=5  → 27 × 5  = 135 modelos\n\n"
    "- K=3 ofrece un buen equilibrio entre:\n"
    "  ✔️ Estabilidad de la métrica\n"
    "  ✔️ Tiempo de ejecución razonable\n\n"
    "Especialmente recomendado para datasets medianos o grandes"
)

# ------------------------------------------------------------
# RETO 3: TRAIN–VALIDATION SPLIT
# ------------------------------------------------------------

from pyspark.ml.tuning import TrainValidationSplit
import time

print("\n" + "-"*60)
print("RETO 3: TRAIN–VALIDATION SPLIT")
print("-"*60)

print(
    "Objetivo:\n"
    "- Entrenar modelos usando un único split de validación\n"
    "- Comparar velocidad y desempeño frente a Cross-Validation\n"
    "- Usar esta técnica como alternativa más rápida\n"
)

# Configurar TrainValidationSplit
tvs = TrainValidationSplit(
    estimator=lr,
    estimatorParamMaps=grid,
    evaluator=evaluator,
    trainRatio=0.8,   # 80% train / 20% validation
    seed=42
)

print("Configuración:")
print("  • trainRatio: 80% entrenamiento / 20% validación")
print("  • Misma grilla de hiperparámetros que Grid Search")

# Ejecutar Train-Validation Split
print("\nEntrenando con Train-Validation Split...")
start_time = time.time()

tvs_model = tvs.fit(train)

tvs_time = time.time() - start_time
print(f"✓ Train-Validation Split completado en {tvs_time:.2f} segundos")

# ------------------------------------------------------------
# MEJOR MODELO (TRAIN–VALIDATION SPLIT)
# ------------------------------------------------------------

best_tvs_model = tvs_model.bestModel

predictions_tvs = best_tvs_model.transform(test)
rmse_tvs = evaluator.evaluate(predictions_tvs)

print("\n" + "="*60)
print("MEJOR MODELO (TRAIN–VALIDATION SPLIT)")
print("="*60)
print(f"regParam (λ):         {best_tvs_model.getRegParam()}")
print(f"elasticNetParam (α):  {best_tvs_model.getElasticNetParam()}")
print(f"maxIter:              {best_tvs_model.getMaxIter()}")
print(f"RMSE en Test:         ${rmse_tvs:,.2f}")
print("="*60)

print(
    "\nInterpretación:\n"
    "- Train-Validation Split entrena cada combinación UNA sola vez\n"
    "- La métrica depende de un único split\n"
    "- Es significativamente más rápido que Cross-Validation\n"
)

# ------------------------------------------------------------
# RETO 4: COMPARAR ESTRATEGIAS
# ------------------------------------------------------------

print("\n" + "="*70)
print("COMPARACIÓN DE ESTRATEGIAS DE BÚSQUEDA DE HIPERPARÁMETROS")
print("="*70)

print("\nGrid Search + Cross-Validation:")
print(f"  • Tiempo de ejecución: {grid_time:.2f} segundos")
print(f"  • RMSE en Test:        ${rmse_grid:,.2f}")
print(f"  • regParam (λ):        {best_grid_model.getRegParam()}")
print(f"  • elasticNetParam (α): {best_grid_model.getElasticNetParam()}")
print(f"  • maxIter:             {best_grid_model.getMaxIter()}")

print("\nTrain-Validation Split:")
print(f"  • Tiempo de ejecución: {tvs_time:.2f} segundos")
print(f"  • RMSE en Test:        ${rmse_tvs:,.2f}")
print(f"  • regParam (λ):        {best_tvs_model.getRegParam()}")
print(f"  • elasticNetParam (α): {best_tvs_model.getElasticNetParam()}")
print(f"  • maxIter:             {best_tvs_model.getMaxIter()}")

print("\n" + "="*70)

print(
    "\nAnálisis guiado:\n"
    "- ¿Qué estrategia obtuvo menor RMSE en test?\n"
    "- ¿Cuál fue significativamente más rápida?\n"
    "- ¿Coinciden los hiperparámetros seleccionados?\n"
    "- ¿Usarías Train-Validation Split para exploración inicial?\n"
    "- ¿Reservarías Cross-Validation para el modelo final?\n\n"
    "Conclusión esperada:\n"
    "✔ Train-Validation Split → rápido, útil para exploración\n"
    "✔ Cross-Validation       → más robusto, ideal para decisión final"
)


# ------------------------------------------------------------
# RETO 5: SELECCIÓN Y GUARDADO DEL MODELO FINAL
# ------------------------------------------------------------

print("\n" + "="*70)
print("SELECCIÓN Y GUARDADO DEL MODELO FINAL")
print("="*70)

print(
    "Objetivo:\n"
    "- Comparar los modelos obtenidos por ambas estrategias\n"
    "- Seleccionar el modelo con mejor desempeño en test\n"
    "- Guardar el modelo y sus hiperparámetros óptimos\n"
)

# Seleccionar el mejor modelo entre ambas estrategias
if rmse_grid < rmse_tvs:
    mejor_modelo = best_grid_model
    mejor_rmse = rmse_grid
    estrategia = "Grid Search + Cross-Validation"
else:
    mejor_modelo = best_tvs_model
    mejor_rmse = rmse_tvs
    estrategia = "Train-Validation Split"

print(f"Estrategia seleccionada: {estrategia}")
print(f"RMSE en Test: ${mejor_rmse:,.2f}")

# Ruta para guardar el modelo
model_path = "/opt/spark-data/processed/tuned_model"

# Eliminar si ya existe (Spark falla si la ruta existe)
if os.path.exists(model_path):
    import shutil
    shutil.rmtree(model_path)

# Guardar el modelo
mejor_modelo.save(model_path)
print(f"✓ Modelo guardado correctamente en: {model_path}")

# ------------------------------------------------------------
# GUARDAR HIPERPARÁMETROS ÓPTIMOS
# ------------------------------------------------------------

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

print("\nResumen del modelo final:")
for k, v in hiperparametros_optimos.items():
    print(f"  • {k}: {v}")

print("="*70)

# ------------------------------------------------------------
# RETO BONUS: GRID FINO DE HIPERPARÁMETROS
# ------------------------------------------------------------

print("\n" + "="*70)
print("RETO BONUS: GRID FINO DE HIPERPARÁMETROS")
print("="*70)

print(
    "Objetivo:\n"
    "- Refinar la búsqueda alrededor del mejor modelo encontrado\n"
    "- Explorar pequeñas variaciones de los hiperparámetros óptimos\n"
)

# Extraer mejores hiperparámetros
best_reg = float(mejor_modelo.getRegParam())
best_elastic = float(mejor_modelo.getElasticNetParam())
best_iter = int(mejor_modelo.getMaxIter())

print("Mejores hiperparámetros iniciales:")
print(f"  • regParam (λ): {best_reg}")
print(f"  • elasticNetParam (α): {best_elastic}")
print(f"  • maxIter: {best_iter}")

# Definir grid fino alrededor del mejor regParam
reg_fino = sorted(set([
    best_reg * 0.5,
    best_reg * 0.8,
    best_reg,
    best_reg * 1.2,
    best_reg * 1.5
]))

# Ajuste leve de elasticNetParam
elastic_fino = sorted(set([
    max(0.0, best_elastic - 0.1),
    best_elastic,
    min(1.0, best_elastic + 0.1)
]))

print("\nGrid fino propuesto:")
print(f"  • regParam: {reg_fino}")
print(f"  • elasticNetParam: {elastic_fino}")

# Construir grid fino
fine_grid = (
    ParamGridBuilder()
    .addGrid(lr.regParam, reg_fino)
    .addGrid(lr.elasticNetParam, elastic_fino)
    .addGrid(lr.maxIter, [best_iter])
    .build()
)

print(f"\nCombinaciones totales en el grid fino: {len(fine_grid)}")

# Configurar CrossValidator
cv_fino = CrossValidator(
    estimator=lr,
    estimatorParamMaps=fine_grid,
    evaluator=evaluator,
    numFolds=3,
    seed=42
)

# Ejecutar CV fino
print("\nEntrenando Grid Fino con Cross-Validation...")
start_time = time.time()
cv_fino_model = cv_fino.fit(train)
elapsed_time = time.time() - start_time

print(f"✓ Grid fino completado en {elapsed_time:.2f} segundos")

# Evaluar mejor modelo del grid fino
best_fine_model = cv_fino_model.bestModel
preds_fino = best_fine_model.transform(test)
rmse_fino = evaluator.evaluate(preds_fino)

print("\n=== RESULTADO GRID FINO ===")
print(f"regParam óptimo:        {best_fine_model.getRegParam()}")
print(f"elasticNetParam óptimo:{best_fine_model.getElasticNetParam()}")
print(f"maxIter:               {best_fine_model.getMaxIter()}")
print(f"RMSE Test (Grid Fino): ${rmse_fino:,.2f}")

print("\n=== COMPARACIÓN FINAL ===")
print(f"RMSE modelo anterior: ${mejor_rmse:,.2f}")
print(f"RMSE grid fino:       ${rmse_fino:,.2f}")
print(f"Mejora absoluta:      ${mejor_rmse - rmse_fino:,.2f}")

print("="*70)

# ------------------------------------------------------------
# RESPUESTAS A PREGUNTAS DE REFLEXIÓN
# ------------------------------------------------------------

print("\nREFLEXIONES CLAVE:")

print(
    "\n1) ¿Cuándo usar Grid Search vs Random Search?\n"
    "- Grid Search es adecuado cuando el espacio es pequeño y bien definido\n"
    "- Random Search es preferible en espacios grandes y con restricciones de tiempo\n"
    "- En la práctica, Random Search suele encontrar soluciones competitivas con menor costo"
)

print(
    "\n2) ¿Por qué Train-Validation Split es más rápido que Cross-Validation?\n"
    "- Porque cada combinación se entrena una sola vez\n"
    "- Cross-Validation repite el entrenamiento K veces\n"
    "- TVS es ideal para exploración, CV para decisión final"
)

print(
    "\n3) ¿Qué pasa si el grid es demasiado grande?\n"
    "- El costo computacional crece exponencialmente\n"
    "- Muchas combinaciones no aportan mejoras reales\n"
    "- Se recomienda usar grids jerárquicos y refinamiento progresivo"
)

print(
    "\n4) ¿Cómo implementar Random Search en Spark ML?\n"
    "- Generar combinaciones aleatorias manualmente\n"
    "- Construir ParamGridBuilder con esas combinaciones\n"
    "- Ejecutar CrossValidator o TrainValidationSplit\n"
    "- Alternativamente, usar bucles manuales y registrar métricas"
)

print("\n" + "="*60)
print("RESUMEN OPTIMIZACIÓN DE HIPERPARÁMETROS")
print("="*60)
print("Verifica que hayas completado:")
print("  [ ] Diseñado grid de hiperparámetros")
print("  [ ] Ejecutado Grid Search + Cross-Validation")
print("  [ ] Ejecutado Train-Validation Split")
print("  [ ] Comparado ambas estrategias")
print("  [ ] Guardado modelo final y parámetros")
print("="*60)

spark.stop()
