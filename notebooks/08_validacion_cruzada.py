# ============================================================
# NOTEBOOK 08: Validación Cruzada (K-Fold)
# ============================================================

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import when, col
from pyspark.sql.functions import abs as spark_abs, col
from pyspark.ml.feature import StandardScaler, PCA, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from delta import configure_spark_with_delta_pip
from pyspark.ml.feature import VectorAssembler
import numpy as np
import time

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
# RETO 1: ENTENDER K-FOLD CROSS-VALIDATION
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO 1: ENTENDER K-FOLD CROSS-VALIDATION")
print("-"*60)

K = 5

print(f"\nSuposición del ejercicio: K = {K}")

print(
    "\nPregunta 1:\n"
    "¿En cuántos subconjuntos se dividen los datos de entrenamiento?"
)
print(
    f"Respuesta:\n"
    f"→ En {K} subconjuntos (folds) de tamaño aproximadamente igual."
)

print(
    "\nPregunta 2:\n"
    "¿Cuántos modelos se entrenan en total?"
)
print(
    "Respuesta:\n"
    f"→ Se entrenan {K} modelos.\n"
    "  En cada iteración:\n"
    "  • 1 fold se usa como validación\n"
    f"  • {K-1} folds se usan como entrenamiento"
)

print(
    "\nPregunta 3:\n"
    "¿Qué porcentaje de datos se usa para validación en cada iteración?"
)
print(
    "Respuesta:\n"
    f"→ 1/K del total de los datos de entrenamiento.\n"
    f"→ Para K = {K}: 20% validación y 80% entrenamiento en cada iteración."
)

print(
    "\nPregunta 4:\n"
    "¿Qué métrica se reporta al final del proceso?"
)
print(
    "Respuesta:\n"
    "→ El promedio de la métrica evaluada en cada fold\n"
    "  (y opcionalmente su desviación estándar).\n"
    "→ Ejemplo: RMSE promedio de los 5 folds."
)

print(
    "\nPregunta clave:\n"
    "¿Por qué K-Fold Cross-Validation es mejor que un simple train/test split?"
)

print(
    "\nRespuesta:\n"
    "- Reduce la dependencia de una sola partición aleatoria.\n"
    "- Usa todos los datos tanto para entrenamiento como para validación.\n"
    "- Produce métricas más estables y confiables.\n"
    "- Detecta mejor overfitting y underfitting.\n"
    "- Es especialmente útil cuando el dataset no es muy grande."
)

# ------------------------------------------------------------
# RETO 2: CREAR EL MODELO BASE Y EL EVALUADOR
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO 2: CREAR EL MODELO BASE Y EL EVALUADOR")
print("-"*60)

print(
    "\nObjetivo del reto:\n"
    "- Definir un modelo base de regresión lineal\n"
    "- Definir un evaluador consistente para comparar modelos"
)

# Modelo base de regresión lineal (baseline)
lr = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100
)

print("\n✓ Modelo base LinearRegression creado")
print(f"  • featuresCol: {lr.getFeaturesCol()}")
print(f"  • labelCol:    {lr.getLabelCol()}")
print(f"  • maxIter:     {lr.getMaxIter()}")

# Evaluador
evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

print("\n✓ Evaluador configurado correctamente")
print("  • Métrica seleccionada: RMSE")

print(
    "\nJustificación de la métrica RMSE:\n"
    "- Penaliza más los errores grandes.\n"
    "- Es especialmente relevante cuando errores grandes\n"
    "  implican alto impacto financiero (ej. contratos grandes).\n"
    "- Mantiene las mismas unidades de la variable objetivo,\n"
    "  lo que facilita interpretación para negocio."
)

print(
    "\nReflexión adicional sobre métricas:\n"
    "- Usaría MAE si:\n"
    "  • Todos los errores tienen el mismo impacto.\n"
    "  • Quiero robustez frente a outliers.\n\n"
    "- Usaría R² si:\n"
    "  • Quiero comparar capacidad explicativa entre modelos.\n"
    "  • El objetivo es más analítico que predictivo.\n\n"
    "Conclusión:\n"
    "✔️ RMSE es una excelente métrica principal.\n"
    "✔️ MAE y R² son métricas complementarias útiles."
)

# ------------------------------------------------------------
# RETO 3: CONSTRUIR EL PARAMGRID
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO 3: CONSTRUIR EL PARAMGRID")
print("-"*60)

print(
    "\nObjetivo del reto:\n"
    "- Definir combinaciones de hiperparámetros para Cross-Validation\n"
    "- Explorar distintos niveles y tipos de regularización"
)

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
print(f"\n✓ Combinaciones de hiperparámetros en el grid: {num_combinations}")

# Cross-Validation con K-Fold
K = 5
total_models = num_combinations * K

print(f"✓ Número de folds (K): {K}")
print(f"✓ Total de modelos a entrenar: {total_models}")

print(
    "\nExplicación del costo computacional:\n"
    "- 3 valores de regParam × 3 valores de elasticNetParam = 9 combinaciones\n"
    "- Con K = 5 folds:\n"
    "  → 9 × 5 = 45 modelos entrenados\n\n"
    "Conclusión:\n"
    "Cross-Validation produce modelos más robustos,\n"
    "pero incrementa significativamente el costo computacional."
)

# ------------------------------------------------------------
# RETO 4: CONFIGURAR CROSSVALIDATOR
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO 4: CONFIGURAR CROSSVALIDATOR")
print("-"*60)

print(
    "\nObjetivo del reto:\n"
    "- Ensamblar el proceso completo de Cross-Validation\n"
    "- Entrenar múltiples modelos automáticamente\n"
    "- Seleccionar el mejor modelo según la métrica RMSE"
)

# Elección de K
print(f"\nElección de K-Fold Cross-Validation:")
print(f"→ K = {K}")

print(
    "\nJustificación de K = 5:\n"
    "- Reduce la varianza del error de evaluación\n"
    "- Es menos costoso que K = 10\n"
    "- Es un estándar ampliamente usado en problemas reales"
)

# Configuración del CrossValidator
crossval = CrossValidator(
    estimator=lr,                     # Modelo base
    estimatorParamMaps=param_grid,    # Grid de hiperparámetros
    evaluator=evaluator,              # Métrica (RMSE)
    numFolds=K,                       # K-Fold Cross-Validation
    seed=42                           # Reproducibilidad
)

print("\n✓ CrossValidator configurado correctamente")
print(f"  • Número de folds (K): {K}")
print(f"  • Combinaciones de hiperparámetros: {len(param_grid)}")
print(f"  • Total de modelos a entrenar: {len(param_grid) * K}")

print(
    "\nAdvertencia práctica:\n"
    "⚠️ El número de modelos crece rápidamente con:\n"
    "- Más hiperparámetros\n"
    "- Más valores por hiperparámetro\n"
    "- Valores altos de K\n\n"
    "En datasets grandes, es común usar:\n"
    "- TrainValidationSplit\n"
    "- Menos combinaciones\n"
    "- Búsqueda guiada (no grid exhaustivo)"
)

# ------------------------------------------------------------
# RETO 5: EJECUTAR CROSS-VALIDATION Y ANALIZAR RESULTADOS
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO 5: EJECUTAR CROSS-VALIDATION Y ANALIZAR RESULTADOS")
print("-"*60)

print(
    "\nObjetivo del reto:\n"
    "- Ejecutar Cross-Validation\n"
    "- Analizar métricas promedio\n"
    "- Identificar el mejor modelo\n"
    "- Evaluarlo en el set de test"
)

print("\nEntrenando modelos con Cross-Validation...")
cv_model = crossval.fit(train)
print("✓ Cross-Validation completada correctamente")

# ------------------------------------------------------------
# Análisis de métricas promedio (RMSE)
# ------------------------------------------------------------

avg_metrics = cv_model.avgMetrics

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
        f"RMSE CV=${metric:,.2f}"
        f"{marker}"
    )

print(
    "\nInterpretación:\n"
    "- avgMetrics contiene el RMSE promedio de cada combinación\n"
    "- El mejor modelo es el que minimiza el RMSE promedio\n"
    "- La selección NO se hace con datos de train ni test\n"
    "- Esto reduce overfitting y mejora la generalización"
)

# ------------------------------------------------------------
# Mejor modelo encontrado
# ------------------------------------------------------------

best_model = cv_model.bestModel

print("\n=== MEJOR MODELO SELECCIONADO POR CROSS-VALIDATION ===")
print(f"regParam (λ):        {best_model.getRegParam()}")
print(f"elasticNetParam (α): {best_model.getElasticNetParam()}")

# ------------------------------------------------------------
# Evaluación final en test
# ------------------------------------------------------------

predictions = best_model.transform(test)
rmse_test = evaluator.evaluate(predictions)

print("\n=== EVALUACIÓN FINAL EN TEST ===")
print(f"RMSE Test (modelo CV): ${rmse_test:,.2f}")

print(
    "\nConclusión del RETO 5:\n"
    "✔️ El modelo seleccionado es el que minimiza el RMSE promedio en Cross-Validation\n"
    "✔️ Es más robusto que seleccionar un modelo con un solo split\n"
    "✔️ Representa mejor el desempeño esperado en datos no vistos"
)

# ------------------------------------------------------------
# RETO 6: COMPARAR CROSS-VALIDATION VS SIMPLE SPLIT
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO 6: COMPARAR CROSS-VALIDATION VS SIMPLE SPLIT")
print("-"*60)

print(
    "\nObjetivo del reto:\n"
    "- Entrenar un modelo SIN Cross-Validation\n"
    "- Comparar su desempeño contra el modelo con CV\n"
    "- Evaluar cuál enfoque es más confiable"
)

print("\nEntrenando modelo SIMPLE (sin Cross-Validation)...")

lr_simple = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=best_model.getRegParam(),
    elasticNetParam=best_model.getElasticNetParam()
)

model_simple = lr_simple.fit(train)

rmse_simple = evaluator.evaluate(model_simple.transform(test))

print("\n=== COMPARACIÓN DE DESEMPEÑO ===")
print(f"RMSE con Cross-Validation: ${rmse_test:,.2f}")
print(f"RMSE sin Cross-Validation: ${rmse_simple:,.2f}")
print(f"Diferencia absoluta:       ${abs(rmse_test - rmse_simple):,.2f}")

print(
    "\nInterpretación final:\n"
    "- El modelo con Cross-Validation es más confiable porque:\n"
    "  • Evalúa múltiples particiones del train\n"
    "  • Reduce la dependencia de un solo split aleatorio\n"
    "  • Produce métricas más estables\n\n"
    "- El modelo sin CV puede:\n"
    "  • Sobreestimar o subestimar el desempeño real\n"
    "  • Depender fuertemente de la casualidad del split\n\n"
    "Conclusión:\n"
    "✔️ Cross-Validation ofrece una mejor estimación del desempeño real del modelo\n"
    "✔️ Es preferible cuando el costo computacional lo permite"
)

# ------------------------------------------------------------
# RETO BONUS: EXPERIMENTAR CON DIFERENTES VALORES DE K
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO BONUS: EXPERIMENTAR CON DIFERENTES VALORES DE K (CROSS-VALIDATION)")
print("-"*60)

print(
    "\nObjetivo del experimento:\n"
    "- Comparar K = 3, K = 5 y K = 10\n"
    "- Analizar el impacto en RMSE\n"
    "- Medir el tiempo de ejecución\n"
    "- Entender el trade-off entre robustez y costo computacional"
)

print("\n=== EJECUCIÓN DE EXPERIMENTOS ===")

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
        f"Tiempo de ejecución: {elapsed_time:.1f} segundos"
    )

# ------------------------------------------------------------
# Resumen comparativo
# ------------------------------------------------------------

print("\n" + "="*60)
print("RESUMEN COMPARATIVO POR VALOR DE K")
print("="*60)

for r in resultados_k:
    print(
        f"K={r['K']:2d} | "
        f"RMSE: ${r['best_rmse']:,.2f} | "
        f"Tiempo: {r['time_seconds']:.1f}s"
    )

print(
    "\nInterpretación del experimento:\n"
    "- K pequeño (ej. K=3):\n"
    "  • Menor tiempo de ejecución\n"
    "  • Métrica menos estable\n\n"
    "- K intermedio (K=5):\n"
    "  • Buen balance entre costo y robustez\n"
    "  • Es el valor más usado en práctica\n\n"
    "- K grande (K=10):\n"
    "  • Métrica más robusta\n"
    "  • Costo computacional significativamente mayor\n\n"
    "Conclusión:\n"
    "❌ Más folds NO siempre es mejor\n"
    "✔️ El valor óptimo de K depende del tamaño del dataset\n"
    "✔️ También depende del tiempo y recursos disponibles"
)

# ------------------------------------------------------------
# Guardar el mejor modelo entrenado con Cross-Validation
# ------------------------------------------------------------

model_path = "/opt/spark-data/processed/cv_best_model"
best_model.write().overwrite().save(model_path)

print(f"\n✓ Modelo guardado correctamente en: {model_path}")

# ------------------------------------------------------------
# Cierre del módulo
# ------------------------------------------------------------

print("\n" + "="*60)
print("RESUMEN VALIDACIÓN CRUZADA")
print("="*60)
print("Verifica que hayas completado:")
print("  [✓] Entendido el concepto de K-Fold")
print("  [✓] Configurado ParamGrid con hiperparámetros")
print("  [✓] Ejecutado CrossValidator")
print("  [✓] Identificado el mejor modelo")
print("  [✓] Comparado con entrenamiento simple")
print("  [✓] Analizado el impacto del valor de K")
print("="*60)

spark.stop()

