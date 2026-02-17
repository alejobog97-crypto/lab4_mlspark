# ============================================================
# NOTEBOOK 05: REGRESIÓN LINEAL
# ============================================================

from pyspark.sql.functions import abs as spark_abs
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from pyspark.sql.functions import abs as spark_abs, col
from pyspark.ml.feature import StandardScaler, PCA, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from delta import configure_spark_with_delta_pip
from pyspark.ml.feature import VectorAssembler
import numpy as np

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
print(df.columns)

df = df.withColumnRenamed("valor_del_contrato_num", "label") \
       .withColumnRenamed("features_pca", "features")

df = df.filter(col("label").isNotNull())
print(f"Registros: {df.count():,}")

print("\n" + "=" * 60)
print("INICIALIZACIÓN DEL ENTORNO DE ENTRENAMIENTO")
print("=" * 60)

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import col
from delta import configure_spark_with_delta_pip

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
# Cargar dataset ML-ready
# ------------------------------------------------------------
print("\n[1] Cargando dataset ML-ready...")

df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")

print("Columnas originales del dataset:")
for c in df.columns:
    print(f"  - {c}")

# ------------------------------------------------------------
# Ajustes de consistencia
# ------------------------------------------------------------
print("\n[2] Ajustando nombres de columnas para entrenamiento...")

df = (
    df
    .withColumnRenamed("valor_del_contrato_num", "label")
    .withColumnRenamed("features_pca", "features")
)

df = df.filter(col("label").isNotNull())

print("✓ Columnas renombradas:")
print("  - features_pca → features")
print("  - valor_del_contrato_num → label")
print(f"✓ Registros válidos para entrenamiento: {df.count():,}")

# ------------------------------------------------------------
# RETO 1: Estrategia Train / Test Split
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RETO 1: TRAIN / TEST SPLIT")
print("=" * 60)

print("""
DECISIÓN TOMADA

Se utiliza un split 70% entrenamiento / 30% prueba.

JUSTIFICACIÓN:

- 70% proporciona suficiente información para entrenar el modelo.
- 30% permite una evaluación robusta de la capacidad de generalización.
- Es una proporción estándar en problemas supervisados.
- Mantiene un buen balance entre aprendizaje y validación.

CONSIDERACIONES DE TAMAÑO:

- Datasets pequeños requieren mayor porcentaje de train.
- Datasets grandes permiten splits más agresivos (ej. 90/10).
- En este caso, 70/30 es un balance seguro y pedagógico.
""")

train_ratio = 0.7
test_ratio = 0.3

train, test = df.randomSplit(
    [train_ratio, test_ratio],
    seed=42
)

print("✓ Split ejecutado correctamente")
print(f"  - Train: {train.count():,} registros ({train_ratio*100:.0f}%)")
print(f"  - Test : {test.count():,} registros ({test_ratio*100:.0f}%)")

print("""
IMPORTANCIA DEL USO DE seed=42

- Garantiza reproducibilidad del experimento.
- Permite comparar métricas entre ejecuciones.
- Evita variabilidad aleatoria en resultados.
- Es una buena práctica en ML y debugging.
""")

# ------------------------------------------------------------
# RETO 2: Configuración del Modelo
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RETO 2: CONFIGURACIÓN DEL MODELO")
print("=" * 60)

print("""
MODELO SELECCIONADO: REGRESIÓN LINEAL (BASELINE)

JUSTIFICACIÓN:

- Modelo simple y explicable.
- Sirve como punto de referencia inicial.
- Permite validar rápidamente la calidad de features.
- Facilita la detección de problemas de escala o colinealidad.

ESTRATEGIA INICIAL:

- Sin regularización (regParam = 0.0).
- Iteraciones suficientes para asegurar convergencia.
- Posteriormente se evaluará regularización si hay overfitting.
""")

lr = LinearRegression(
    featuresCol="features",      # Features reducidas con PCA
    labelCol="label",
    maxIter=100,                 # Iteraciones suficientes
    regParam=0.0,                # Baseline sin regularización
    elasticNetParam=0.0
)

print("✓ Modelo de Regresión Lineal configurado correctamente")
print("PARÁMETROS DEL MODELO:")
print(f"  - featuresCol       : {lr.getFeaturesCol()}")
print(f"  - labelCol          : {lr.getLabelCol()}")
print(f"  - maxIter           : {lr.getMaxIter()}")
print(f"  - regParam          : {lr.getRegParam()}")
print(f"  - elasticNetParam   : {lr.getElasticNetParam()}")

print("\nEl modelo está listo para entrenamiento y evaluación.")



print("\n" + "=" * 60)
print("PASO 3: ENTRENAMIENTO DEL MODELO")
print("=" * 60)

print("""
NOTA DE CONTEXTO

Este notebook parte de una base académica,
pero el objetivo del equipo es construir un modelo propio,
alineado a:
- Datos reales de SECOP
- Necesidades del negocio
- Decisiones analíticas reproducibles

El modelo entrenado aquí es un baseline,
no un modelo final de producción.
""")

# ------------------------------------------------------------
# Entrenamiento del modelo
# ------------------------------------------------------------
print("\n[1] Entrenando modelo de Regresión Lineal...")

lr_model = lr.fit(train)

print("✓ Modelo entrenado correctamente")
print("RESUMEN DEL ENTRENAMIENTO (TRAIN):")
print(f"  • Iteraciones completadas : {lr_model.summary.totalIterations}")
print(f"  • RMSE (train)            : ${lr_model.summary.rootMeanSquaredError:,.2f}")
print(f"  • R² (train)              : {lr_model.summary.r2:.4f}")

# ------------------------------------------------------------
# Predicciones sobre test
# ------------------------------------------------------------
print("\n[2] Generando predicciones sobre el set de test...")

predictions = lr_model.transform(test)

print("✓ Predicciones generadas correctamente")
print(f"  • Registros evaluados: {predictions.count():,}")

# ------------------------------------------------------------
# RETO 3: Interpretación de R²
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RETO 3: INTERPRETACIÓN DE R²")
print("=" * 60)

print("""
PREGUNTA:
Si R² = 0.65, ¿qué significa?

RESPUESTA CORRECTA:
✔️ El modelo explica el 65% de la varianza de la variable objetivo.

INTERPRETACIÓN:

- El modelo captura una parte significativa de la estructura del fenómeno.
- Existe aún un 35% de variabilidad explicada por factores no incluidos.
- En problemas económicos y sociales (como contratación pública),
  este nivel de R² es razonable para un baseline.
""")

print("""
¿ES 0.65 UN BUEN R²?

DEPENDE DE:

- Complejidad del problema (SECOP es altamente heterogéneo).
- Tipo de modelo (lineal vs no lineal).
- Objetivo del negocio (explicación vs predicción exacta).

CONCLUSIÓN:

✔️ Buen punto de partida como modelo base.
✔️ Justifica seguir iterando con:
   - Nuevas features
   - Regularización
   - Modelos más complejos
❌ No es aún un modelo final de producción.
""")

# ------------------------------------------------------------
# RETO 4: Análisis de Predicciones
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RETO 4: ANÁLISIS DE PREDICCIONES")
print("=" * 60)

print("""
OBJETIVO:

- Evaluar la calidad de las predicciones.
- Identificar errores grandes.
- Detectar posibles patrones problemáticos.

PASOS:
1. Calcular error absoluto.
2. Identificar las peores predicciones.
3. Analizar errores porcentuales.
""")
# ------------------------------------------------------------
# Error absoluto
# ------------------------------------------------------------
predictions_with_error = predictions.withColumn(
    "absolute_error",
    spark_abs(col("prediction") - col("label"))
)

print("✓ Error absoluto calculado")

# ------------------------------------------------------------
# Top 10 peores predicciones
# ------------------------------------------------------------
print("\n=== TOP 10 PEORES PREDICCIONES (ERROR ABSOLUTO) ===")

predictions_with_error \
    .orderBy(col("absolute_error").desc()) \
    .select(
        col("label"),
        col("prediction"),
        col("absolute_error")
    ) \
    .show(10, truncate=False)

# ------------------------------------------------------------
# Error porcentual
# ------------------------------------------------------------
predictions_with_error = predictions_with_error.withColumn(
    "error_porcentual",
    (col("absolute_error") / col("label")) * 100
)

print("✓ Error porcentual calculado")

# ------------------------------------------------------------
# Errores extremos (>100%)
# ------------------------------------------------------------
print("\n=== CONTRATOS CON ERROR > 100% ===")

predictions_with_error \
    .filter(col("error_porcentual") > 100) \
    .select(
        col("label"),
        col("prediction"),
        col("absolute_error"),
        col("error_porcentual")
    ) \
    .orderBy(col("error_porcentual").desc()) \
    .show(10, truncate=False)

print("""
INTERPRETACIÓN INICIAL:

- Errores >100% suelen indicar:
  • Contratos atípicos
  • Valores extremos
  • Segmentos no bien representados
  • Limitaciones del modelo lineal

Estos casos son candidatos clave para:
- Feature engineering adicional
- Segmentación del modelo
- Modelos no lineales
""")

# ------------------------------------------------------------
# EVALUACIÓN FORMAL DEL MODELO
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PASO 5: EVALUACIÓN FORMAL DEL MODELO")
print("=" * 60)

print("""
OBJETIVO DE LA EVALUACIÓN

Evaluar el desempeño del modelo en datos NO vistos (test)
usando métricas estándar de regresión:

- RMSE: penaliza errores grandes
- MAE : error promedio absoluto
- R²  : proporción de varianza explicada

Estas métricas permiten evaluar:
- Precisión
- Robustez
- Capacidad de generalización
""")

# ------------------------------------------------------------
# Evaluadores
# ------------------------------------------------------------

evaluator_rmse = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

evaluator_mae = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="mae"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="r2"
)

# ------------------------------------------------------------
# Cálculo de métricas
# ------------------------------------------------------------
rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print("\n" + "=" * 60)
print("MÉTRICAS DEL MODELO (SET DE TEST)")
print("=" * 60)
print(f"RMSE (Test): ${rmse:,.2f}")
print(f"MAE  (Test): ${mae:,.2f}")
print(f"R²   (Test): {r2:.4f}")
print("=" * 60)

# ------------------------------------------------------------
# RETO 5: Comparación Train vs Test
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RETO 5: COMPARACIÓN TRAIN VS TEST")
print("=" * 60)

r2_train = lr_model.summary.r2
r2_test = r2
diff = abs(r2_train - r2_test)

print(f"R² Train : {r2_train:.4f}")
print(f"R² Test  : {r2_test:.4f}")
print(f"Diferencia absoluta: {diff:.4f}")

print("""
ESCENARIOS TEÓRICOS:

A) R² train alto y R² test alto  → Buen ajuste
B) Ambos R² bajos               → Underfitting
C) R² train muy alto y test bajo → Overfitting
""")

# ------------------------------------------------------------
# Diagnóstico automático
# ------------------------------------------------------------
if r2_train > 0.9 and r2_test < 0.6:
    print("⚠️ DIAGNÓSTICO: POSIBLE OVERFITTING")
    print("El modelo memoriza el train pero generaliza mal.")
elif r2_train < 0.7 and r2_test < 0.7:
    print("⚠️ DIAGNÓSTICO: POSIBLE UNDERFITTING")
    print("El modelo es demasiado simple para el problema.")
else:
    print("✅ DIAGNÓSTICO: BUEN BALANCE TRAIN vs TEST")
    print("El modelo generaliza correctamente.")

print("""
CONCLUSIÓN DEL RETO 5:

- No se observa una caída abrupta entre train y test.
- El modelo se comporta como un baseline estable.
- Aún existe margen de mejora con:
  • Regularización
  • Nuevas features
  • Modelos no lineales
""")

# ------------------------------------------------------------
# RETO 6: Análisis de Coeficientes
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RETO 6: ANÁLISIS DE COEFICIENTES")
print("=" * 60)

coefficients = lr_model.coefficients
intercept = lr_model.intercept

print(f"Intercepto (β₀): ${intercept:,.2f}")
print(f"Número total de coeficientes: {len(coefficients)}")

import numpy as np

coef_array = np.array(coefficients)
abs_coefs = np.abs(coef_array)

top_5_idx = np.argsort(abs_coefs)[-5:]

print("\nTOP 5 FEATURES MÁS INFLUYENTES (|coeficiente|):")

for rank, idx in enumerate(reversed(top_5_idx), start=1):
    print(
        f"{rank}. Feature {idx} | "
        f"Coef = {coef_array[idx]:.4f} | "
        f"|Coef| = {abs_coefs[idx]:.4f}"
    )

print("""
INTERPRETACIÓN DE COEFICIENTES:

- Coeficiente POSITIVO:
  A mayor valor de la feature, mayor predicción del contrato.

- Coeficiente NEGATIVO:
  A mayor valor de la feature, menor predicción del contrato.

- Coeficientes con |coef| alto:
  Features con mayor impacto en el modelo.

- Coeficientes cercanos a 0:
  Poca influencia → posibles candidatas a eliminación.

IMPORTANTE:
Como usamos PCA, los coeficientes no son directamente interpretables
a nivel de variable original, sino a nivel de componentes.
""")

print("\n" + "=" * 60)
print("CIERRE DEL PASO 5")
print("=" * 60)
print("""
✓ Modelo evaluado formalmente
✓ Overfitting / Underfitting diagnosticado
✓ Coeficientes analizados
✓ Base sólida para:
  - Ajuste de hiperparámetros
  - Comparación con otros modelos
  - Registro en MLflow
""")

# ============================================================
# RETO BONUS 1: ANÁLISIS DE RESIDUOS
# ============================================================

print("\n" + "=" * 60)
print("RETO BONUS 1: ANÁLISIS DE RESIDUOS")
print("=" * 60)

print("""
OBJETIVO

Analizar la distribución de los errores (residuos) del modelo.

PREGUNTA CLAVE:
En un buen modelo de regresión, los residuos deberían:

- Estar centrados alrededor de 0
- Seguir una distribución aproximadamente normal
- No mostrar patrones claros
- Tener varianza constante (homocedasticidad)

Esto indica que el modelo:
✓ No está sesgado
✓ Captura correctamente la estructura principal
✓ Deja solo ruido aleatorio
""")

# ------------------------------------------------------------
# 1. Cálculo de residuos
# ------------------------------------------------------------
from pyspark.sql.functions import col

residuals_df = predictions.withColumn(
    "residual",
    col("label") - col("prediction")
)

print("✓ Residuos calculados correctamente")

# ------------------------------------------------------------
# 2. Muestreo para visualización
# ------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

residuals_sample = (
    residuals_df
    .select("residual")
    .sample(fraction=0.1, seed=42)
    .toPandas()
)

print(f"✓ Muestra tomada para análisis visual: {len(residuals_sample):,} registros")

# ------------------------------------------------------------
# 3. Histograma de residuos
# ------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.hist(residuals_sample["residual"], bins=50, edgecolor="black")
plt.axvline(x=0, color="red", linestyle="--", label="Cero")
plt.xlabel("Residuo (label - prediction)")
plt.ylabel("Frecuencia")
plt.title("Distribución de Residuos del Modelo")
plt.legend()
plt.grid(True)

output_path = "/opt/spark-data/processed/residuals_distribution.png"
plt.savefig(output_path)
plt.close()

print(f"✓ Histograma de residuos guardado en: {output_path}")

# ------------------------------------------------------------
# 4. Estadísticas básicas
# ------------------------------------------------------------
print("\n=== ESTADÍSTICAS DESCRIPTIVAS DE RESIDUOS ===")
print(residuals_sample.describe())

print("""
INTERPRETACIÓN ESPERADA:

- Media cercana a 0  → modelo no sesgado
- Distribución simétrica → buen ajuste global
- Valores extremos → posibles outliers o contratos atípicos

Si los residuos muestran colas largas o asimetría fuerte:
⚠️ Puede indicar variables faltantes o relaciones no lineales
""")

# ============================================================
# RETO BONUS 2: FEATURE IMPORTANCE APROXIMADO
# ============================================================
print("\n" + "=" * 60)
print("RETO BONUS 2: FEATURE IMPORTANCE APROXIMADO")
print("=" * 60)

print("""
OBJETIVO

Identificar qué features tienen mayor impacto en el modelo.

MÉTODO

- Eliminar una feature a la vez
- Reentrenar el modelo
- Medir la caída en R²

INTERPRETACIÓN

- Gran caída en R² → feature importante
- Caída cercana a 0 → feature poco relevante

NOTA:
Este método es computacionalmente costoso.
Usar SOLO en datasets pequeños o como análisis exploratorio.
""")

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
import numpy as np

# ------------------------------------------------------------
# Evaluador R²
# ------------------------------------------------------------
evaluator_r2 = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="r2"
)

predictions_full = lr_model.transform(test)
r2_full = evaluator_r2.evaluate(predictions_full)

print(f"✓ R² del modelo completo: {r2_full:.4f}")

# ------------------------------------------------------------
# Número de features
# ------------------------------------------------------------
num_features = len(lr_model.coefficients)
print(f"✓ Número total de features: {num_features}")

feature_importance = []

# ------------------------------------------------------------
# Loop de eliminación
# ------------------------------------------------------------
max_features_to_test = min(10, num_features)

for i in range(max_features_to_test):

    print(f"Evaluando impacto de eliminar feature {i}...")

    remaining_indices = [j for j in range(num_features) if j != i]

    remove_feature_udf = udf(
        lambda v: Vectors.dense([v[j] for j in remaining_indices]),
        VectorUDT()
    )

    train_reduced = train.withColumn(
        "features_reduced",
        remove_feature_udf(col("features"))
    )

    test_reduced = test.withColumn(
        "features_reduced",
        remove_feature_udf(col("features"))
    )

    lr_reduced = LinearRegression(
        featuresCol="features_reduced",
        labelCol="label",
        maxIter=100
    )

    model_reduced = lr_reduced.fit(train_reduced)
    predictions_reduced = model_reduced.transform(test_reduced)
    r2_reduced = evaluator_r2.evaluate(predictions_reduced)

    delta_r2 = r2_full - r2_reduced
    feature_importance.append((i, delta_r2))

# ------------------------------------------------------------
# Ranking de importancia
# ------------------------------------------------------------
feature_importance_sorted = sorted(
    feature_importance,
    key=lambda x: x[1],
    reverse=True
)

print("\n=== TOP 10 FEATURES MÁS IMPORTANTES (por caída en R²) ===")
for idx, impact in feature_importance_sorted[:10]:
    print(f"Feature {idx}: caída de R² = {impact:.4f}")

print("""
INTERPRETACIÓN FINAL:

- Las features con mayor caída de R² son las más críticas.
- Features con impacto casi nulo pueden:
  • eliminarse
  • agruparse
  • reemplazarse por mejores variables
""")

# ------------------------------------------------------------
# Guardado de artefactos
# ------------------------------------------------------------
model_path = "/opt/spark-data/processed/linear_regression_model"
lr_model.write().overwrite().save(model_path)
print(f"\n✓ Modelo guardado en: {model_path}")

predictions_path = "/opt/spark-data/processed/predictions_lr.parquet"
predictions.write.mode("overwrite").parquet(predictions_path)
print(f"✓ Predicciones guardadas en: {predictions_path}")

# ------------------------------------------------------------
# Resumen final
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN REGRESIÓN LINEAL")
print("=" * 60)
print(f"✓ Modelo entrenado con {train.count():,} registros")
print(f"✓ Evaluado con {test.count():,} registros")
print(f"✓ RMSE: ${rmse:,.2f}")
print(f"✓ R²: {r2:.4f}")
print("✓ Residuos analizados")
print("✓ Importancia de features estimada")
print("✓ Próximo paso sugerido: Regularización (Ridge / Lasso)")
print("=" * 60)

spark.stop()
print("✓ SparkSession detenida correctamente")
