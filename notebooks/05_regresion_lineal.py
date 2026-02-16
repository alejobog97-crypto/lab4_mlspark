# %% [markdown]
# # Notebook 05: Regresión Lineal
#
# **Sección 14 - Regresión**: Predicción del valor de contratos
#
# **Objetivo**: Entrenar un modelo de regresión lineal para predecir el precio base.
#
# ## Actividades:
# 1. Dividir datos en train/test
# 2. Entrenar LinearRegression
# 3. Evaluar con RMSE, MAE, R²
# 4. Analizar coeficientes

# %%
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
print(df.columns)

# Renombrar columnas para consistencia
df = df.withColumnRenamed("valor_del_contrato_num", "label") \
       .withColumnRenamed("features_pca", "features")

# Filtrar valores nulos
df = df.filter(col("label").isNotNull())
print(f"Registros: {df.count():,}")

# %% [markdown]
# ## RETO 1: Train/Test Split Strategy
#
# **Decisión tomada**: Opción B → 70/30
#
# **Justificación**:
# - 70% permite entrenar el modelo con suficiente información
# - 30% deja un conjunto de test robusto para validar generalización
# - Es un estándar ampliamente usado en problemas supervisados
#
# **Consideración de tamaño del dataset**:
# - Con pocos datos (ej. 1.000 registros) es crítico no reducir demasiado el train
# - Con muchos datos (ej. 1.000.000 registros), incluso 90/10 puede ser válido
# - En este caso, 70/30 es un balance seguro y pedagógico

# %%
# Estrategia Train/Test Split
# Usamos randomSplit con seed para reproducibilidad

train_ratio = 0.7
test_ratio = 0.3

train, test = df.randomSplit(
    [train_ratio, test_ratio],
    seed=42
)

print(f"Train: {train.count():,} registros ({train_ratio*100:.0f}%)")
print(f"Test: {test.count():,} registros ({test_ratio*100:.0f}%)")

# %%
# ¿Por qué es importante usar seed=42?
#
# - Garantiza que el split sea reproducible
# - Permite comparar resultados entre ejecuciones
# - Evita variaciones aleatorias en métricas de evaluación
# - Es fundamental en experimentos de ML y debugging


# %% [markdown]
# ## RETO 2: Configurar el Modelo
#
# **Modelo elegido**: Linear Regression (baseline)
#
# **Justificación**:
# - Es un modelo simple y explicable
# - Sirve como punto de referencia (baseline)
# - Permite detectar rápidamente problemas de features o escalas
#
# **Estrategia inicial**:
# - Sin regularización (regParam = 0.0)
# - Iteraciones suficientes para converger
# - Luego se ajustará con regularización si hay overfitting


# Configuración del modelo de regresión lineal
lr = LinearRegression(
    featuresCol="features",   # Usamos features reducidas con PCA
    labelCol="label",
    maxIter=100,                  # Iteraciones suficientes para convergencia
    regParam=0.0,                 # Sin regularización (baseline)
    elasticNetParam=0.0           # No aplica sin regularización
)

print("✓ Modelo de Regresión Lineal configurado")
print(f"  • featuresCol: {lr.getFeaturesCol()}")
print(f"  • labelCol: {lr.getLabelCol()}")
print(f"  • maxIter: {lr.getMaxIter()}")
print(f"  • regParam: {lr.getRegParam()}")
print(f"  • elasticNetParam: {lr.getElasticNetParam()}")


# %% [markdown]
# ## PASO 3: Entrenar el Modelo
#
# **Nota de contexto**:
# Este notebook parte de una base académica, pero el objetivo del equipo
# es desarrollar un modelo propio adaptado al negocio, datos reales
# y decisiones analíticas específicas (SECOP / contratación pública).

# %%
print("Entrenando modelo de regresión lineal...")

# Entrenamiento del modelo con el set de entrenamiento
lr_model = lr.fit(train)

print("✓ Modelo entrenado correctamente")
print(f"  • Iteraciones completadas: {lr_model.summary.totalIterations}")
print(f"  • RMSE (train): ${lr_model.summary.rootMeanSquaredError:,.2f}")
print(f"  • R² (train): {lr_model.summary.r2:.4f}")

print("Entrenando modelo de regresión lineal...")

# Entrenar modelo
lr_model = lr.fit(train)

print("✓ Modelo entrenado correctamente")
print(f"  • Iteraciones completadas: {lr_model.summary.totalIterations}")
print(f"  • RMSE (train): ${lr_model.summary.rootMeanSquaredError:,.2f}")
print(f"  • R² (train): {lr_model.summary.r2:.4f}")


# Generar predicciones en el set de test
predictions = lr_model.transform(test)

print("✓ Predicciones generadas")


# %% [markdown]
# ## RETO 3: Interpretar R²
#
# **Pregunta**: Si R² = 0.65, ¿qué significa?
#
# **Respuesta correcta**:
# ✅ **B) El modelo explica el 65% de la varianza en los datos**
#
# **Explicación**:
# R² (coeficiente de determinación) mide qué proporción de la variabilidad
# de la variable objetivo (valor del contrato) es explicada por el modelo.
#
# Un R² = 0.65 indica que el modelo logra capturar una parte significativa
# de la estructura del fenómeno, pero aún existe un 35% de variabilidad
# explicada por factores no incluidos en el modelo.

# %%
# RESPUESTAS AL RETO

# ¿Qué significa R²?
# R² significa que el modelo explica el 65% de la variabilidad
# observada en el valor del contrato a partir de las features usadas.

# ¿Es 0.65 un buen R²?
# Depende de:
# - La complejidad del problema (económico / social suele ser ruidoso)
# - El tipo de modelo (baseline lineal vs modelos no lineales)
# - El objetivo del negocio (predicción exacta vs análisis explicativo)
#
# En este contexto:
# ✔️ Es un buen punto de partida como modelo base
# ✔️ Justifica seguir iterando con mejores features o modelos más avanzados
# ❌ No es aún un modelo final de producción


# %% [markdown]
# ## RETO 4: Análisis de Predicciones
#
# **Objetivo**: Analizar la calidad de las predicciones
#
# **Instrucciones**:
# 1. Calcular el error absoluto por predicción
# 2. Identificar las 10 predicciones con mayor error
# 3. Analizar si existe un patrón en los errores grandes



# Calcular error absoluto
predictions_with_error = predictions.withColumn(
    "absolute_error",
    spark_abs(col("prediction") - col("label"))
)

print("✓ Error absoluto calculado")

# %%
# Top 10 predicciones con mayor error absoluto
print("\n=== TOP 10 PEORES PREDICCIONES (ERROR ABSOLUTO) ===")
predictions_with_error \
    .orderBy(col("absolute_error").desc()) \
    .select(
        col("label"),
        col("prediction"),
        col("absolute_error")
    ) \
    .show(10, truncate=False)

# %%
# Calcular error porcentual
predictions_with_error = predictions_with_error.withColumn(
    "error_porcentual",
    (col("absolute_error") / col("label")) * 100
)

print("✓ Error porcentual calculado")

# %%
# ¿Hay contratos donde el error es >100%?
# Contratos con error porcentual mayor al 100%
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


# %% [markdown]
# ## PASO 5: Evaluación Formal
#
# Se evalúa el modelo usando métricas estándar de regresión:
# - RMSE: Penaliza errores grandes
# - MAE: Error promedio absoluto
# - R²: Varianza explicada en datos no vistos (test)

# %%
from pyspark.ml.evaluation import RegressionEvaluator

# Evaluadores
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

# Calcular métricas
rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

# Mostrar resultados
print("\n" + "="*60)
print("MÉTRICAS DEL MODELO (SET DE TEST)")
print("="*60)
print(f"RMSE (Test): ${rmse:,.2f}")
print(f"MAE  (Test): ${mae:,.2f}")
print(f"R²   (Test): {r2:.4f}")
print("="*60)

# %% [markdown]
# ## RETO 5: Comparar Train vs Test
#
# **Objetivo**: Detectar overfitting o underfitting comparando desempeño
# en datos de entrenamiento vs datos de prueba.
#
# **Escenarios teóricos**:
# - A) R² train = 0.9,  R² test = 0.85 → Buen ajuste (generaliza bien)
# - B) R² train = 0.6,  R² test = 0.58 → Underfitting (modelo muy simple)
# - C) R² train = 0.95, R² test = 0.45 → Overfitting (memoriza train)

# %%
from pyspark.sql.functions import abs as spark_abs

print("\n=== COMPARACIÓN TRAIN VS TEST ===")
r2_train = lr_model.summary.r2
r2_test = r2

print(f"R² Train:  {r2_train:.4f}")
print(f"R² Test:   {r2_test:.4f}")
print(f"Diferencia absoluta: {abs(r2_train - r2_test):.4f}")

# %%
# Análisis automático básico
if r2_train > 0.9 and r2_test < 0.6:
    print("\n⚠️ Posible OVERFITTING detectado")
    print("El modelo aprende muy bien el train pero generaliza mal en test.")
elif r2_train < 0.7 and r2_test < 0.7:
    print("\n⚠️ Posible UNDERFITTING detectado")
    print("El modelo es demasiado simple y no explica bien la varianza.")
else:
    print("\n✅ Buen balance entre train y test")
    print("El modelo generaliza correctamente.")

# %%
# Reflexión (completa como comentario en tu notebook):
#
# ¿Hay overfitting? → Sí, si R² train >> R² test
# ¿Hay underfitting? → Sí, si ambos R² son bajos
#
# En este experimento:
# - Resultado real:
#   R² Train = {:.4f}
#   R² Test  = {:.4f}
#
# Conclusión:
# (escribe aquí tu interpretación final)


# %% [markdown]
# ## RETO 6: Analizar Coeficientes
#
# **Objetivo**: Entender qué features son más importantes en el modelo
#
# **Pregunta**: Si un coeficiente es muy grande (positivo o negativo),
# ¿qué significa?


# Extraer coeficientes e intercepto
coefficients = lr_model.coefficients
intercept = lr_model.intercept

print(f"\nIntercept (β₀): ${intercept:,.2f}")
print(f"Número total de coeficientes: {len(coefficients)}")

# %%
# Convertir coeficientes a numpy
coef_array = np.array(coefficients)
abs_coefs = np.abs(coef_array)

# Identificar los 5 coeficientes más grandes en valor absoluto
top_5_idx = np.argsort(abs_coefs)[-5:]

print("\n=== TOP 5 FEATURES MÁS INFLUYENTES (por |coef|) ===")
for rank, idx in enumerate(reversed(top_5_idx), start=1):
    print(
        f"{rank}. Feature {idx} | "
        f"Coeficiente = {coef_array[idx]:.4f} | "
        f"|Coef| = {abs_coefs[idx]:.4f}"
    )

# %%
# Interpretación (completa como comentario en tu notebook):
#
# - Un coeficiente POSITIVO indica que:
#   A mayor valor de esa feature, mayor será la predicción del valor del contrato.
#
# - Un coeficiente NEGATIVO indica que:
#   A mayor valor de esa feature, menor será la predicción del valor del contrato.
#
# - Un coeficiente con gran magnitud (|coef| alto) significa:
#   Esa feature tiene mayor impacto en la predicción del modelo.
#
# - Coeficientes cercanos a 0:
#   La feature tiene poca influencia (posible candidata a eliminación).

# %% [markdown]
# ## RETO BONUS 1: Residuos
#
# **Objetivo**: Analizar la distribución de los errores (residuos)
#
# **Pregunta**: En un buen modelo, ¿cómo deberían distribuirse los residuos?

# %%
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import pandas as pd

# %%
# 1. Calcular residuos: residuo = label - prediction
residuals_df = predictions.withColumn(
    "residual",
    col("label") - col("prediction")
)

print("✓ Residuos calculados")

# %%
# 2. Tomar una muestra para visualización (evita problemas de memoria)
residuals_sample = (
    residuals_df
    .select("residual")
    .sample(fraction=0.1, seed=42)
    .toPandas()
)

print(f"Muestra de residuos: {len(residuals_sample):,} registros")

# %%
# 3. Histograma de residuos
plt.figure(figsize=(10, 5))
plt.hist(residuals_sample["residual"], bins=50, edgecolor="black")
plt.axvline(x=0, color="red", linestyle="--", label="Cero")
plt.xlabel("Residuo (label - prediction)")
plt.ylabel("Frecuencia")
plt.title("Distribución de Residuos del Modelo")
plt.legend()
plt.grid(True)

# Guardar gráfico
output_path = "/opt/spark-data/processed/residuals_distribution.png"
plt.savefig(output_path)
plt.close()

print(f"✓ Gráfico de residuos guardado en: {output_path}")

# %%
# 4. Estadísticas básicas de residuos
print("\n=== ESTADÍSTICAS DE RESIDUOS ===")
print(residuals_sample.describe())


# %% [markdown]
# ## RETO BONUS 2: Feature Importance Aproximado
#
# **Objetivo**: Identificar las features más importantes
#
# **Método**: Eliminar una feature a la vez y medir el impacto en R²
#
# **Nota**: Computacionalmente costoso. Usar SOLO en datasets pequeños.



# %%
# Evaluador R²
evaluator_r2 = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="r2"
)

# %%
# R² del modelo completo (baseline)
predictions_full = lr_model.transform(test)
r2_full = evaluator_r2.evaluate(predictions_full)

print(f"R² modelo completo: {r2_full:.4f}")

# %%
# Obtener número de features
num_features = len(lr_model.coefficients)
print(f"Número de features: {num_features}")

# %%
# Almacenar impacto por feature
feature_importance = []

# %%
# Iterar quitando una feature a la vez
for i in range(num_features):
    print(f"Evaluando feature {i}...")

    # Índices de features SIN la i
    remaining_indices = [j for j in range(num_features) if j != i]

    # UDF para remover una posición del vector
    from pyspark.sql.functions import udf
    from pyspark.ml.linalg import Vectors, VectorUDT

    remove_feature_udf = udf(
        lambda v: Vectors.dense([v[j] for j in remaining_indices]),
        VectorUDT()
    )

    # Crear nuevo dataframe sin la feature i
    train_reduced = train.withColumn(
        "features_reduced",
        remove_feature_udf(col("features"))
    )

    test_reduced = test.withColumn(
        "features_reduced",
        remove_feature_udf(col("features"))
    )

    # Entrenar modelo reducido
    lr_reduced = LinearRegression(
        featuresCol="features_reduced",
        labelCol="label",
        maxIter=100
    )

    model_reduced = lr_reduced.fit(train_reduced)

    # Evaluar
    predictions_reduced = model_reduced.transform(test_reduced)
    r2_reduced = evaluator_r2.evaluate(predictions_reduced)

    # Impacto
    delta_r2 = r2_full - r2_reduced

    feature_importance.append((i, delta_r2))

# %%
# Ordenar por mayor impacto
feature_importance_sorted = sorted(
    feature_importance,
    key=lambda x: x[1],
    reverse=True
)

print("\n=== FEATURE IMPORTANCE APROXIMADO (por caída en R²) ===")
for idx, impact in feature_importance_sorted[:10]:
    print(f"Feature {idx}: caída R² = {impact:.4f}")

# %%
# Guardar modelo
model_path = "/opt/spark-data/processed/linear_regression_model"
lr_model.write().overwrite().save(model_path)
print(f"\n✓ Modelo guardado en: {model_path}")

# %%
# Guardar predicciones
predictions_path = "/opt/spark-data/processed/predictions_lr.parquet"
predictions.write.mode("overwrite").parquet(predictions_path)
print(f"✓ Predicciones guardadas en: {predictions_path}")

# %%
print("\n" + "="*60)
print("RESUMEN REGRESIÓN LINEAL")
print("="*60)
print(f"✓ Modelo entrenado con {train.count():,} registros")
print(f"✓ Evaluado con {test.count():,} registros")
print(f"✓ RMSE: ${rmse:,.2f}")
print(f"✓ R²: {r2:.4f}")
print(f"✓ Próximo paso: Probar regularización (notebook 07)")
print("="*60)

# %%
spark.stop()
