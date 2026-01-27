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

# %%
spark = SparkSession.builder \
    .appName("SECOP_RegresionLineal") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %%
# Cargar datos
df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")

# Renombrar columnas para consistencia
df = df.withColumnRenamed("valor_del_contrato_num", "label") \
       .withColumnRenamed("features_pca", "features")

# Filtrar valores nulos
df = df.filter(col("label").isNotNull())
print(f"Registros: {df.count():,}")

# %% [markdown]
# ## RETO 1: Train/Test Split Strategy
#
# **Pregunta**: ¿Qué proporción usarías para train vs test?
#
# **Opciones**:
# - A) 50/50 - Máxima validación
# - B) 70/30 - Balance clásico
# - C) 80/20 - Más datos para entrenar
# - D) 90/10 - Máximo entrenamiento
#
# **Consideración**: ¿Qué pasa si tienes 1 millón de registros vs 1000?
#
# **Justifica tu decisión**

# %%
# TODO: Define tu estrategia de split
# Pista: Usa randomSplit con semilla (seed) para reproducibilidad

train_ratio = 0.7  # TODO: Ajusta según tu decisión
test_ratio = 0.3

train, test = df.randomSplit([train_ratio, test_ratio], seed=42)

print(f"Train: {train.count():,} registros ({train_ratio*100:.0f}%)")
print(f"Test: {test.count():,} registros ({test_ratio*100:.0f}%)")

# TODO: Responde:
# ¿Por qué es importante usar seed=42?
# Respuesta:

# %% [markdown]
# ## RETO 2: Configurar el Modelo
#
# **Objetivo**: Configurar LinearRegression con los parámetros correctos
#
# **Parámetros clave**:
# - featuresCol: Columna de features
# - labelCol: Columna objetivo
# - maxIter: Iteraciones máximas (¿10, 100, 1000?)
# - regParam: Regularización (0.0 = sin regularización)
# - elasticNetParam: Tipo de regularización (0=L2, 1=L1)

# %%
# TODO: Crea el modelo de regresión lineal
# Empezamos SIN regularización (modelo baseline)

lr = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,           # TODO: ¿Es suficiente?
    regParam=0.0,          # Sin regularización
    elasticNetParam=0.0    # No aplica sin regParam
)

print("✓ Modelo configurado")
print(f"  maxIter: {lr.getMaxIter()}")
print(f"  regParam: {lr.getRegParam()}")

# %% [markdown]
# ## PASO 3: Entrenar el Modelo

# %%
print("Entrenando modelo de regresión lineal...")
lr_model = lr.fit(train)

print("✓ Modelo entrenado")
print(f"  Iteraciones completadas: {lr_model.summary.totalIterations}")
print(f"  RMSE (train): ${lr_model.summary.rootMeanSquaredError:,.2f}")
print(f"  R² (train): {lr_model.summary.r2:.4f}")

# %% [markdown]
# ## RETO 3: Interpretar R²
#
# **Pregunta**: Si R² = 0.65, ¿qué significa?
#
# **Opciones**:
# - A) El modelo es 65% preciso
# - B) El modelo explica 65% de la varianza en los datos
# - C) El modelo tiene 65% de error
# - D) El modelo está 35% equivocado
#
# **Responde y explica**

# %%
# TODO: Escribe tu respuesta
# R² significa:

# TODO: ¿Es 0.65 un buen R²?
# Depende de:

# %% [markdown]
# ## PASO 4: Predicciones en Test

# %%
predictions = lr_model.transform(test)

print("\n=== PREDICCIONES EN TEST ===")
predictions.select("label", "prediction").show(10)

# %% [markdown]
# ## RETO 4: Análisis de Predicciones
#
# **Objetivo**: Analizar la calidad de las predicciones
#
# **Instrucciones**:
# 1. Calcula el error absoluto por cada predicción
# 2. Identifica las 10 predicciones con mayor error
# 3. ¿Hay un patrón en los errores grandes?

# %%
# TODO: Calcula el error absoluto
# Pista: error = abs(prediction - label)

from pyspark.sql.functions import abs as spark_abs

predictions_with_error = predictions.withColumn(
    "absolute_error",
    spark_abs(col("prediction") - col("label"))
)

# TODO: Encuentra las 10 peores predicciones
# predictions_with_error.orderBy(col("absolute_error").desc()).select(...).show(10)

# TODO: Calcula el error porcentual
# error_porcentual = (absolute_error / label) * 100
# ¿Hay contratos donde el error es >100%?

# %% [markdown]
# ## PASO 5: Evaluación Formal

# %%
# Crear evaluadores para diferentes métricas
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

print("\n" + "="*60)
print("MÉTRICAS DEL MODELO")
print("="*60)
print(f"RMSE (Test): ${rmse:,.2f}")
print(f"MAE (Test):  ${mae:,.2f}")
print(f"R² (Test):   {r2:.4f}")
print("="*60)

# %% [markdown]
# ## RETO 5: Comparar Train vs Test
#
# **Objetivo**: Detectar overfitting o underfitting
#
# **Pregunta**: ¿Qué indica cada escenario?
#
# **Escenarios**:
# - A) R² train = 0.9, R² test = 0.85 → ¿Qué pasa?
# - B) R² train = 0.6, R² test = 0.58 → ¿Qué pasa?
# - C) R² train = 0.95, R² test = 0.45 → ¿Qué pasa?
#
# **Compara tus resultados**

# %%
print("\n=== COMPARACIÓN TRAIN VS TEST ===")
print(f"R² Train:  {lr_model.summary.r2:.4f}")
print(f"R² Test:   {r2:.4f}")
print(f"Diferencia: {abs(lr_model.summary.r2 - r2):.4f}")

# TODO: Analiza:
# ¿Hay overfitting? (Sí/No)
# ¿Hay underfitting? (Sí/No)
# Justifica:

# %% [markdown]
# ## RETO 6: Analizar Coeficientes
#
# **Objetivo**: Entender qué features son más importantes
#
# **Pregunta**: Si un coeficiente es muy grande (positivo o negativo),
# ¿qué significa?

# %%
coefficients = lr_model.coefficients
intercept = lr_model.intercept

print(f"\nIntercept: ${intercept:,.2f}")
print(f"Número de coeficientes: {len(coefficients)}")

# TODO: Encuentra los 5 coeficientes más grandes (en valor absoluto)
# Pista: Usa numpy para ordenar por abs(coef)

import numpy as np

coef_array = np.array(coefficients)
abs_coefs = np.abs(coef_array)
top_5_idx = np.argsort(abs_coefs)[-5:]

print("\n=== TOP 5 FEATURES MÁS INFLUYENTES ===")
for i, idx in enumerate(reversed(top_5_idx)):
    print(f"{i+1}. Feature {idx}: coef = {coef_array[idx]:.4f}")

# TODO: Interpreta:
# ¿Qué significa un coeficiente positivo vs negativo?
# Respuesta:

# %% [markdown]
# ## RETO BONUS 1: Residuos
#
# **Objetivo**: Analizar la distribución de los errores (residuos)
#
# **Pregunta**: En un buen modelo, ¿cómo deberían distribuirse los residuos?
#
# **Instrucciones**:
# 1. Calcula residuo = label - prediction
# 2. Genera un histograma de residuos
# 3. ¿Están centrados en cero?
# 4. ¿Hay sesgo (bias)?

# %%
# TODO: Implementa el análisis de residuos
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # Calcular residuos
# residuals_df = predictions.withColumn("residual", col("label") - col("prediction"))
#
# # Convertir muestra a Pandas para graficar
# residuals_sample = residuals_df.select("residual").sample(0.1).toPandas()
#
# # Histograma
# plt.figure(figsize=(10, 5))
# plt.hist(residuals_sample['residual'], bins=50, edgecolor='black')
# plt.xlabel('Residuo (Error)')
# plt.ylabel('Frecuencia')
# plt.title('Distribución de Residuos')
# plt.axvline(x=0, color='red', linestyle='--', label='Cero')
# plt.legend()
# plt.savefig('/opt/spark-data/processed/residuals_distribution.png')
# print("Gráfico guardado en /opt/spark-data/processed/residuals_distribution.png")

# %% [markdown]
# ## RETO BONUS 2: Feature Importance Aproximado
#
# **Objetivo**: Identificar las features más importantes
#
# **Método**: Entrenar modelo quitando una feature a la vez,
# medir impacto en R²
#
# **Nota**: Este experimento es computacionalmente costoso,
# solo para datasets pequeños

# %%
# TODO: (Opcional) Implementa el análisis de feature importance
# Este reto es avanzado, requiere:
# 1. Iterar sobre cada feature
# 2. Entrenar modelo sin esa feature
# 3. Comparar R² con el modelo completo
# 4. Features que causan mayor caída en R² son más importantes

# %% [markdown]
# ## Preguntas de Reflexión
#
# 1. **¿Por qué usar RMSE en lugar de solo MAE?**
#    Respuesta:
#
# 2. **Si todas las predicciones fueran = promedio de labels, ¿cuál sería el R²?**
#    Respuesta:
#
# 3. **¿Cuándo preferirías optimizar para RMSE vs MAE?**
#    Respuesta:
#
# 4. **¿Cómo mejorarías este modelo? (menciona al menos 3 estrategias)**
#    Respuestas:
#    -
#    -
#    -

# %%
# TODO: Escribe tus respuestas arriba

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
