# %% [markdown]
# # Notebook 07: Regularización L1, L2 y ElasticNet
#
# **Sección 14**: Prevención de overfitting con regularización
#
# **Objetivo**: Comparar Ridge (L2), Lasso (L1) y ElasticNet
#
# ## Conceptos clave:
# - **Ridge (L2)**: regParam > 0, elasticNetParam = 0
#   - Penaliza coeficientes grandes, NO los elimina
# - **Lasso (L1)**: regParam > 0, elasticNetParam = 1
#   - Puede eliminar features (coeficientes = 0)
# - **ElasticNet**: regParam > 0, elasticNetParam ∈ (0, 1)
#   - Combinación de L1 y L2
#
# ## Actividades:
# 1. Entrenar modelos con diferentes regularizaciones
# 2. Comparar resultados
# 3. Identificar el mejor modelo

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

train, test = df.randomSplit([0.7, 0.3], seed=42)

print(f"Train: {train.count():,}")
print(f"Test: {test.count():,}")

# %%
# RETO 1: Entender la Regularización

# Pregunta conceptual:
# ¿Por qué necesitamos regularización?

# Escenario:
# R² train = 0.95
# R² test  = 0.45

# Respuesta:

# El escenario indica:
# - A) El modelo está underfitting
# - B) El modelo está overfitting
# - C) El modelo es perfecto
# - D) Necesitas más features
# ✅ Opción B) El modelo está OVERFITTING

# Explicación:
# El modelo se ajusta excesivamente a los datos de entrenamiento,
# capturando ruido y patrones específicos del train que no se
# generalizan a datos nuevos (test).

# Evidencia clara de overfitting:
# - Desempeño muy alto en train (R² = 0.95)
# - Caída fuerte en test (R² = 0.45)

# ¿Cómo ayuda la regularización en este caso?
#
# La regularización:
# - Penaliza coeficientes muy grandes
# - Reduce la complejidad del modelo
# - Evita que el modelo dependa excesivamente de pocas features
# - Obliga al modelo a aprender patrones más generales
#
# En regresión lineal:
# - L2 (Ridge) reduce la magnitud de los coeficientes
# - L1 (Lasso) puede llevar coeficientes a cero (selección de features)
#
# Resultado esperado al aplicar regularización:
# - R² train ↓ ligeramente
# - R² test ↑
# - Mejor capacidad de generalización
#
# Conclusión:
# La regularización es clave para controlar el overfitting
# y mejorar el desempeño del modelo en datos no vistos.


# %%
# RETO 2: Configurar el Evaluador

# Objetivo:
# Crear un evaluador para comparar modelos de regresión

# Métrica elegida: RMSE

# Justificación:
# - RMSE penaliza más los errores grandes
# - Es especialmente útil cuando errores grandes son más costosos
#   (ej. sobre/infraestimar valores de contratos)
# - Mantiene las mismas unidades de la variable objetivo,
#   lo que facilita interpretación en términos de negocio

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

print("✓ Evaluador configurado correctamente")
print(f"  • Métrica seleccionada: {evaluator.getMetricName()}")

# Reflexión:
#
# ¿Cambiarías esta métrica?
#
# - Usaría MAE si:
#   * Todos los errores tienen el mismo impacto
#   * Quiero una métrica más robusta a outliers
#
# - Usaría R² si:
#   * Quiero comparar capacidad explicativa entre modelos
#   * El objetivo es análisis y no solo predicción
#
# En este proyecto:
# ✔️ RMSE es una buena métrica principal
# ✔️ MAE y R² pueden usarse como métricas complementarias


# %%
# RETO 3: Experimento de Regularización

from pyspark.ml.regression import LinearRegression

# 1. Definir valores de regularización a probar
# regParam = lambda (fuerza de regularización)
reg_params = [0.0, 0.01, 0.1, 1.0, 10.0]

# elasticNetParam = alpha (tipo de regularización)
# 0.0 = Ridge (L2)
# 0.5 = ElasticNet
# 1.0 = Lasso (L1)
elastic_params = [0.0, 0.5, 1.0]

print(f"Combinaciones totales a entrenar: {len(reg_params) * len(elastic_params)}")

# %%
# 2. Bucle de experimentación
resultados = []

for reg in reg_params:
    for elastic in elastic_params:

        lr = LinearRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=100,
            regParam=reg,
            elasticNetParam=elastic
        )

        # Entrenar modelo
        model = lr.fit(train)

        # Predicciones en test
        predictions = model.transform(test)

        # Evaluar RMSE en test
        rmse_test = evaluator.evaluate(predictions)

        # Identificar tipo de regularización
        if reg == 0.0:
            reg_type = "Sin regularización"
        elif elastic == 0.0:
            reg_type = "Ridge (L2)"
        elif elastic == 1.0:
            reg_type = "Lasso (L1)"
        else:
            reg_type = "ElasticNet"

        # Guardar resultados
        resultados.append({
            "regParam": reg,
            "elasticNetParam": elastic,
            "tipo": reg_type,
            "rmse_test": rmse_test,
            "rmse_train": model.summary.rootMeanSquaredError,
            "r2_train": model.summary.r2
        })

        print(
            f"{reg_type:20s} | "
            f"λ={reg:5.2f} | "
            f"α={elastic:.1f} | "
            f"RMSE Test: ${rmse_test:,.2f}"
        )

# %%
# 3. Convertir resultados a DataFrame para análisis
import pandas as pd

results_df = pd.DataFrame(resultados)
print("\n=== RESUMEN DE RESULTADOS ===")
results_df.sort_values("rmse_test").head(10)


# %%
# RETO 4: Analizar Resultados

import pandas as pd

# 1. Convertir resultados a DataFrame de pandas
df_resultados = pd.DataFrame(resultados)

print("\n=== RESULTADOS COMPLETOS DE LOS MODELOS ===")
print(df_resultados.sort_values("rmse_test").to_string(index=False))

# %%
# 2. Identificar el mejor modelo (menor RMSE en test)
mejor_modelo = df_resultados.loc[df_resultados["rmse_test"].idxmin()]

print("\n" + "=" * 60)
print("MEJOR MODELO SEGÚN RMSE EN TEST")
print("=" * 60)
print(f"Tipo de regularización: {mejor_modelo['tipo']}")
print(f"regParam (λ):          {mejor_modelo['regParam']}")
print(f"elasticNetParam (α):   {mejor_modelo['elasticNetParam']}")
print(f"RMSE Train:            ${mejor_modelo['rmse_train']:,.2f}")
print(f"RMSE Test:             ${mejor_modelo['rmse_test']:,.2f}")
print(f"R² Train:              {mejor_modelo['r2_train']:.4f}")
print("=" * 60)

# %%
# 3. Comparar RMSE train vs test para detectar overfitting
df_resultados["rmse_gap"] = df_resultados["rmse_test"] - df_resultados["rmse_train"]

print("\n=== MODELOS ORDENADOS POR GAP (TEST - TRAIN) ===")
print(
    df_resultados
    .sort_values("rmse_gap")
    [["tipo", "regParam", "elasticNetParam", "rmse_train", "rmse_test", "rmse_gap"]]
    .to_string(index=False)
)

# %%
# 4. Análisis conceptual (responde en markdown o comentario)

# ¿El mejor modelo es siempre el de menor RMSE en test?
# NO necesariamente.
#
# Otros factores a considerar:
# - Diferencia entre RMSE train y test (estabilidad / generalización)
# - Simplicidad del modelo (menor regularización vs complejidad)
# - Interpretabilidad (Lasso elimina features, Ridge no)
# - Robustez frente a nuevos datos
# - Coherencia con el negocio
#
# En práctica, suele preferirse:
# ✔ Un modelo ligeramente peor en RMSE pero más estable
# ✔ Un modelo más simple y explicable


# %%
# RETO 5: Comparar Overfitting
#
# Objetivo:
# Analizar la brecha entre RMSE de train y RMSE de test
# para identificar overfitting y underfitting según la regularización.

import numpy as np

# Definir un umbral práctico para considerar overfitting
# (ajústalo según escala del problema)
threshold = 0.10 * df_resultados["rmse_test"].mean()

print("\n=== ANÁLISIS DE OVERFITTING (BRECHA TRAIN vs TEST) ===")

for _, row in df_resultados.iterrows():
    gap = row["rmse_test"] - row["rmse_train"]

    if gap > threshold:
        estado = "Overfitting"
    elif gap < -threshold:
        estado = "Underfitting"
    else:
        estado = "Balanceado"

    print(
        f"{row['tipo']:25s} | "
        f"λ={row['regParam']:6.2f} | "
        f"α={row['elasticNetParam']:3.1f} | "
        f"RMSE Train=${row['rmse_train']:,.2f} | "
        f"RMSE Test=${row['rmse_test']:,.2f} | "
        f"Gap=${gap:,.2f} | "
        f"Estado: {estado}"
    )

# %%
# Análisis agregado por tipo de regularización
print("\n=== PROMEDIO DE GAP POR TIPO DE REGULARIZACIÓN ===")

gap_por_tipo = (
    df_resultados
    .assign(gap=lambda d: d["rmse_test"] - d["rmse_train"])
    .groupby("tipo")["gap"]
    .mean()
    .sort_values()
)

for tipo, gap_mean in gap_por_tipo.items():
    print(f"{tipo:25s} | Gap promedio: ${gap_mean:,.2f}")

# %%
# Respuestas conceptuales (escribe también en Markdown si lo deseas):
#
# 1. ¿Qué regularización reduce más el overfitting?
#    → Normalmente Ridge (L2) y ElasticNet con λ moderado,
#      porque reducen la magnitud de los coeficientes sin
#      eliminar completamente variables relevantes.
#
# 2. ¿Hay trade-off entre overfitting y rendimiento?
#    → Sí. A mayor regularización:
#       - ↓ Overfitting
#       - ↑ Bias (riesgo de underfitting)
#
# 3. Interpretación de escenarios:
#    - regParam = 0.0, RMSE_train bajo y RMSE_test alto → Overfitting
#    - regParam = 10.0, RMSE_train y RMSE_test altos → Underfitting
#
# La clave es encontrar un punto intermedio (sweet spot).


# %%
# RETO 6: Entrenar Modelo Final
#
# Objetivo:
# Entrenar el modelo definitivo usando los mejores hiperparámetros
# encontrados durante el experimento de regularización.

from pyspark.ml.regression import LinearRegression

# 1. Identificar el mejor modelo (menor RMSE en test)
mejor_modelo = df_resultados.loc[df_resultados["rmse_test"].idxmin()]

best_reg = mejor_modelo["regParam"]
best_elastic = mejor_modelo["elasticNetParam"]
best_tipo = mejor_modelo["tipo"]

print("\n=== MEJOR CONFIGURACIÓN ENCONTRADA ===")
print(f"Tipo de regularización: {best_tipo}")
print(f"regParam (λ): {best_reg}")
print(f"elasticNetParam (α): {best_elastic}")
print(f"RMSE Test: ${mejor_modelo['rmse_test']:,.2f}")

# %%
# 2. Entrenar el modelo final con los mejores hiperparámetros
print("\nEntrenando modelo final...")

lr_final = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=best_reg,
    elasticNetParam=best_elastic
)

modelo_final = lr_final.fit(train)

print("✓ Modelo final entrenado correctamente")
print(f"  RMSE Train: ${modelo_final.summary.rootMeanSquaredError:,.2f}")
print(f"  R² Train:   {modelo_final.summary.r2:.4f}")

# %%
# 3. Evaluar el modelo final en el set de test
predictions_final = modelo_final.transform(test)
rmse_final = evaluator.evaluate(predictions_final)

print("\n=== EVALUACIÓN FINAL (TEST) ===")
print(f"RMSE Test Final: ${rmse_final:,.2f}")

# %%
# 4. Guardar el modelo entrenado
model_path = "/opt/spark-data/processed/regularized_linear_regression_model"

modelo_final.write().overwrite().save(model_path)

print(f"\n✓ Modelo final guardado en: {model_path}")

# %%
# Conclusión (para documentar en Markdown):
#
# - Se seleccionó el modelo con menor RMSE en test
# - Se redujo el overfitting respecto al modelo sin regularización
# - El modelo final está listo para:
#   • scoring batch
#   • comparación con modelos no lineales
#   • despliegue productivo


# %%
# RETO BONUS: Efecto de Lambda en los Coeficientes
#
# Objetivo:
# Analizar cómo el parámetro lambda (regParam) en Lasso (L1)
# afecta la cantidad de coeficientes que se vuelven exactamente 0.

import numpy as np
from pyspark.ml.regression import LinearRegression

print("\n=== EFECTO DE LAMBDA (LASSO) EN LOS COEFICIENTES ===")

reg_values = [0.01, 0.1, 1.0, 10.0]

coef_analysis = []

for reg in reg_values:
    print(f"\nEntrenando Lasso con λ = {reg}")

    lr_lasso = LinearRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        regParam=reg,
        elasticNetParam=1.0   # LASSO (L1)
    )

    model_lasso = lr_lasso.fit(train)

    # Extraer coeficientes
    coefs = np.array(model_lasso.coefficients)

    # Contar coeficientes ~ 0
    zeros = np.sum(np.abs(coefs) < 1e-6)

    # Evaluar desempeño
    rmse_test = evaluator.evaluate(model_lasso.transform(test))

    coef_analysis.append({
        "regParam": reg,
        "coef_total": len(coefs),
        "coef_cero": int(zeros),
        "rmse_test": rmse_test
    })

    print(
        f"λ={reg:5.2f} | "
        f"Coeficientes en 0: {zeros}/{len(coefs)} | "
        f"RMSE Test: ${rmse_test:,.2f}"
    )

# %%
# Resumen tabular
import pandas as pd

df_coef_analysis = pd.DataFrame(coef_analysis)

print("\n=== RESUMEN EFECTO DE LAMBDA ===")
print(df_coef_analysis.to_string(index=False))

# %%
# Interpretación (documentar en Markdown):
#
# - A mayor λ (regParam), mayor número de coeficientes exactamente 0
# - Lasso realiza selección automática de features
# - Modelos con λ alto son más simples pero pueden perder capacidad predictiva

# %%
# Pregunta conceptual:
#
# ¿Por qué Lasso puede poner coeficientes en 0 y Ridge no?
#
# Respuesta (para Markdown):
#
# - Lasso (L1) usa una penalización basada en el valor absoluto del coeficiente
# - Esto genera esquinas en la función de optimización
# - En esas esquinas, el óptimo puede caer exactamente en 0
#
# - Ridge (L2) penaliza el cuadrado del coeficiente
# - Reduce magnitudes pero rara vez las lleva exactamente a 0
# - Por eso Ridge NO hace selección de variables explícita

# %%
print("\n" + "="*60)
print("RESUMEN REGULARIZACIÓN")
print("="*60)
print("Verifica que hayas completado:")
print("  [ ] Entendido diferencia entre L1, L2 y ElasticNet")
print("  [ ] Experimentado con múltiples combinaciones")
print("  [ ] Identificado el mejor modelo")
print("  [ ] Analizado overfitting vs underfitting")
print("  [ ] Guardado modelo final")
print("="*60)

# %%
spark.stop()
