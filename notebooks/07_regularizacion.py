# ============================================================
# NOTEBOOK 07: Regularización L1, L2 y ElasticNet
# ============================================================

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
import pandas as pd

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

train, test = df.randomSplit([0.7, 0.3], seed=42)

print(f"Train: {train.count():,}")
print(f"Test: {test.count():,}")

# ------------------------------------------------------------
# RETO 1: ENTENDER LA REGULARIZACIÓN
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO 1: ENTENDER LA REGULARIZACIÓN")
print("-"*60)

print(
    "\nPregunta conceptual:\n"
    "¿Por qué necesitamos regularización?\n"
)

print(
    "Escenario observado:\n"
    "- R² Train = 0.95\n"
    "- R² Test  = 0.45\n"
)

print(
    "\nOpciones:\n"
    "A) El modelo está underfitting\n"
    "B) El modelo está overfitting\n"
    "C) El modelo es perfecto\n"
    "D) Necesitas más features\n"
)

print("Respuesta correcta:")
print("✅ Opción B) El modelo está OVERFITTING")

print(
    "\nExplicación:\n"
    "El modelo se ajusta excesivamente a los datos de entrenamiento,\n"
    "capturando ruido y patrones específicos del train que no se\n"
    "generalizan a datos nuevos (test)."
)

print(
    "\nEvidencia clara de overfitting:\n"
    "- Desempeño muy alto en train (R² = 0.95)\n"
    "- Caída fuerte en test  (R² = 0.45)"
)

print(
    "\n¿Cómo ayuda la regularización?\n"
    "- Penaliza coeficientes muy grandes\n"
    "- Reduce la complejidad del modelo\n"
    "- Evita dependencia excesiva de pocas features\n"
    "- Obliga al modelo a aprender patrones más generales"
)

print(
    "\nTipos de regularización en regresión:\n"
    "- L2 (Ridge): Reduce la magnitud de los coeficientes\n"
    "- L1 (Lasso): Puede llevar coeficientes a cero (selección de features)"
)

print(
    "\nResultado esperado al aplicar regularización:\n"
    "- R² Train ↓ ligeramente\n"
    "- R² Test  ↑\n"
    "- Mejor capacidad de generalización"
)

print(
    "\nConclusión:\n"
    "La regularización es clave para controlar el overfitting\n"
    "y mejorar el desempeño del modelo en datos no vistos."
)

# ------------------------------------------------------------
# RETO 2: CONFIGURAR EL EVALUADOR
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO 2: CONFIGURAR EL EVALUADOR")
print("-"*60)

print(
    "\nObjetivo:\n"
    "Crear un evaluador para comparar modelos de regresión."
)

print(
    "\nMétrica elegida: RMSE\n"
)

print(
    "Justificación:\n"
    "- RMSE penaliza más los errores grandes\n"
    "- Es útil cuando errores grandes son más costosos\n"
    "  (ej. sobre/infraestimar valores de contratos)\n"
    "- Mantiene las mismas unidades de la variable objetivo,\n"
    "  facilitando la interpretación para negocio"
)

evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

print("\n✓ Evaluador configurado correctamente")
print(f"  • Métrica seleccionada: {evaluator.getMetricName()}")

print(
    "\nReflexión sobre métricas alternativas:\n"
    "- Usaría MAE si:\n"
    "  • Todos los errores tienen el mismo impacto\n"
    "  • Quiero una métrica más robusta a outliers\n\n"
    "- Usaría R² si:\n"
    "  • Quiero comparar capacidad explicativa entre modelos\n"
    "  • El objetivo es análisis y no solo predicción"
)

print(
    "\nDecisión en este proyecto:\n"
    "✔️ RMSE como métrica principal\n"
    "✔️ MAE y R² como métricas complementarias"
)


# ------------------------------------------------------------
# RETO 3: EXPERIMENTO DE REGULARIZACIÓN
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO 3: EXPERIMENTO DE REGULARIZACIÓN")
print("-"*60)

print(
    "\nObjetivo del experimento:\n"
    "Evaluar cómo diferentes configuraciones de regularización\n"
    "afectan el desempeño del modelo y su capacidad de generalización."
)

# ------------------------------------------------------------
# 1. Definir valores de regularización
# ------------------------------------------------------------

reg_params = [0.0, 0.01, 0.1, 1.0, 10.0]
elastic_params = [0.0, 0.5, 1.0]

print(
    "\nParámetros evaluados:\n"
    f"- Valores de regParam (λ): {reg_params}\n"
    f"- Valores de elasticNetParam (α): {elastic_params}\n"
)

print(
    f"Combinaciones totales a entrenar: {len(reg_params) * len(elastic_params)}"
)

# ------------------------------------------------------------
# 2. Bucle de experimentación
# ------------------------------------------------------------

resultados = []

print("\n=== INICIO DE EXPERIMENTOS ===")

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
        preds = model.transform(test)

        # Evaluar RMSE en test
        rmse_test = evaluator.evaluate(preds)

        # Identificar tipo de regularización
        if reg == 0.0:
            reg_type = "Sin regularización"
        elif elastic == 0.0:
            reg_type = "Ridge (L2)"
        elif elastic == 1.0:
            reg_type = "Lasso (L1)"
        else:
            reg_type = "ElasticNet"

        resultados.append({
            "tipo": reg_type,
            "regParam": reg,
            "elasticNetParam": elastic,
            "rmse_train": model.summary.rootMeanSquaredError,
            "rmse_test": rmse_test,
            "r2_train": model.summary.r2
        })

        print(
            f"{reg_type:20s} | "
            f"λ={reg:5.2f} | "
            f"α={elastic:.1f} | "
            f"RMSE Test: ${rmse_test:,.2f}"
        )

# ------------------------------------------------------------
# RETO 4: ANALIZAR RESULTADOS
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO 4: ANÁLISIS DE RESULTADOS")
print("-"*60)

df_resultados = pd.DataFrame(resultados)

print("\n=== RESULTADOS COMPLETOS (ORDENADOS POR RMSE TEST) ===")
print(
    df_resultados
    .sort_values("rmse_test")
    .to_string(index=False)
)

# ------------------------------------------------------------
# Identificar mejor modelo
# ------------------------------------------------------------

mejor_modelo = df_resultados.loc[df_resultados["rmse_test"].idxmin()]

print("\n" + "="*60)
print("MEJOR MODELO SEGÚN RMSE EN TEST")
print("="*60)
print(f"Tipo de regularización: {mejor_modelo['tipo']}")
print(f"regParam (λ):          {mejor_modelo['regParam']}")
print(f"elasticNetParam (α):   {mejor_modelo['elasticNetParam']}")
print(f"RMSE Train:            ${mejor_modelo['rmse_train']:,.2f}")
print(f"RMSE Test:             ${mejor_modelo['rmse_test']:,.2f}")
print(f"R² Train:              {mejor_modelo['r2_train']:.4f}")
print("="*60)

# ------------------------------------------------------------
# Análisis de estabilidad: gap train vs test
# ------------------------------------------------------------

df_resultados["rmse_gap"] = df_resultados["rmse_test"] - df_resultados["rmse_train"]

print("\n=== MODELOS ORDENADOS POR GAP (TEST - TRAIN) ===")
print(
    df_resultados
    .sort_values("rmse_gap")
    [["tipo", "regParam", "elasticNetParam", "rmse_train", "rmse_test", "rmse_gap"]]
    .to_string(index=False)
)

# ------------------------------------------------------------
# Reflexión conceptual
# ------------------------------------------------------------

print(
    "\nReflexión clave:\n"
    "¿El mejor modelo es siempre el de menor RMSE en test?\n"
    "❌ NO necesariamente.\n"
)

print(
    "Otros factores a considerar:\n"
    "- Diferencia entre RMSE train y test (estabilidad)\n"
    "- Simplicidad del modelo\n"
    "- Interpretabilidad (Lasso vs Ridge)\n"
    "- Robustez frente a nuevos datos\n"
    "- Coherencia con el negocio\n"
)

print(
    "\nDecisión práctica habitual:\n"
    "✔ Preferir un modelo ligeramente peor en RMSE pero más estable\n"
    "✔ Priorizar modelos simples, explicables y consistentes\n"
)


# ------------------------------------------------------------
# RETO 5: COMPARAR OVERFITTING
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO 5: COMPARAR OVERFITTING")
print("-"*60)

print(
    "\nObjetivo del análisis:\n"
    "Evaluar la brecha entre RMSE de entrenamiento y RMSE de prueba\n"
    "para identificar escenarios de overfitting, underfitting\n"
    "y modelos balanceados según la regularización aplicada."
)

# Umbral práctico para clasificar el gap
threshold = 0.10 * df_resultados["rmse_test"].mean()

print(
    f"\nUmbral de referencia para el gap (10% del RMSE promedio): "
    f"${threshold:,.2f}"
)

print("\n=== ANÁLISIS MODELO A MODELO ===")

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

# ------------------------------------------------------------
# Análisis agregado por tipo de regularización
# ------------------------------------------------------------

print("\n" + "-"*60)
print("ANÁLISIS AGREGADO POR TIPO DE REGULARIZACIÓN")
print("-"*60)

gap_por_tipo = (
    df_resultados
    .assign(gap=lambda d: d["rmse_test"] - d["rmse_train"])
    .groupby("tipo")["gap"]
    .mean()
    .sort_values()
)

for tipo, gap_mean in gap_por_tipo.items():
    print(f"{tipo:25s} | Gap promedio: ${gap_mean:,.2f}")

# ------------------------------------------------------------
# Respuestas conceptuales
# ------------------------------------------------------------

print(
    "\nConclusiones conceptuales:\n"
    "1) ¿Qué regularización reduce mejor el overfitting?\n"
    "→ Normalmente Ridge (L2) y ElasticNet con λ moderado,\n"
    "  ya que reducen la magnitud de los coeficientes sin\n"
    "  eliminar completamente variables relevantes.\n"
)

print(
    "2) ¿Existe un trade-off entre overfitting y rendimiento?\n"
    "→ Sí. A mayor regularización:\n"
    "   - Disminuye el overfitting\n"
    "   - Aumenta el sesgo (riesgo de underfitting)\n"
)

print(
    "3) Interpretación de escenarios típicos:\n"
    "- regParam = 0.0, RMSE_train bajo y RMSE_test alto → Overfitting\n"
    "- regParam = 10.0, RMSE_train y RMSE_test altos → Underfitting\n"
)

print(
    "\nConclusión clave:\n"
    "La clave no es maximizar desempeño en train,\n"
    "sino encontrar un punto intermedio (sweet spot)\n"
    "que generalice bien a datos no vistos."
)

# ------------------------------------------------------------
# RETO 6: ENTRENAR MODELO FINAL
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO 6: ENTRENAR MODELO FINAL")
print("-"*60)

# Identificar el mejor modelo según RMSE en test
mejor_modelo = df_resultados.loc[df_resultados["rmse_test"].idxmin()]

best_reg = mejor_modelo["regParam"]
best_elastic = mejor_modelo["elasticNetParam"]
best_tipo = mejor_modelo["tipo"]

print("\n=== MEJOR CONFIGURACIÓN ENCONTRADA ===")
print(f"Tipo de regularización: {best_tipo}")
print(f"regParam (λ):          {best_reg}")
print(f"elasticNetParam (α):   {best_elastic}")
print(f"RMSE Test:             ${mejor_modelo['rmse_test']:,.2f}")

# Entrenamiento del modelo final
print("\nEntrenando modelo final con la mejor configuración...")

lr_final = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=best_reg,
    elasticNetParam=best_elastic
)

modelo_final = lr_final.fit(train)

print("\n✓ Modelo final entrenado correctamente")
print(f"RMSE Train: ${modelo_final.summary.rootMeanSquaredError:,.2f}")
print(f"R² Train:   {modelo_final.summary.r2:.4f}")

# Evaluación final en test
predictions_final = modelo_final.transform(test)
rmse_final = evaluator.evaluate(predictions_final)

print("\n=== EVALUACIÓN FINAL (TEST) ===")
print(f"RMSE Test Final: ${rmse_final:,.2f}")

# Guardar modelo
model_path = "/opt/spark-data/processed/regularized_linear_regression_model"
modelo_final.write().overwrite().save(model_path)

print(f"\n✓ Modelo final guardado en: {model_path}")

print(
    "\nConclusión final del experimento:\n"
    "- Se seleccionó el modelo con mejor desempeño en test\n"
    "- Se redujo el overfitting respecto al modelo base\n"
    "- El modelo está listo para:\n"
    "  • scoring batch\n"
    "  • comparación con modelos no lineales\n"
    "  • despliegue productivo"
)

# ------------------------------------------------------------
# RETO BONUS: EFECTO DE LAMBDA (LASSO) EN LOS COEFICIENTES
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO BONUS: EFECTO DE LAMBDA (LASSO) EN LOS COEFICIENTES")
print("-"*60)

print(
    "\nObjetivo del experimento:\n"
    "Analizar cómo el parámetro lambda (regParam) en Lasso (L1)\n"
    "afecta la cantidad de coeficientes que se vuelven exactamente 0\n"
    "y cómo impacta el desempeño del modelo."
)

reg_values = [0.01, 0.1, 1.0, 10.0]

coef_analysis = []

print("\n=== ENTRENAMIENTO DE MODELOS LASSO ===")

for reg in reg_values:
    print(f"\nEntrenando modelo Lasso con λ = {reg}")

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

    # Contar coeficientes cercanos a cero
    zeros = np.sum(np.abs(coefs) < 1e-6)

    # Evaluar desempeño en test
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

# ------------------------------------------------------------
# Resumen tabular del efecto de lambda
# ------------------------------------------------------------

df_coef_analysis = pd.DataFrame(coef_analysis)

print("\n" + "-"*60)
print("RESUMEN DEL EFECTO DE LAMBDA (LASSO)")
print("-"*60)
print(df_coef_analysis.to_string(index=False))

# ------------------------------------------------------------
# Interpretación de resultados
# ------------------------------------------------------------

print(
    "\nInterpretación de los resultados:\n"
    "- A mayor valor de λ (regParam), mayor número de coeficientes\n"
    "  que se vuelven exactamente 0.\n"
    "- Esto confirma que Lasso realiza selección automática de features.\n"
    "- Modelos con λ alto son más simples y más interpretables,\n"
    "  pero pueden perder capacidad predictiva si se elimina\n"
    "  información relevante."
)

# ------------------------------------------------------------
# Pregunta conceptual clave
# ------------------------------------------------------------

print(
    "\nPregunta conceptual:\n"
    "¿Por qué Lasso puede llevar coeficientes exactamente a 0\n"
    "y Ridge no lo hace?"
)

print(
    "\nRespuesta:\n"
    "- Lasso (L1) utiliza una penalización basada en el valor absoluto\n"
    "  de los coeficientes.\n"
    "- Esta penalización genera esquinas en la función de optimización.\n"
    "- En dichas esquinas, el punto óptimo puede caer exactamente en 0,\n"
    "  anulando completamente una feature.\n\n"
    "- Ridge (L2) penaliza el cuadrado del coeficiente.\n"
    "- Reduce la magnitud de los coeficientes de forma continua,\n"
    "  pero rara vez los lleva exactamente a 0.\n"
    "- Por esta razón, Ridge NO realiza selección explícita de variables."
)

# ------------------------------------------------------------
# Cierre del bloque de regularización
# ------------------------------------------------------------

print("\n" + "="*60)
print("RESUMEN FINAL – REGULARIZACIÓN")
print("="*60)
print("Verifica que hayas logrado:")
print("  ✓ Entender la diferencia entre L1, L2 y ElasticNet")
print("  ✓ Experimentar con múltiples valores de lambda")
print("  ✓ Observar selección automática de variables con Lasso")
print("  ✓ Comparar simplicidad vs desempeño")
print("  ✓ Elegir un modelo alineado con el objetivo del negocio")
print("="*60)

# Detener Spark
spark.stop()
print("✓ SparkSession detenida correctamente")
