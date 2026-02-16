# %% [markdown]
# # Notebook 06: Regresión Logística para Clasificación
#
# **Sección 14**: Clasificación Binaria
#
# **Objetivo**: Clasificar contratos según riesgo de incumplimiento
#
# ## RETO PRINCIPAL: Crear tu propia variable objetivo
#
# **Problema**: El dataset no tiene una columna de "riesgo de incumplimiento".
# ¡TENDRÁS QUE CREARLA!
#
# **Instrucciones**:
# Define un criterio para clasificar contratos como "alto riesgo" (1) o "bajo riesgo" (0)
#
# **Posibles criterios**:
# - Contratos con valor > percentil 90
# - Contratos con duración > 365 días
# - Contratos de ciertos departamentos
# - Combinación de múltiples factores
#
# **TU DECISIÓN**: ¿Qué define un contrato de alto riesgo?

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
df = spark.read.parquet("/opt/spark-data/processed/secop_features.parquet")
print(f"Registros: {df.count():,}")

# %% 
# RETO 1: Crear Variable Objetivo Binaria (Riesgo)


# 1. Calcular percentil 90 del valor del contrato
percentil_90 = df.approxQuantile(
    "valor_del_contrato_num",
    [0.9],
    0.01
)[0]

print(f"Percentil 90 del valor del contrato: ${percentil_90:,.2f}")

# 2. Definir variable objetivo binaria: riesgo
# Criterio:
# - Contratos cuyo valor está en el 10% superior (percentil 90)
# - Se consideran de ALTO RIESGO por:
#   • Mayor impacto financiero
#   • Mayor complejidad contractual
#   • Mayor probabilidad de retrasos o incumplimientos

df = df.withColumn(
    "riesgo",
    when(
        col("valor_del_contrato_num") >= percentil_90,
        1  # Alto riesgo
    ).otherwise(
        0  # Bajo riesgo
    )
)

# 3. Validación rápida
print("\nDistribución de la variable objetivo (riesgo):")
df.groupBy("riesgo").count().show()

# 4. Ver ejemplos
print("\nEjemplos de contratos clasificados como alto riesgo:")
df.filter(col("riesgo") == 1) \
  .select("valor_del_contrato_num", "riesgo") \
  .orderBy(col("valor_del_contrato_num").desc()) \
  .show(5, truncate=False)

# Criterio elegido:
#Valor del contrato ≥ percentil 90

#Razón:
#Los contratos más grandes concentran mayor riesgo financiero
#Tienen mayor complejidad operativa y administrativa
#En análisis de contratación pública, el tamaño del contrato es un proxy razonable de riesgo


# %%
# RETO 2: Balance de Clases

from pyspark.sql.functions import col

print("\n=== DISTRIBUCIÓN DE CLASES ===")

# Distribución absoluta
class_distribution = df.groupBy("riesgo").count()
class_distribution.show()

# Totales
total = df.count()
clase_0 = df.filter(col("riesgo") == 0).count()
clase_1 = df.filter(col("riesgo") == 1).count()

# Porcentajes
pct_0 = clase_0 / total * 100
pct_1 = clase_1 / total * 100

print(f"Clase 0 (Bajo riesgo): {clase_0:,} registros ({pct_0:.1f}%)")
print(f"Clase 1 (Alto riesgo): {clase_1:,} registros ({pct_1:.1f}%)")

# Evaluación automática de balance
if pct_1 < 10:
    print("\n⚠️ Dataset DESBALANCEADO")
elif pct_1 < 30:
    print("\n⚠️ Dataset PARCIALMENTE DESBALANCEADO")
else:
    print("\n✅ Dataset razonablemente balanceado")

# -------------------------------
# DECISIÓN TÉCNICA (COMENTARIOS)
# -------------------------------

# ¿Está balanceado?
# NO, la clase de alto riesgo es claramente minoritaria.

# ¿Qué haría?
# Opción elegida: C) Usar class_weight en el modelo

# Justificación:
# - Evita duplicar registros artificialmente (oversampling)
# - No pierde información (a diferencia de undersampling)
# - Es la estrategia más robusta para datasets medianos/grandes
# - Está soportada directamente por LogisticRegression en Spark
#
# Alternativas futuras:
# - Ajustar threshold de clasificación
# - Probar oversampling solo para experimentación

# %% [markdown]
# ## PASO 1: Preparar Datos

# %%
# Renombrar columnas para el modelo
df_binary = df.withColumnRenamed("riesgo", "label") \
               .withColumnRenamed("features_raw", "features")

# Filtrar nulos
df_binary = df_binary.filter(col("label").isNotNull() & col("features").isNotNull())

# Split train/test
train, test = df_binary.randomSplit([0.7, 0.3], seed=42)

print(f"Train: {train.count():,} registros")
print(f"Test:  {test.count():,} registros")

# %%
# RETO 3: Entender la Regresión Logística

#**Opciones**:
# - A) Predice probabilidades entre 0 y 1
# - B) Usa función sigmoid
# - C) Es para clasificación, no para valores continuos
# - D) Todas las anteriore
# Respuesta correcta:
# ✅ D) Todas las anteriores

# Explicación:
#
# La regresión logística se diferencia de la regresión lineal porque:
#
# - Predice probabilidades entre 0 y 1 (opción A),
#   que representan la probabilidad de pertenecer a la clase positiva.
#
# - Utiliza la función sigmoide (opción B) para transformar
#   una combinación lineal de las features en una probabilidad:
#
#       p = 1 / (1 + e^(-z))
#
# - Está diseñada para problemas de clasificación (opción C),
#   no para predecir valores continuos como precios o montos.
#
# En resumen:
# La regresión logística modela la probabilidad de ocurrencia
# de un evento (ej. contrato de alto riesgo) y luego aplica
# un umbral para decidir la clase final (0 o 1).

# %%
# RETO 4: Configurar el Modelo de Regresión Logística

from pyspark.ml.classification import LogisticRegression

# Configuración del modelo
lr_classifier = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,        # Iteraciones suficientes para convergencia
    regParam=0.1,       # Regularización L2 para prevenir overfitting
    elasticNetParam=0.0,# L2 (ridge)
    threshold=0.4       # Umbral ajustado por posible desbalance de clases
)

print("✓ Clasificador configurado")
print(f"  • maxIter: {lr_classifier.getMaxIter()}")
print(f"  • regParam: {lr_classifier.getRegParam()}")
print(f"  • threshold: {lr_classifier.getThreshold()}")

# %%
# Respuesta conceptual:
#
# Si tienes 90% clase 0 y 10% clase 1:
# - Usar threshold=0.5 suele favorecer demasiado la clase mayoritaria
# - Es recomendable bajar el threshold (ej. 0.3–0.4)
# - Esto aumenta recall de la clase minoritaria (alto riesgo)
# - A costa de más falsos positivos (trade-off aceptable en riesgo)

# %%
# Entrenar modelo
print("\nEntrenando clasificador logístico...")
lr_model = lr_classifier.fit(train)
print("✓ Modelo entrenado correctamente")

# %%
# PASO 2: Predicciones sobre el set de test

predictions = lr_model.transform(test)

print("\n=== PRIMERAS PREDICCIONES ===")
predictions.select(
    "label",
    "prediction",
    "probability"
).show(10, truncate=False)


# %%
# RETO 5: Interpretar Probabilidades (SOLUCIÓN DEFINITIVA)

from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array

# Explicación conceptual:
#
# En Spark ML:
# probability = [p(clase 0), p(clase 1)]
#
# Ejemplo:
# probability = [0.8, 0.2]
# → 80% probabilidad de clase 0 (bajo riesgo)
# → 20% probabilidad de clase 1 (alto riesgo)
#
# ✅ Respuesta correcta: A)

# %%
# Convertir VectorUDT → Array
predictions = predictions.withColumn(
    "prob_array",
    vector_to_array(col("probability"))
)

# Extraer probabilidad de la clase positiva (índice 1)
predictions = predictions.withColumn(
    "prob_clase_1",
    col("prob_array")[1]
)

print("✓ Probabilidad de clase 1 extraída correctamente")

# %%
# Analizar casos "inseguros" (probabilidades cercanas al threshold)
print("\n=== CASOS CON PREDICCIÓN INSEGURA (0.4 < P(clase 1) < 0.6) ===")

predicciones_dudosas = predictions.filter(
    (col("prob_clase_1") > 0.4) & (col("prob_clase_1") < 0.6)
)

predicciones_dudosas.select(
    "label",
    "prediction",
    "prob_clase_1"
).show(10, truncate=False)



# %%
# RETO 6: Evaluación con Múltiples Métricas

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

# =========================
# AUC - ROC
# =========================
evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator_auc.evaluate(predictions)

# =========================
# Métricas clásicas de clasificación
# =========================
evaluator_multi = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction"
)

accuracy = evaluator_multi.evaluate(
    predictions,
    {evaluator_multi.metricName: "accuracy"}
)

precision = evaluator_multi.evaluate(
    predictions,
    {evaluator_multi.metricName: "weightedPrecision"}
)

recall = evaluator_multi.evaluate(
    predictions,
    {evaluator_multi.metricName: "weightedRecall"}
)

f1 = evaluator_multi.evaluate(
    predictions,
    {evaluator_multi.metricName: "f1"}
)

# =========================
# Resultados
# =========================
print("\n" + "="*60)
print("MÉTRICAS DE CLASIFICACIÓN")
print("="*60)
print(f"AUC-ROC:   {auc:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("="*60)

# =========================
# Interpretación (comentarios)
# =========================
#
# ¿Es bueno un AUC de 0.75?
#
# ✔️ Sí, es un modelo razonablemente bueno.
#
# Interpretación:
# - AUC = 0.5  → modelo aleatorio
# - AUC ≈ 0.7  → modelo aceptable
# - AUC ≈ 0.8  → buen modelo
# - AUC ≥ 0.9  → excelente modelo
#
# Un AUC de 0.75 indica que el modelo distingue correctamente
# entre clases positivas y negativas el 75% del tiempo.
#
# En problemas de riesgo / fraude / incumplimiento:
# - AUC suele ser más importante que accuracy
# - Recall suele priorizarse sobre precision
# - El threshold puede ajustarse según el costo del error


# %%
# RETO 7: Matriz de Confusión
from pyspark.sql.functions import col

print("\n=== MATRIZ DE CONFUSIÓN (label vs prediction) ===")

# Matriz de confusión agregada
confusion_matrix = (
    predictions
    .groupBy("label", "prediction")
    .count()
    .orderBy("label", "prediction")
)

confusion_matrix.show()

# =========================
# Cálculo manual de métricas
# =========================
TP = predictions.filter(
    (col("label") == 1) & (col("prediction") == 1)
).count()

TN = predictions.filter(
    (col("label") == 0) & (col("prediction") == 0)
).count()

FP = predictions.filter(
    (col("label") == 0) & (col("prediction") == 1)
).count()

FN = predictions.filter(
    (col("label") == 1) & (col("prediction") == 0)
).count()

print("\n=== DESGLOSE DE LA MATRIZ DE CONFUSIÓN ===")
print(f"TP (True Positives):  {TP:,}")
print(f"TN (True Negatives):  {TN:,}")
print(f"FP (False Positives): {FP:,}")
print(f"FN (False Negatives): {FN:,}")

# =========================
# Métricas derivadas (útiles para negocio)
# =========================
precision_manual = TP / (TP + FP) if (TP + FP) > 0 else 0
recall_manual = TP / (TP + FN) if (TP + FN) > 0 else 0

print("\n=== MÉTRICAS DERIVADAS ===")
print(f"Precision (manual): {precision_manual:.4f}")
print(f"Recall    (manual): {recall_manual:.4f}")

# =========================
# Interpretación del problema (comentarios)
# =========================
#
# ¿Qué es peor en ESTE problema?
#
# FALSO POSITIVO (FP):
# - Predecir ALTO riesgo cuando en realidad es BAJO
# - Consecuencia:
#   • Posible rechazo innecesario
#   • Mayor control / burocracia
#   • Costo operativo
#
# FALSO NEGATIVO (FN):
# - Predecir BAJO riesgo cuando en realidad es ALTO
# - Consecuencia:
#   • Contrato riesgoso no detectado
#   • Pérdidas económicas
#   • Riesgo legal / reputacional
#
# 👉 En problemas de riesgo / incumplimiento:
# ✔️ Normalmente el FALSO NEGATIVO es MÁS GRAVE
# ✔️ Por eso se prioriza RECALL sobre accuracy
# ✔️ Se puede bajar el threshold para detectar más casos de riesgo

# %%
# RETO BONUS 1: Ajustar Threshold
# Objetivo: Evaluar el impacto del threshold en métricas clave

from pyspark.ml.classification import LogisticRegression

thresholds = [0.3, 0.5, 0.7]

print("\n=== COMPARACIÓN DE THRESHOLDS ===")

for t in thresholds:
    print(f"\n--- Threshold = {t} ---")

    # Configurar modelo con threshold específico
    lr_temp = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        regParam=0.0,
        threshold=t
    )

    # Entrenar modelo
    model_temp = lr_temp.fit(train)

    # Predicciones
    preds_temp = model_temp.transform(test)

    # Métricas
    acc_temp = evaluator_multi.evaluate(
        preds_temp,
        {evaluator_multi.metricName: "accuracy"}
    )

    prec_temp = evaluator_multi.evaluate(
        preds_temp,
        {evaluator_multi.metricName: "weightedPrecision"}
    )

    rec_temp = evaluator_multi.evaluate(
        preds_temp,
        {evaluator_multi.metricName: "weightedRecall"}
    )

    f1_temp = evaluator_multi.evaluate(
        preds_temp,
        {evaluator_multi.metricName: "f1"}
    )

    auc_temp = evaluator_auc.evaluate(preds_temp)

    # Resultados
    print(f"Accuracy : {acc_temp:.4f}")
    print(f"Precision: {prec_temp:.4f}")
    print(f"Recall   : {rec_temp:.4f}")
    print(f"F1-Score : {f1_temp:.4f}")
    print(f"AUC-ROC  : {auc_temp:.4f}")

# =========================
# Reflexión (completar en markdown o comentario)
# =========================
#
# ¿Qué threshold elegirías?
#
# - Threshold bajo (0.3):
#   • Mayor recall
#   • Detecta más casos de alto riesgo
#   • Más falsos positivos
#
# - Threshold medio (0.5):
#   • Balance general
#   • Default en la mayoría de modelos
#
# - Threshold alto (0.7):
#   • Mayor precisión
#   • Menos falsos positivos
#   • Riesgo de perder casos críticos
#
# Elección recomendada para riesgo:
# 👉 Threshold que MAXIMICE recall y controle FP aceptables


# %%
# RETO BONUS 2: Curva ROC
# Objetivo: Visualizar el trade-off entre TPR y FPR

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. Extraer probabilidades y labels a Pandas
# --------------------------------------------
prob_df = predictions.select("label", "probability").toPandas()

# Probabilidad de la clase positiva (1)
probs = np.array([p[1] for p in prob_df["probability"]])
labels = prob_df["label"].values

# --------------------------------------------
# 2. Calcular TPR y FPR para múltiples thresholds
# --------------------------------------------
thresholds_roc = np.linspace(0, 1, 100)
tpr_list = []
fpr_list = []

for t in thresholds_roc:
    y_pred = (probs >= t).astype(int)

    tp = np.sum((y_pred == 1) & (labels == 1))
    fp = np.sum((y_pred == 1) & (labels == 0))
    tn = np.sum((y_pred == 0) & (labels == 0))
    fn = np.sum((y_pred == 0) & (labels == 1))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    tpr_list.append(tpr)
    fpr_list.append(fpr)

# --------------------------------------------
# 3. Graficar Curva ROC
# --------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(fpr_list, tpr_list, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "r--", label="Random")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Curva ROC")
plt.legend()
plt.grid(True)

# Guardar gráfico
roc_path = "/opt/spark-data/processed/roc_curve.png"
plt.savefig(roc_path)
plt.close()

print(f"✓ Curva ROC guardada en: {roc_path}")


# %%
# Guardar modelo
model_path = "/opt/spark-data/processed/logistic_regression_model"
lr_model.write().overwrite().save(model_path)
print(f"\n✓ Modelo guardado en: {model_path}")

# %%
print("\n" + "="*60)
print("RESUMEN CLASIFICACIÓN")
print("="*60)
print(f"✓ Criterio de riesgo definido")
print(f"✓ Modelo entrenado")
print(f"✓ AUC-ROC: {auc:.4f}")
print(f"✓ F1-Score: {f1:.4f}")
print(f"✓ Próximo paso: Regularización (notebook 07)")
print("="*60)

# %%
spark.stop()
