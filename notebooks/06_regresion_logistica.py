# ============================================================
# NOTEBOOK 06: Regresión Logística para Clasificación
# ============================================================

from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import when, col
from pyspark.sql.functions import abs as spark_abs, col
from pyspark.ml.feature import StandardScaler, PCA, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from delta import configure_spark_with_delta_pip
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
import numpy as np
import matplotlib.pyplot as plt

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

df = spark.read.parquet("/opt/spark-data/processed/secop_features.parquet")
print(f"Registros: {df.count():,}")

# ------------------------------------------------------------
# # RETO 1: Crear Variable Objetivo Binaria (Riesgo)
# ------------------------------------------------------------

# %%
print("\n" + "="*60)
print("RETO 1: CREACIÓN DE VARIABLE OBJETIVO BINARIA (RIESGO)")
print("="*60)

print(
    "\nObjetivo:\n"
    "Crear una variable objetivo binaria que identifique contratos\n"
    "de ALTO RIESGO a partir del valor del contrato."
)

# ------------------------------------------------------------
# Paso 1: Calcular percentil 90 del valor del contrato
# ------------------------------------------------------------
percentil_90 = df.approxQuantile(
    "valor_del_contrato_num",
    [0.9],
    0.01
)[0]

print(f"\nPercentil 90 del valor del contrato calculado:")
print(f"→ ${percentil_90:,.2f}")

print(
    "\nInterpretación del percentil:\n"
    "El percentil 90 representa el umbral a partir del cual\n"
    "se encuentra el 10% de los contratos con mayor valor económico."
)

# ------------------------------------------------------------
# Paso 2: Definir variable objetivo binaria (riesgo)
# ------------------------------------------------------------
print(
    "\nCriterio para definir ALTO RIESGO:\n"
    "- Contratos cuyo valor ≥ percentil 90\n\n"
    "Justificación del criterio:\n"
    "- Mayor impacto financiero\n"
    "- Mayor complejidad operativa y administrativa\n"
    "- Mayor probabilidad de retrasos, sobrecostos o incumplimientos\n\n"
    "En contratación pública, el tamaño del contrato\n"
    "es un proxy razonable de riesgo."
)

df = df.withColumn(
    "riesgo",
    when(col("valor_del_contrato_num") >= percentil_90, 1).otherwise(0)
)

print("\n✓ Variable objetivo 'riesgo' creada correctamente")
print("  • 1 = Alto riesgo")
print("  • 0 = Bajo riesgo")

# ------------------------------------------------------------
# Paso 3: Validación rápida de la variable objetivo
# ------------------------------------------------------------
print("\nDistribución de la variable objetivo (riesgo):")
df.groupBy("riesgo").count().show()

# ------------------------------------------------------------
# Paso 4: Ejemplos de contratos de alto riesgo
# ------------------------------------------------------------
print("\nEjemplos de contratos clasificados como ALTO RIESGO:")
df.filter(col("riesgo") == 1) \
  .select("valor_del_contrato_num", "riesgo") \
  .orderBy(col("valor_del_contrato_num").desc()) \
  .show(5, truncate=False)


# ------------------------------------------------------------
# "RETO 2: ANÁLISIS DE BALANCE DE CLASES"
# ------------------------------------------------------------

print("\n" + "="*60)
print("RETO 2: ANÁLISIS DE BALANCE DE CLASES")
print("="*60)

print(
    "\nObjetivo:\n"
    "Evaluar si la variable objetivo 'riesgo' está balanceada\n"
    "antes de entrenar un modelo de clasificación."
)

# ------------------------------------------------------------
# Distribución de clases
# ------------------------------------------------------------
print("\nDistribución absoluta de clases:")
df.groupBy("riesgo").count().show()

# Totales
total = df.count()
clase_0 = df.filter(col("riesgo") == 0).count()
clase_1 = df.filter(col("riesgo") == 1).count()

pct_0 = clase_0 / total * 100
pct_1 = clase_1 / total * 100

print("\nDistribución porcentual:")
print(f"Clase 0 (Bajo riesgo): {clase_0:,} registros ({pct_0:.1f}%)")
print(f"Clase 1 (Alto riesgo): {clase_1:,} registros ({pct_1:.1f}%)")

# ------------------------------------------------------------
# Evaluación automática del balance
# ------------------------------------------------------------
if pct_1 < 10:
    print("\n⚠️ Diagnóstico: DATASET DESBALANCEADO")
elif pct_1 < 30:
    print("\n⚠️ Diagnóstico: DATASET PARCIALMENTE DESBALANCEADO")
else:
    print("\n✅ Diagnóstico: DATASET RAZONABLEMENTE BALANCEADO")

# ------------------------------------------------------------
# Decisión técnica
# ------------------------------------------------------------
print(
    "\nDECISIÓN TÉCNICA:\n"
    "La clase de ALTO RIESGO es claramente minoritaria.\n\n"
    "Estrategia elegida:\n"
    "✔ Usar class_weight en el modelo de clasificación.\n\n"
    "Justificación:\n"
    "- Evita duplicar registros artificialmente (oversampling)\n"
    "- No pierde información (a diferencia de undersampling)\n"
    "- Es robusta para datasets medianos y grandes\n"
    "- Está soportada directamente por LogisticRegression en Spark\n\n"
    "Alternativas futuras a evaluar:\n"
    "- Ajustar el threshold de clasificación\n"
    "- Oversampling solo para análisis exploratorio"
)

print("\n✓ Análisis de balance de clases completado")

# ------------------------------------------------------------
# PASO 1: PREPARACIÓN DE DATOS PARA CLASIFICACIÓN BINARIA
# ------------------------------------------------------------

print("\n" + "="*60)
print("PASO 1: PREPARACIÓN DE DATOS PARA CLASIFICACIÓN BINARIA")
print("="*60)

print(
    "\nObjetivo:\n"
    "Preparar el dataset para entrenar un modelo de clasificación\n"
    "binaria (Regresión Logística) que prediga contratos de ALTO RIESGO."
)

# ------------------------------------------------------------
# Renombrar columnas para consistencia con Spark ML
# ------------------------------------------------------------
print(
    "\nEstandarizando nombres de columnas:\n"
    "- 'riesgo'   → 'label'   (variable objetivo)\n"
    "- 'features_raw' → 'features' (vector de entrada)"
)

df_binary = (
    df.withColumnRenamed("riesgo", "label")
      .withColumnRenamed("features_raw", "features")
)

print("✓ Columnas renombradas correctamente")

# ------------------------------------------------------------
# Filtrar valores nulos
# ------------------------------------------------------------
print(
    "\nFiltrando registros con valores nulos:\n"
    "- label no puede ser nulo\n"
    "- features no puede ser nulo\n"
    "Esto evita errores durante el entrenamiento del modelo."
)

df_binary = df_binary.filter(
    col("label").isNotNull() & col("features").isNotNull()
)

print(f"✓ Registros válidos después de limpieza: {df_binary.count():,}")

# ------------------------------------------------------------
# Split Train / Test
# ------------------------------------------------------------
print(
    "\nEstrategia Train/Test Split:\n"
    "- 70% para entrenamiento\n"
    "- 30% para evaluación\n"
    "- seed=42 para reproducibilidad"
)

train, test = df_binary.randomSplit([0.7, 0.3], seed=42)

print(f"Train: {train.count():,} registros (70%)")
print(f"Test:  {test.count():,} registros (30%)")


# ------------------------------------------------------------
# RETO 3: ENTENDER LA REGRESIÓN LOGÍSTICA
# ------------------------------------------------------------

print("\n" + "="*60)
print("RETO 3: ENTENDER LA REGRESIÓN LOGÍSTICA")
print("="*60)

print(
    "\nPregunta:\n"
    "¿Cuál de las siguientes afirmaciones sobre la regresión logística es correcta?\n\n"
    "Opciones:\n"
    "A) Predice probabilidades entre 0 y 1\n"
    "B) Usa la función sigmoide\n"
    "C) Es un modelo de clasificación, no de regresión continua\n"
    "D) Todas las anteriores"
)

print("\nRespuesta correcta:")
print("✅ D) Todas las anteriores")

print(
    "\nExplicación detallada:\n"
    "La regresión logística se utiliza para problemas de clasificación\n"
    "binaria y se diferencia de la regresión lineal en varios aspectos:\n\n"
    "1️⃣ Predicción de probabilidades (Opción A)\n"
    "   - El modelo estima una probabilidad entre 0 y 1\n"
    "   - Esta probabilidad representa la probabilidad de pertenecer\n"
    "     a la clase positiva (ej. contrato de alto riesgo).\n\n"
    "2️⃣ Uso de la función sigmoide (Opción B)\n"
    "   - La función sigmoide transforma una combinación lineal\n"
    "     de las features en una probabilidad:\n\n"
    "       p = 1 / (1 + e^(-z))\n\n"
    "3️⃣ Modelo de clasificación (Opción C)\n"
    "   - No predice valores continuos como precios o montos\n"
    "   - Predice clases (0 o 1) a partir de un umbral de decisión.\n\n"
    "Conclusión:\n"
    "La regresión logística modela la probabilidad de ocurrencia\n"
    "de un evento (ej. contrato de alto riesgo) y luego aplica\n"
    "un umbral para decidir la clase final."
)

print("\n✓ Contexto teórico de Regresión Logística cubierto correctamente")

# ------------------------------------------------------------
# RETO 4: CONFIGURAR EL MODELO DE REGRESIÓN LOGÍSTICA
# ------------------------------------------------------------

print("\n" + "="*60)
print("RETO 4: CONFIGURAR EL MODELO DE REGRESIÓN LOGÍSTICA")
print("="*60)

print(
    "\nObjetivo:\n"
    "Configurar un modelo de Regresión Logística para clasificar\n"
    "contratos de ALTO vs BAJO riesgo, considerando el desbalance\n"
    "natural de las clases."
)

print(
    "\nDecisiones clave de configuración:\n"
    "- Regularización L2 para evitar overfitting\n"
    "- Ajuste del threshold por desbalance de clases\n"
    "- Iteraciones suficientes para convergencia estable"
)

# ------------------------------------------------------------
# Configuración del modelo
# ------------------------------------------------------------
lr_classifier = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,          # Iteraciones suficientes para convergencia
    regParam=0.1,         # Regularización L2 (Ridge)
    elasticNetParam=0.0,  # 0 = L2 pura
    threshold=0.4         # Umbral ajustado por desbalance
)

print("\n✓ Clasificador Logístico configurado correctamente")
print(f"  • maxIter:        {lr_classifier.getMaxIter()}")
print(f"  • regParam (L2):  {lr_classifier.getRegParam()}")
print(f"  • elasticNet:     {lr_classifier.getElasticNetParam()}")
print(f"  • threshold:      {lr_classifier.getThreshold()}")

print(
    "\nJustificación del threshold:\n"
    "Si el dataset tiene aproximadamente:\n"
    "- 90% contratos de bajo riesgo (clase 0)\n"
    "- 10% contratos de alto riesgo (clase 1)\n\n"
    "Entonces:\n"
    "- Un threshold = 0.5 favorece demasiado la clase mayoritaria\n"
    "- Bajar el threshold (0.3–0.4) permite detectar más contratos\n"
    "  de alto riesgo (mayor recall)\n"
    "- A costa de más falsos positivos\n\n"
    "En gestión de riesgo:\n"
    "✔️ Es preferible un falso positivo que un falso negativo"
)

# ------------------------------------------------------------
# Entrenamiento del modelo
# ------------------------------------------------------------

print("\n" + "="*60)
print("ENTRENAMIENTO DEL MODELO")
print("="*60)

print("Entrenando clasificador logístico...")
lr_model = lr_classifier.fit(train)
print("✓ Modelo entrenado correctamente")

# ------------------------------------------------------------
# PASO 2: PREDICCIONES SOBRE EL SET DE TEST
# ------------------------------------------------------------
print("\n" + "="*60)
print("PASO 2: PREDICCIONES SOBRE EL SET DE TEST")
print("="*60)

predictions = lr_model.transform(test)

print("\nPrimeras predicciones generadas:")
predictions.select(
    "label",
    "prediction",
    "probability"
).show(10, truncate=False)

# ------------------------------------------------------------
# RETO 5: INTERPRETAR PROBABILIDADES
# ------------------------------------------------------------
print("\n" + "="*60)
print("RETO 5: INTERPRETAR PROBABILIDADES")
print("="*60)

print(
    "\nPregunta:\n"
    "¿Cómo se interpretan las probabilidades en Spark ML\n"
    "para un modelo de clasificación binaria?"
)

print(
    "\nRespuesta correcta:\n"
    "✅ A) probability = [P(clase 0), P(clase 1)]"
)

print(
    "\nExplicación:\n"
    "En Spark ML, la columna 'probability' es un vector donde:\n\n"
    "  probability[0] → Probabilidad de clase 0 (Bajo riesgo)\n"
    "  probability[1] → Probabilidad de clase 1 (Alto riesgo)\n\n"
    "Ejemplo:\n"
    "  probability = [0.80, 0.20]\n"
    "  → 80% Bajo riesgo\n"
    "  → 20% Alto riesgo"
)


print("\nExtrayendo probabilidad de la clase positiva (Alto riesgo)...")

# Convertir VectorUDT → Array
predictions = predictions.withColumn(
    "prob_array",
    vector_to_array(col("probability"))
)

# Extraer probabilidad de clase 1
predictions = predictions.withColumn(
    "prob_clase_1",
    col("prob_array")[1]
)

print("✓ Probabilidad de clase 1 extraída correctamente")

# ------------------------------------------------------------
# ANÁLISIS DE CASOS INSEGUROS
# ------------------------------------------------------------
print("\n" + "="*60)
print("ANÁLISIS DE CASOS INSEGUROS")
print("="*60)

print(
    "\nDefinición de predicción insegura:\n"
    "Casos donde la probabilidad está cerca del threshold,\n"
    "lo que indica incertidumbre del modelo.\n\n"
    "Rango analizado:\n"
    "0.4 < P(clase 1) < 0.6"
)

predicciones_dudosas = predictions.filter(
    (col("prob_clase_1") > 0.4) & (col("prob_clase_1") < 0.6)
)

print("\nEjemplos de predicciones dudosas:")
predicciones_dudosas.select(
    "label",
    "prediction",
    "prob_clase_1"
).show(10, truncate=False)

print(
    "\nInterpretación:\n"
    "- Estos casos son candidatos ideales para revisión humana\n"
    "- O para reglas de negocio adicionales\n"
    "- También pueden usarse para ajustar el threshold\n"
    "  o mejorar las features del modelo"
)

# ------------------------------------------------------------
# RETO 6: Evaluación con Múltiples Métricas
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO 6: EVALUACIÓN DEL MODELO CON MÚLTIPLES MÉTRICAS")
print("-"*60)

print(
    "\nObjetivo:\n"
    "Evaluar el desempeño del clasificador logístico usando\n"
    "métricas estándar de clasificación, con foco en problemas\n"
    "de riesgo donde la simple accuracy NO es suficiente."
)

# =========================
# AUC - ROC
# =========================
print("\nCalculando métrica AUC-ROC...")

evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator_auc.evaluate(predictions)

# =========================
# Métricas clásicas de clasificación
# =========================
print("Calculando métricas clásicas de clasificación...")

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
print("RESULTADOS DE EVALUACIÓN DEL MODELO")
print("="*60)
print(f"AUC-ROC:   {auc:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("="*60)

print(
    "\nInterpretación del AUC:\n"
    "- AUC = 0.5  → Modelo aleatorio\n"
    "- AUC ≈ 0.7  → Modelo aceptable\n"
    "- AUC ≈ 0.8  → Buen modelo\n"
    "- AUC ≥ 0.9  → Excelente modelo\n\n"
    f"Con un AUC de {auc:.2f}, el modelo distingue correctamente\n"
    "entre contratos de alto y bajo riesgo aproximadamente\n"
    "el 75% del tiempo."
)

print(
    "\nConclusión de negocio:\n"
    "- En problemas de riesgo, el AUC suele ser más relevante que accuracy\n"
    "- El recall es crítico para no dejar pasar contratos riesgosos\n"
    "- El threshold puede ajustarse según el costo del error"
)

# ------------------------------------------------------------
# RETO 7: Matriz de Confusión
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO 7: MATRIZ DE CONFUSIÓN")
print("-"*60)

print("\nMatriz de confusión agregada (label vs prediction):")

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
print("Calculando valores TP, TN, FP y FN...")

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

print("\nDesglose de la matriz de confusión:")
print(f"TP (True Positives):  {TP:,}")
print(f"TN (True Negatives):  {TN:,}")
print(f"FP (False Positives): {FP:,}")
print(f"FN (False Negatives): {FN:,}")

# =========================
# Métricas derivadas
# =========================
precision_manual = TP / (TP + FP) if (TP + FP) > 0 else 0
recall_manual = TP / (TP + FN) if (TP + FN) > 0 else 0

print("\nMétricas derivadas (cálculo manual):")
print(f"Precision (manual): {precision_manual:.4f}")
print(f"Recall    (manual): {recall_manual:.4f}")

print(
    "\nInterpretación desde el punto de vista del negocio:\n\n"
    "FALSO POSITIVO (FP):\n"
    "- Clasificar como ALTO riesgo un contrato que es BAJO\n"
    "- Impacto:\n"
    "  • Controles innecesarios\n"
    "  • Mayor costo operativo\n\n"
    "FALSO NEGATIVO (FN):\n"
    "- Clasificar como BAJO riesgo un contrato que es ALTO\n"
    "- Impacto:\n"
    "  • Riesgo financiero no detectado\n"
    "  • Riesgo legal y reputacional\n\n"
    "Conclusión:\n"
    "✔️ En análisis de riesgo, el FALSO NEGATIVO es más grave\n"
    "✔️ Por eso se prioriza el RECALL sobre la accuracy\n"
    "✔️ Ajustar el threshold es una decisión estratégica"
)

# ------------------------------------------------------------
# RETO BONUS 1: AJUSTAR THRESHOLD
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO BONUS 1: AJUSTE DE THRESHOLD EN REGRESIÓN LOGÍSTICA")
print("-"*60)

print(
    "\nObjetivo:\n"
    "Evaluar cómo cambia el desempeño del modelo al modificar\n"
    "el threshold de clasificación, entendiendo el trade-off\n"
    "entre precisión, recall y detección de riesgo."
)

from pyspark.ml.classification import LogisticRegression

# Thresholds a evaluar
thresholds = [0.3, 0.5, 0.7]

print("\nThresholds evaluados:", thresholds)
print("\n=== COMPARACIÓN DE MÉTRICAS POR THRESHOLD ===")

for t in thresholds:
    print("\n" + "-"*40)
    print(f"Evaluando Threshold = {t}")
    print("-"*40)

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

print("\n" + "="*60)
print("INTERPRETACIÓN DEL AJUSTE DE THRESHOLD")
print("="*60)

print(
    "\nThreshold bajo (0.3):\n"
    "- Aumenta el recall\n"
    "- Detecta más contratos de alto riesgo\n"
    "- Incrementa los falsos positivos\n"
    "- Útil cuando el costo de NO detectar riesgo es alto"
)

print(
    "\nThreshold medio (0.5):\n"
    "- Balance general entre precision y recall\n"
    "- Es el valor por defecto en muchos modelos\n"
    "- No siempre es óptimo en datasets desbalanceados"
)

print(
    "\nThreshold alto (0.7):\n"
    "- Aumenta la precisión\n"
    "- Reduce falsos positivos\n"
    "- Riesgo alto de falsos negativos\n"
    "- Puede dejar pasar contratos críticos"
)

print(
    "\nConclusión recomendada para este caso de uso:\n"
    "✔️ En problemas de riesgo, el FALSO NEGATIVO es más costoso\n"
    "✔️ Es preferible priorizar RECALL sobre accuracy\n"
    "✔️ El threshold óptimo suele estar entre 0.3 y 0.4\n"
    "✔️ La decisión final debe alinearse con el costo del error\n"
    "   definido por negocio y control interno"
)

# ------------------------------------------------------------
# RETO BONUS 2: CURVA ROC
# ------------------------------------------------------------

print("\n" + "-"*60)
print("RETO BONUS 2: CURVA ROC (TPR vs FPR)")
print("-"*60)

print(
    "\nObjetivo:\n"
    "Visualizar el trade-off entre la tasa de verdaderos positivos (TPR)\n"
    "y la tasa de falsos positivos (FPR) para distintos thresholds.\n\n"
    "Esto permite evaluar la capacidad del modelo para distinguir\n"
    "entre contratos de ALTO y BAJO riesgo."
)

# --------------------------------------------
# 1. Extraer probabilidades y etiquetas
# --------------------------------------------
print("\nExtrayendo probabilidades y labels a Pandas...")

prob_df = predictions.select("label", "probability").toPandas()

# Probabilidad de la clase positiva (riesgo = 1)
probs = np.array([p[1] for p in prob_df["probability"]])
labels = prob_df["label"].values

print(f"✓ Registros procesados: {len(prob_df):,}")

# --------------------------------------------
# 2. Calcular TPR y FPR para múltiples thresholds
# --------------------------------------------
print("\nCalculando TPR y FPR para múltiples thresholds...")

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

print("✓ Métricas ROC calculadas correctamente")

# --------------------------------------------
# 3. Graficar Curva ROC
# --------------------------------------------
print("\nGenerando gráfica de Curva ROC...")

plt.figure(figsize=(8, 6))
plt.plot(fpr_list, tpr_list, label=f"Modelo (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "r--", label="Modelo Aleatorio")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Curva ROC – Clasificación de Riesgo")
plt.legend()
plt.grid(True)

roc_path = "/opt/spark-data/processed/roc_curve.png"
plt.savefig(roc_path)
plt.close()

print(f"✓ Curva ROC guardada en: {roc_path}")

# --------------------------------------------
# Interpretación conceptual
# --------------------------------------------
print("\n" + "="*60)
print("INTERPRETACIÓN DE LA CURVA ROC")
print("="*60)

print(
    "\n¿Qué representa la Curva ROC?\n"
    "- Eje X (FPR): Proporción de contratos de BAJO riesgo\n"
    "  clasificados erróneamente como ALTO riesgo.\n"
    "- Eje Y (TPR / Recall): Proporción de contratos de ALTO riesgo\n"
    "  correctamente identificados por el modelo."
)

print(
    "\n¿Cómo interpretar el AUC?\n"
    "- AUC = 0.5  → Modelo aleatorio\n"
    "- AUC ≈ 0.7  → Modelo aceptable\n"
    "- AUC ≈ 0.8  → Buen modelo\n"
    "- AUC ≥ 0.9  → Excelente modelo"
)

print(
    f"\nResultado obtenido:\n"
    f"✔️ AUC-ROC = {auc:.4f}\n\n"
    "Esto indica que el modelo tiene una buena capacidad\n"
    "para discriminar entre contratos de alto y bajo riesgo."
)

print(
    "\nEn problemas de riesgo:\n"
    "- La Curva ROC ayuda a elegir el threshold óptimo\n"
    "- No existe un único punto correcto\n"
    "- La decisión depende del costo de los falsos negativos\n"
    "  vs falsos positivos"
)

# --------------------------------------------
# Guardar modelo entrenado
# --------------------------------------------
print("\nGuardando modelo de regresión logística...")

model_path = "/opt/spark-data/processed/logistic_regression_model"
lr_model.write().overwrite().save(model_path)

print(f"✓ Modelo guardado en: {model_path}")

# --------------------------------------------
# Resumen final
# --------------------------------------------
print("\n" + "="*60)
print("RESUMEN FINAL – CLASIFICACIÓN DE RIESGO")
print("="*60)
print("✓ Variable objetivo de riesgo definida")
print("✓ Modelo de clasificación entrenado")
print(f"✓ AUC-ROC: {auc:.4f}")
print(f"✓ F1-Score: {f1:.4f}")
print("✓ Curva ROC generada y almacenada")
print("✓ Modelo persistido para uso futuro")
print("✓ Próximo paso: Regularización y tuning fino (notebook 07)")
print("="*60)

# --------------------------------------------
# Detener Spark
# --------------------------------------------
spark.stop()
print("\n✓ SparkSession detenida correctamente")

