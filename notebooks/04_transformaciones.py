# ============================================================
# NOTEBOOK 04: Transformaciones Avanzadas
# ============================================================

from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, PCA, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import col
from delta import configure_spark_with_delta_pip
import pandas as pd
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
print("\n[2] Cargando dataset transformado (Notebook 03)...")

df = spark.read.parquet("/opt/spark-data/processed/secop_features.parquet")

print(f"✓ Registros cargados: {df.count():,}")
print(f"✓ Columnas disponibles: {len(df.columns)}")


# ------------------------------------------------------------
# RETO 1: ¿Por qué normalizar?
# ------------------------------------------------------------

print("\n[3] RETO 1 – ¿Por qué normalizar las features?")

print("""
INSPECCIÓN INICIAL:
Analizaremos las magnitudes de las features para verificar
si existen escalas muy diferentes.
""")

sample = df.select("features_raw").limit(5).collect()

#2
for i, row in enumerate(sample, start=1):
    features_array = row["features_raw"].toArray()
    print(f"Registro {i} - primeros 10 valores:")
    print(features_array[:10])

print("""
OBSERVACIÓN:
- Se identifican valores binarios (0/1) provenientes de OneHotEncoding
- También valores numéricos con magnitudes mucho mayores

CONCLUSIÓN:
Muchos algoritmos (Regresión, KMeans, PCA, SVM) son sensibles a la escala.
Sin normalización, las variables grandes dominan el aprendizaje.
""")

# ------------------------------------------------------------
# PASO 1: StandardScaler
# ------------------------------------------------------------
print("\n[4] PASO 1 – StandardScaler")

print("""
CONCEPTO:
StandardScaler transforma cada feature para que tenga:
- Media ≈ 0
- Desviación estándar ≈ 1

Fórmula:
z = (x - μ) / σ

NOTA:
withMean=False es obligatorio para vectores sparse.
""")


# Crear StandardScaler
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features_scaled",
    withMean=False,  # Requerido para vectores sparse
    withStd=True     # Normaliza por desviación estándar
)

print("✓ StandardScaler creado correctamente")

scaler_model = scaler.fit(df)
df_scaled = scaler_model.transform(df)

print("✓ Features escaladas correctamente")

# %%
# Verificación rápida
print("Columnas finales del DataFrame:")
for c in df_scaled.columns:
    print(f"  - {c}")

# %%
# Inspeccionar estructura del vector escalado
print("\nEsquema de la columna features_scaled:")
df_scaled.select("features_scaled").printSchema()


# ------------------------------------------------------------
# RETO 2: Verificación de escalado
# ------------------------------------------------------------
# %% [markdown]
# ## RETO 2: Comparar antes y después de escalar
#
# **Objetivo**: Verificar que el escalado funcionó correctamente
#
# **Instrucciones**:
# 1. Calcula estadísticas del vector `features_raw`
# 2. Calcula estadísticas del vector `features_scaled`
# 3. Compara las magnitudes

# %%
# ## RETO 2: Comparar antes y después de escalar
#
# **Objetivo**: Verificar que el escalado funcionó correctamente
#
# **Qué esperamos observar**:
# - Antes: magnitudes muy distintas (ej. 0, 1, 1000000)
# - Después: valores centrados alrededor de 0 y con desviación ≈ 1

# %%

print("\n[5] Verificando escalado...")

sample_df = (
    df_scaled
    .select("features_raw", "features_scaled")
    .limit(1000)
    .toPandas()
)

# Convertir vectores Spark a matrices NumPy
raw_matrix = np.array([
    row['features_raw'].toArray()
    for row in sample_df.to_dict('records')
])

scaled_matrix = np.array([
    row['features_scaled'].toArray()
    for row in sample_df.to_dict('records')
])

print("ANTES DEL ESCALADO")
print(f"Min: {raw_matrix.min():.4f}")
print(f"Max: {raw_matrix.max():.4f}")
print(f"Std: {raw_matrix.std():.4f}")

print("\nDESPUÉS DEL ESCALADO")
print(f"Min: {scaled_matrix.min():.4f}")
print(f"Max: {scaled_matrix.max():.4f}")
print(f"Std: {scaled_matrix.std():.4f}")


# Conclusión automática
print("\n=== CONCLUSIÓN ===")
if scaled_matrix.std() < raw_matrix.std():
    print("✓ El escalado redujo la dispersión de las features")
    print("✓ Las variables ahora tienen magnitudes comparables")
    print("✓ El dataset está mejor preparado para algoritmos de ML")
else:
    print("⚠️ Revisar el proceso de escalado: no se observan mejoras claras")


# ============================================================
# RETO 3: PCA para Reducción de Dimensionalidad
# ============================================================

print("\n" + "="*60)
print("RETO 3 – PCA PARA REDUCCIÓN DE DIMENSIONALIDAD")
print("="*60)

print("""
PREGUNTA:
Si tu vector de features tiene 50 dimensiones,
¿cuántos componentes principales deberías conservar?

OPCIONES:
A) Todos (50)
B) La mitad (25)
C) Los que expliquen ~95% de la varianza
D) Solo 5–10 componentes
""")

print("""
RESPUESTA SELECCIONADA:
Opción C – Conservar los componentes que expliquen la mayor parte de la varianza (≈80%–95%)

JUSTIFICACIÓN:
- PCA no busca mantener un número fijo de dimensiones,
  sino preservar la mayor cantidad de información posible.
- La varianza explicada indica cuánta información del dataset original
  se conserva en los componentes principales.
- En la práctica, se suele apuntar a un rango entre 80% y 95%.
- Como punto de partida, se prueba con un número reducido de componentes
  (por ejemplo 5–10) y luego se valida con la varianza explicada.
""")

# ------------------------------------------------------------
# Configuración de PCA
# ------------------------------------------------------------
from pyspark.ml.feature import PCA

# Determinar cuántas features tiene el vector original
sample_vec = df_scaled.select("features_scaled").first()[0]
num_features = len(sample_vec)

print(f"Número total de features originales: {num_features}")

print("""
DECISIÓN PRÁCTICA:
Se inicia con un valor exploratorio pequeño (k = 5–10),
y posteriormente se valida si ese k explica suficiente varianza.
""")

# Selección inicial de k
k_components = min(10, num_features)

print(f"Valor inicial seleccionado para k: {k_components}")

# Configurar PCA
pca = PCA(
    k=k_components,
    inputCol="features_scaled",
    outputCol="features_pca"
)

print(f"✓ PCA configurado correctamente con k = {k_components}")

# ------------------------------------------------------------
# Entrenamiento y aplicación de PCA
# ------------------------------------------------------------
print("\nEntrenando modelo PCA...")
pca_model = pca.fit(df_scaled)

print("Aplicando transformación PCA al dataset...")
df_pca = pca_model.transform(df_scaled)

print("✓ PCA aplicado correctamente")
print(f"Dimensión original del vector: {num_features}")
print(f"Dimensión reducida tras PCA: {k_components}")

# ------------------------------------------------------------
# Verificación del resultado
# ------------------------------------------------------------
print("\nEsquema de la columna features_pca:")
df_pca.select("features_pca").printSchema()



# ============================================================
# RETO 4: ANÁLISIS DE VARIANZA EXPLICADA (PCA)
# ============================================================

print("\n" + "="*60)
print("RETO 4 – ANÁLISIS DE VARIANZA EXPLICADA")
print("="*60)

print("""
OBJETIVO:
Entender cuánta información del dataset original
se conserva al aplicar PCA.

PREGUNTA:
¿Qué porcentaje de varianza explican los k componentes principales?
""")

# ------------------------------------------------------------
# Varianza explicada por componente
# ------------------------------------------------------------
explained_variance = pca_model.explainedVariance

print("\nVARIANZA EXPLICADA POR COMPONENTE:")
for i, var in enumerate(explained_variance):
    print(f"  PC{i+1}: {var * 100:.2f}%")

# ------------------------------------------------------------
# Varianza explicada acumulada
# ------------------------------------------------------------
print("\nVARIANZA EXPLICADA ACUMULADA:")
cumulative_variance = 0.0

for i, var in enumerate(explained_variance):
    cumulative_variance += var
    print(f"  Hasta PC{i+1}: {cumulative_variance * 100:.2f}%")

# ------------------------------------------------------------
# Componentes necesarios para explicar ≥ 80%
# ------------------------------------------------------------
threshold = 0.80
components_80 = 0
cumulative_variance = 0.0

for var in explained_variance:
    cumulative_variance += var
    components_80 += 1
    if cumulative_variance >= threshold:
        break

print("\nRESULTADO CLAVE:")
print(
    f"Con los primeros {len(explained_variance)} componentes "
    f"solo se explica aproximadamente {sum(explained_variance) * 100:.2f}% de la varianza."
)

print(
    f"Para alcanzar al menos el 80% de la varianza "
    f"se necesitarían aproximadamente {components_80} componentes."
)

# ------------------------------------------------------------
# Respuesta final a la pregunta
# ------------------------------------------------------------
print("""
RESPUESTA FINAL – INTERPRETACIÓN DE LA VARIANZA EXPLICADA

De acuerdo con los resultados obtenidos:

- El primer componente principal (PC1) explica únicamente el 7.76% de la varianza.
- Los siguientes componentes aportan entre 3.7% y 6.4% cada uno.
- Al sumar los primeros 10 componentes, la varianza explicada acumulada es 45.49%.

Esto indica que:

1. La varianza del dataset está distribuida de forma bastante uniforme
   entre muchas dimensiones.
2. No existe un componente dominante que concentre una gran parte
   de la información.
3. Con k = 10 componentes NO se alcanza el umbral recomendado
   de 80% de varianza explicada.

Por lo tanto:

- Para conservar al menos el 80% de la información del dataset,
  sería necesario utilizar un número considerablemente mayor de componentes.
- Usar solo 5–10 componentes implica una reducción agresiva de dimensionalidad
  y una pérdida significativa de información.

Conclusión técnica:

- PCA con k pequeño es útil si el objetivo es compresión,
  reducción de ruido o eficiencia computacional.
- Si el objetivo es maximizar capacidad predictiva,
  se recomienda aumentar k hasta alcanzar al menos 80% de varianza explicada,
  o evaluar si PCA es realmente necesario para este caso.

En este contexto, la elección de k debe balancear:
reducción dimensional vs pérdida de información.
""")


# ============================================================
# RETO 5: PIPELINE COMPLETO
# ============================================================

#
#print("\n" + "=" * 60)
print("RETO 5: PIPELINE COMPLETO")
print("=" * 60)


# Cargar el pipeline de feature engineering (Notebook 03)
feature_pipeline = PipelineModel.load("/opt/spark-data/processed/feature_pipeline")
print("✓ Pipeline de feature engineering cargado desde Notebook 03")

print("""
El pipeline cargado incluye:
- StringIndexer para variables categóricas
- OneHotEncoder para codificación
- VectorAssembler para construir el vector de features_raw
""")

# Construir el pipeline completo
print("\nConstruyendo pipeline completo...")

complete_pipeline_stages = (
    list(feature_pipeline.stages) +
    [scaler, pca]
)

complete_pipeline = Pipeline(stages=complete_pipeline_stages)

print(f"\n✓ Pipeline completo creado con {len(complete_pipeline_stages)} stages:")
for i, stage in enumerate(complete_pipeline_stages, start=1):
    print(f"  Stage {i}: {type(stage).__name__}")

print("""
RESPUESTA A LA PREGUNTA:
¿POR QUÉ ES IMPORTANTE ESTE ORDEN?

El orden del pipeline es crítico porque cada transformación
depende del resultado de la anterior.

1. TRANSFORMACIONES BASE
   Las variables crudas (categóricas y numéricas) deben convertirse primero
   en un vector numérico consistente (features_raw).
   Esto incluye indexación, codificación y ensamblaje.

2. ESCALADO (StandardScaler)
   El escalado se aplica después de construir el vector completo.
   Esto garantiza que todas las features tengan magnitudes comparables
   y evita que algunas dominen el aprendizaje del modelo.

3. REDUCCIÓN DE DIMENSIONALIDAD (PCA)
   PCA debe aplicarse sobre datos ya escalados.
   Si se aplica antes, las features con mayor escala
   dominarían los componentes principales,
   produciendo resultados incorrectos.

BENEFICIOS DEL ORDEN CORRECTO

- Consistencia entre entrenamiento e inferencia
- Reproducibilidad del proceso completo
- Prevención de data leakage
- Correcta interpretación de la varianza en PCA
- Pipeline listo para producción

CONCLUSIÓN

Este orden refleja una buena práctica estándar en pipelines de Machine Learning
y es fundamental para garantizar estabilidad, calidad y confiabilidad del modelo.
""")

# ============================================================
# RETO BONUS 1
# ============================================================

print("\n" + "=" * 60)
print("RETO BONUS 1: EXPERIMENTAR CON DIFERENTES VALORES DE k (PCA)")
print("=" * 60)

# Valores de k a evaluar
k_values = [5, 10, 15, 20]
explained_vars = []

print("\nEJECUTANDO EXPERIMENTO PCA...")
print("Evaluando varianza explicada acumulada para cada valor de k:\n")

for k in k_values:
    k_eff = min(k, num_features)  # Evitar k mayor al número de features
    
    pca_temp = PCA(
        k=k_eff,
        inputCol="features_scaled",
        outputCol="temp_pca"
    )
    
    pca_temp_model = pca_temp.fit(df_scaled)
    cumulative_var = float(sum(pca_temp_model.explainedVariance))
    explained_vars.append(cumulative_var)
    
    print(f"  - k = {k_eff}: {cumulative_var * 100:.2f}% de varianza explicada")

print("\nEXPERIMENTO FINALIZADO")
print("Se procede a graficar los resultados...")

# ------------------------------------------------------------
# Graficar resultados
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(
    k_values,
    [v * 100 for v in explained_vars],
    marker='o',
    linestyle='-'
)
plt.xlabel("Número de Componentes (k)")
plt.ylabel("Varianza Explicada (%)")
plt.title("PCA: Varianza Explicada vs Número de Componentes")
plt.grid(True)

output_plot = "/opt/spark-data/processed/pca_variance.png"
plt.savefig(output_plot, dpi=150, bbox_inches="tight")
plt.close()

print(f"\n✓ Gráfico de varianza explicada guardado en: {output_plot}")

print("""
RESPUESTA Y CONCLUSIÓN

El análisis de la curva k vs varianza explicada permite observar
cómo aumenta la información conservada a medida que se incrementa k.

Criterios utilizados para la decisión:

- No existe un salto abrupto de varianza en valores bajos de k.
- La varianza se incrementa de forma gradual, indicando
  una distribución relativamente uniforme de la información.
- El punto óptimo suele ubicarse donde la curva empieza a estabilizarse
  (criterio del "codo").

Conclusión técnica:

- Un valor bajo de k (5–10) ofrece buena reducción dimensional,
  pero conserva menos de la mitad de la información.
- Valores más altos de k incrementan la varianza explicada,
  a costa de menor reducción dimensional.
- La selección final de k debe balancear:
    eficiencia computacional vs desempeño del modelo.

La decisión óptima de k debe validarse posteriormente
comparando métricas del modelo con y sin PCA.
""")

# ------------------------------------------------------------
# Selección de columna objetivo (label)
# ------------------------------------------------------------
print("\nPREPARANDO DATASET FINAL PARA MACHINE LEARNING...")

if "valor_del_contrato_num" in df_pca.columns:
    df_ml_ready = df_pca.select(
        "features_pca",
        col("valor_del_contrato_num").alias("label")
    )
    print("✓ Columna objetivo 'label' creada correctamente")
else:
    print("⚠️ ADVERTENCIA: No se encontró columna objetivo.")
    print("Se continuará con el dataset completo sin label explícito.")
    df_ml_ready = df_pca

print("\n" + "=" * 60)
print("PREGUNTAS DE REFLEXIÓN")
print("=" * 60)

# ------------------------------------------------------------
# Preguntas de Reflexión
# ------------------------------------------------------------
print("""
1. ¿Por qué StandardScaler usa withMean=False?

StandardScaler se configura con withMean=False porque las features
incluyen vectores sparse generados por OneHotEncoder.

Centrar los datos (restar la media) convertiría estos vectores
a formato denso, lo que incrementaría significativamente
el uso de memoria y afectaría el rendimiento en Spark.

Mantener withMean=False permite:
- Preservar el formato sparse
- Reducir consumo de memoria
- Mantener escalabilidad del pipeline
""")

# ------------------------------------------------------------
# Pregunta 2
# ------------------------------------------------------------
print("""
2. ¿Cuándo NO deberías usar PCA?

PCA no es recomendable en los siguientes casos:

- Cuando la interpretabilidad de las variables originales es crítica,
  ya que los componentes principales son combinaciones lineales.
- Cuando se utilizan modelos basados en árboles
  (Random Forest, Gradient Boosting), que manejan bien alta dimensionalidad.
- Cuando las features ya están bien escaladas
  y no presentan redundancia significativa.
- Cuando el costo de perder explicabilidad es mayor
  que el beneficio de reducir dimensionalidad.
""")

# ------------------------------------------------------------
# Pregunta 3
# ------------------------------------------------------------
print("""
3. Si tienes 100 features y aplicas PCA con k=10, ¿perdiste información?

Sí, se pierde información, pero de forma controlada.

PCA conserva la mayor cantidad posible de varianza en los primeros
componentes, por lo que la información descartada suele corresponder
a ruido o redundancia.

La pérdida es aceptable cuando:
- La varianza explicada acumulada es alta (ej. ≥80%)
- El desempeño del modelo se mantiene o mejora
- Se reduce complejidad y tiempo de entrenamiento
""")

# ------------------------------------------------------------
# Pregunta 4
# ------------------------------------------------------------
print("""
4. ¿Qué ventaja tiene aplicar StandardScaler ANTES de PCA?

PCA es altamente sensible a la escala de las variables.

Aplicar StandardScaler antes de PCA garantiza que:
- Todas las features contribuyan de manera equilibrada
- Variables con grandes magnitudes no dominen los componentes
- La varianza capturada por PCA refleje estructura real de los datos

Este orden es fundamental para obtener componentes principales
matemáticamente correctos y estables.
""")

print("\n" + "=" * 60)
print("FIN DE PREGUNTAS DE REFLEXIÓN")
print("=" * 60)
    

# ------------------------------------------------------------
# Guardar dataset ML-ready
# ------------------------------------------------------------
output_path = "/opt/spark-data/processed/secop_ml_ready.parquet"
df_ml_ready.write.mode("overwrite").parquet(output_path)

print(f"\n✓ Dataset ML-ready guardado en: {output_path}")
print(f"✓ Registros finales: {df_ml_ready.count():,}")
print(f"✓ Columnas finales: {len(df_ml_ready.columns)}")

print("\n" + "=" * 60)
print("RESUMEN FINAL – PCA Y TRANSFORMACIONES")
print("=" * 60)
print("✓ Evaluación experimental de múltiples valores de k")
print("✓ Análisis de varianza explicada y curva de estabilización")
print("✓ Selección informada del número de componentes")
print("✓ Dataset preparado para entrenamiento de modelos ML")
print("=" * 60)

# ------------------------------------------------------------
# Detener SparkSession
# ------------------------------------------------------------
spark.stop()
print("✓ SparkSession detenida correctamente")


