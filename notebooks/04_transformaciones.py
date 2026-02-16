# %% [markdown]
# # Notebook 04: Transformaciones Avanzadas
#
# **Sección 13**: StandardScaler, PCA y Normalización
#
# **Objetivo**: Aplicar transformaciones avanzadas para mejorar el desempeño del modelo.
#
# ## Actividades:
# 1. Normalizar features numéricas con StandardScaler
# 2. Aplicar PCA para reducción de dimensionalidad
# 3. Construir pipeline completo
# 4. Comparar resultados con y sin transformaciones

# %%
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, PCA, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import col
from delta import configure_spark_with_delta_pip


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
# Cargar datos transformados del notebook anterior
df = spark.read.parquet("/opt/spark-data/processed/secop_features.parquet")
print(f"Registros: {df.count():,}")
print(f"Columnas: {len(df.columns)}")

# %% [markdown]
# ## RETO 1: ¿Por qué normalizar?
#
# **Pregunta de análisis**: Examina los valores en `features_raw`.
# ¿Hay features con escalas muy diferentes?
#
# **Instrucciones**:
# 1. Toma una muestra de 5 registros
# 2. Convierte `features_raw` a array y examina los valores
# 3. Identifica si hay features con magnitudes muy diferentes (ej: 0.01 vs 1000000)
# 4. Explica por qué esto es un problema para ML

#1 

sample = df.select("features_raw").limit(5).collect()

#2
for i, row in enumerate(sample, start=1):
    features_array = row["features_raw"].toArray()
    print(f"Registro {i} - primeros 10 valores:")
    print(features_array[:10])

#3
# Se observan diferencias claras en las magnitudes de las features.
# Algunas variables tienen valores pequeños (0 o 1), típicos de
# variables categóricas codificadas con OneHotEncoder,
# mientras que otras variables numéricas pueden tener valores
# considerablemente mayores.

#4
# Esto es un problema porque muchos algoritmos de ML
# (regresión, k-means, PCA, SVM) son sensibles a la escala
# y las features grandes dominan el aprendizaje.


# TODO: Responde:
# ¿Observas diferencias grandes en las magnitudes? (Sí/No)
# Sí, se observan diferencias grandes en las magnitudes.
# Algunas features tienen valores cercanos a 0/1 (OneHot),
# mientras que otras pueden tener valores muy altos.
# ¿Por qué es importante normalizar?
# Normalizar asegura que todas las variables contribuyan
# de manera equilibrada al modelo.

# %% [markdown]
# ## PASO 1: StandardScaler
#
# **Concepto**: StandardScaler centra los datos (media=0) y escala (std=1)
#
# Formula: z = (x - μ) / σ

# %%
# TODO: Crea un StandardScaler
# - inputCol: "features_raw" (del notebook anterior)
# - outputCol: "features_scaled"
# - withMean: False (requerido para vectores sparse)
# - withStd: True (normalizar por desviación estándar)
# %% [markdown]
# ## PASO 1: StandardScaler

# %%
from pyspark.ml.feature import StandardScaler

# Crear StandardScaler
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features_scaled",
    withMean=False,  # Requerido para vectores sparse
    withStd=True     # Normaliza por desviación estándar
)

print("✓ StandardScaler creado correctamente")

# %%
# Entrenar el scaler (fit)
scaler_model = scaler.fit(df)

# Aplicar transformación (transform)
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
import pandas as pd
import numpy as np

# Tomar una muestra manejable para análisis local
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

# Estadísticas antes del escalado
print("=== ANTES (features_raw) ===")
print(f"Min:   {raw_matrix.min():.4f}")
print(f"Max:   {raw_matrix.max():.4f}")
print(f"Mean:  {raw_matrix.mean():.4f}")
print(f"Std:   {raw_matrix.std():.4f}")

# Estadísticas después del escalado
print("\n=== DESPUÉS (features_scaled) ===")
print(f"Min:   {scaled_matrix.min():.4f}")
print(f"Max:   {scaled_matrix.max():.4f}")
print(f"Mean:  {scaled_matrix.mean():.4f}")
print(f"Std:   {scaled_matrix.std():.4f}")

# %%
# Conclusión automática
print("\n=== CONCLUSIÓN ===")
if scaled_matrix.std() < raw_matrix.std():
    print("✓ El escalado redujo la dispersión de las features")
    print("✓ Las variables ahora tienen magnitudes comparables")
    print("✓ El dataset está mejor preparado para algoritmos de ML")
else:
    print("⚠️ Revisar el proceso de escalado: no se observan mejoras claras")


# %% [markdown]
# ## RETO 3: PCA para Reducción de Dimensionalidad
#
# **Pregunta**: Si tu vector de features tiene 50 dimensiones,
# ¿cuántos componentes principales deberías conservar?
#
# **Opciones**:
# - A) Todos (50)
# - B) La mitad (25)
# - C) Los que expliquen 95% de la varianza
# - D) Solo 5-10 componentes
#
# **Justifica tu respuesta**

# %%
# TODO: Configura PCA
# - inputCol: "features_scaled" (ya normalizadas)
# - outputCol: "features_pca"
# - k: número de componentes (experimenta con diferentes valores)
# %% [markdown]
# ## RETO 3: PCA para Reducción de Dimensionalidad
#
# **Pregunta**: Si tu vector de features tiene 50 dimensiones,
# ¿cuántos componentes principales deberías conservar?
#
# **Respuesta conceptual**:
# La mejor práctica es conservar los componentes que expliquen
# la mayor parte de la varianza (≈80%–95%). Como punto de partida,
# se usa un número reducido (5–10) y luego se valida con la varianza explicada.

# %%
from pyspark.ml.feature import PCA

# ¿Cuántas features tiene el vector?
sample_vec = df_scaled.select("features_scaled").first()[0]
num_features = len(sample_vec)
print(f"Número total de features: {num_features}")

# Decisión inicial de k (exploratoria)
# Se empieza con un valor pequeño y luego se valida con varianza explicada
k_components = min(10, num_features)

# Configurar PCA
pca = PCA(
    k=k_components,
    inputCol="features_scaled",
    outputCol="features_pca"
)

print(f"✓ PCA configurado con k={k_components} componentes")

# %%
# Entrenar PCA
pca_model = pca.fit(df_scaled)

# Aplicar transformación
df_pca = pca_model.transform(df_scaled)

print("✓ PCA aplicado correctamente")
print(f"Dimensión original del vector: {num_features}")
print(f"Dimensión reducida con PCA: {k_components}")

# %%
# Verificar esquema del nuevo vector
df_pca.select("features_pca").printSchema()


# %% [markdown]
# ## RETO 4: Analizar Varianza Explicada
#
# **Objetivo**: Entender cuánta información conservamos con PCA
#
# **Pregunta**: ¿Qué porcentaje de varianza explican los k componentes?

# %%
# TODO: Obtén la varianza explicada por cada componente

# %%
# Obtener la varianza explicada por cada componente principal
explained_variance = pca_model.explainedVariance

print("\n=== VARIANZA EXPLICADA POR COMPONENTE ===")
for i, var in enumerate(explained_variance):
    print(f"Componente {i+1}: {var * 100:.2f}%")

# %%
# Calcular varianza acumulada
print("\n=== VARIANZA EXPLICADA ACUMULADA ===")
cumulative_variance = 0.0

for i, var in enumerate(explained_variance):
    cumulative_variance += var
    print(f"Acumulada hasta PC{i+1}: {cumulative_variance * 100:.2f}%")

# %%
# Identificar cuántos componentes explican al menos el 80% de la varianza
threshold = 0.80
components_80 = 0
cumulative_variance = 0.0

for var in explained_variance:
    cumulative_variance += var
    components_80 += 1
    if cumulative_variance >= threshold:
        break

print(f"\n✓ Componentes necesarios para explicar al menos el 80% de la varianza: {components_80}")

# %%
# Respuesta conceptual (documentación)
print(
    "\nRespuesta:\n"
    "Se requieren aproximadamente "
    f"{components_80} componentes principales "
    "para explicar al menos el 80% de la varianza total del dataset."
)

# TODO: Responde:
# ¿Cuántos componentes necesitas para explicar al menos 80% de la varianza?
# Respuesta:


# %% [markdown]
# ## RETO 5: Pipeline Completo
#
# **Objetivo**: Integrar todas las transformaciones en un solo pipeline
#
# **Orden correcto**:
# 1. Cargar pipeline de feature engineering (notebook 03)
# 2. Agregar StandardScaler
# 3. Agregar PCA
#
# **Pregunta**: ¿Por qué es importante este orden?

# %%
from pyspark.ml import Pipeline, PipelineModel

# Cargar el pipeline de feature engineering (Notebook 03)
feature_pipeline = PipelineModel.load("/opt/spark-data/processed/feature_pipeline")
print("✓ Pipeline de feature engineering cargado")

# %%
# Construir el pipeline completo
# Orden:
# 1. Transformaciones categóricas y numéricas (StringIndexer, OneHotEncoder, VectorAssembler)
# 2. Escalado (StandardScaler)
# 3. Reducción de dimensionalidad (PCA)

complete_pipeline_stages = (
    list(feature_pipeline.stages) + 
    [scaler, pca]
)

complete_pipeline = Pipeline(stages=complete_pipeline_stages)

print(f"\n✓ Pipeline completo creado con {len(complete_pipeline_stages)} stages:")
for i, stage in enumerate(complete_pipeline_stages, start=1):
    print(f"  Stage {i}: {type(stage).__name__}")

# %%
# Explicación conceptual (documentación)
print(
    "\nJustificación del orden del pipeline:\n"
    "1. Primero se transforman las variables crudas (categóricas y numéricas) en features numéricas.\n"
    "2. Luego se normalizan las features con StandardScaler para evitar problemas de escala.\n"
    "3. Finalmente se aplica PCA, que requiere datos escalados para calcular correctamente la varianza.\n"
    "Este orden garantiza consistencia, reproducibilidad y evita data leakage."
)


# %% [markdown]
# ## RETO BONUS 1: Experimentar con diferentes valores de k
#
# **Objetivo**: Encontrar el número óptimo de componentes PCA
#
# **Instrucciones**:
# 1. Prueba con k = [5, 10, 15, 20]
# 2. Para cada k, calcula la varianza acumulada
# 3. Grafica k vs varianza explicada
# 4. Decide cuál es el mejor k (balance entre reducción y información)

# %%
# TODO: Implementa el experimento
# import matplotlib.pyplot as plt
#
# %% [markdown]
# ## RETO BONUS 1: Experimentar con diferentes valores de k
#
# **Objetivo**: Encontrar el número óptimo de componentes PCA
#
# **Instrucciones**:
# 1. Probar con k = [5, 10, 15, 20]
# 2. Para cada k, calcular la varianza acumulada
# 3. Graficar k vs varianza explicada
# 4. Decidir el mejor k (balance entre reducción y retención de información)

# %%
import matplotlib.pyplot as plt
from pyspark.ml.feature import PCA

# Valores de k a evaluar
k_values = [5, 10, 15, 20]
explained_vars = []

print("\n=== EXPERIMENTO PCA: VARIANZA VS COMPONENTES ===")

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
    
    print(f"k = {k_eff}: {cumulative_var * 100:.2f}% de varianza explicada")

# %%
# Graficar resultados
plt.figure(figsize=(8, 5))
plt.plot(
    k_values,
    [v * 100 for v in explained_vars],
    marker='o',
    linestyle='-'
)
plt.xlabel('Número de Componentes (k)')
plt.ylabel('Varianza Explicada (%)')
plt.title('PCA: Varianza Explicada vs Número de Componentes')
plt.grid(True)

output_plot = "/opt/spark-data/processed/pca_variance.png"
plt.savefig(output_plot, dpi=150, bbox_inches="tight")
plt.close()

print(f"\n✓ Gráfico guardado en: {output_plot}")

# %%
# Conclusión sugerida (documentación)
print(
    "\nConclusión sugerida:\n"
    "- El mejor valor de k es aquel donde la curva empieza a estabilizarse (codo).\n"
    "- Normalmente se elige el menor k que explica al menos 80%–95% de la varianza.\n"
    "- Esto logra un buen balance entre reducción dimensional y preservación de información."
)

# %%
# Seleccionar columna objetivo (label) para ML

if "valor_del_contrato_num" in df_pca.columns:
    df_ml_ready = df_pca.select(
        "features_pca",
        col("valor_del_contrato_num").alias("label")
    )
    print("✓ Columna objetivo 'label' creada desde valor_del_contrato_num")
else:
    print("⚠️ ADVERTENCIA: No se encontró columna de valor. Se usará el dataset completo.")
    df_ml_ready = df_pca

# %%
# Guardar dataset listo para ML
output_path = "/opt/spark-data/processed/secop_ml_ready.parquet"
df_ml_ready.write.mode("overwrite").parquet(output_path)

print(f"\n✓ Dataset ML-ready guardado en: {output_path}")
print(f"✓ Registros finales: {df_ml_ready.count():,}")
print(f"✓ Columnas finales: {len(df_ml_ready.columns)}")

# %% [markdown]
# ## Preguntas de Reflexión
#
# 1. **¿Por qué StandardScaler usa withMean=False?**
#    Porque los vectores generados por OneHotEncoder son sparse vectors.
#    Centrar los datos (restar la media) convertiría el vector a denso,
#    incrementando significativamente el uso de memoria y afectando el rendimiento.
#
# 2. **¿Cuándo NO deberías usar PCA?**
#    - Cuando la interpretabilidad de las variables es crítica.
#    - Cuando el modelo ya maneja bien alta dimensionalidad (ej. árboles).
#    - Cuando las features ya están bien escaladas y no son redundantes.
#
# 3. **Si tienes 100 features y aplicas PCA con k=10, ¿perdiste información?**
#    Sí, pero de forma controlada. PCA conserva la mayor varianza posible,
#    por lo que la información perdida suele ser ruido o redundancia.
#
# 4. **¿Qué ventaja tiene aplicar StandardScaler ANTES de PCA?**
#    PCA es sensible a la escala de las variables.
#    Escalar previamente evita que features con grandes magnitudes
#    dominen los componentes principales.

# %%
print("\n" + "=" * 60)
print("RESUMEN DE TRANSFORMACIONES")
print("=" * 60)
print("✓ Features normalizadas con StandardScaler")
print(f"✓ Dimensionalidad reducida: {num_features} → {k_components}")
print(f"✓ Varianza explicada: {sum(pca_model.explainedVariance) * 100:.2f}%")
print("✓ Dataset listo para entrenar modelos de Machine Learning")
print("=" * 60)

# %%
# Detener SparkSession
spark.stop()
print("✓ SparkSession detenida correctamente")

