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

# %%
spark = SparkSession.builder \
    .appName("SECOP_Transformaciones") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

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

# %%
# TODO: Examina los valores del vector de features
# Pista: usa .toArray() para convertir el vector a lista

sample = df.select("features_raw").limit(5).collect()

# TODO: Imprime los primeros 10 valores de cada vector
# for row in sample:
#     features_array = row['features_raw'].toArray()
#     print(features_array[:10])

# TODO: Responde:
# ¿Observas diferencias grandes en las magnitudes? (Sí/No)
# ¿Por qué es importante normalizar?
# Respuesta:

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

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features_scaled",
    withMean=False,  # No centra (incompatible con sparse vectors)
    withStd=True     # Normaliza por std
)

print("✓ StandardScaler creado")

# %%
# Entrenar el scaler
scaler_model = scaler.fit(df)
df_scaled = scaler_model.transform(df)

print("✓ Features escaladas")
print("\nColumnas nuevas:")
print(df_scaled.columns[-3:])  # Últimas 3 columnas

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
# TODO: Convierte una muestra a pandas y calcula estadísticas
# import pandas as pd
# import numpy as np
#
# sample_df = df_scaled.select("features_raw", "features_scaled").limit(1000).toPandas()
#
# # Convertir vectores a matrices
# raw_matrix = np.array([row['features_raw'].toArray() for row in sample_df.to_dict('records')])
# scaled_matrix = np.array([row['features_scaled'].toArray() for row in sample_df.to_dict('records')])
#
# print("ANTES (features_raw):")
# print(f"  Min: {raw_matrix.min():.2f}")
# print(f"  Max: {raw_matrix.max():.2f}")
# print(f"  Mean: {raw_matrix.mean():.2f}")
# print(f"  Std: {raw_matrix.std():.2f}")
#
# print("\nDESPUÉS (features_scaled):")
# print(f"  Min: {scaled_matrix.min():.2f}")
# print(f"  Max: {scaled_matrix.max():.2f}")
# print(f"  Mean: {scaled_matrix.mean():.2f}")
# print(f"  Std: {scaled_matrix.std():.2f}")

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

# ¿Cuántas features tiene tu vector?
sample_vec = df_scaled.select("features_scaled").first()[0]
num_features = len(sample_vec)
print(f"Número total de features: {num_features}")

# TODO: Decide cuántos componentes usar
# Sugerencia: Empieza con min(10, num_features)
k_components = min(10, num_features)

pca = PCA(
    k=k_components,
    inputCol="features_scaled",
    outputCol="features_pca"
)

print(f"✓ PCA configurado con k={k_components} componentes")

# %%
# Entrenar PCA
pca_model = pca.fit(df_scaled)
df_pca = pca_model.transform(df_scaled)

print("✓ PCA aplicado")
print(f"Dimensión original: {num_features}")
print(f"Dimensión reducida: {k_components}")

# %% [markdown]
# ## RETO 4: Analizar Varianza Explicada
#
# **Objetivo**: Entender cuánta información conservamos con PCA
#
# **Pregunta**: ¿Qué porcentaje de varianza explican los k componentes?

# %%
# TODO: Obtén la varianza explicada por cada componente
explained_variance = pca_model.explainedVariance

print("\n=== VARIANZA EXPLICADA POR COMPONENTE ===")
for i, var in enumerate(explained_variance):
    print(f"Componente {i+1}: {var*100:.2f}%")

# TODO: Calcula la varianza acumulada
cumulative_variance = 0
for i, var in enumerate(explained_variance):
    cumulative_variance += var
    print(f"Acumulada hasta PC{i+1}: {cumulative_variance*100:.2f}%")

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
# TODO: Carga el pipeline del notebook 03
feature_pipeline = PipelineModel.load("/opt/spark-data/processed/feature_pipeline")
print("✓ Pipeline de features cargado")

# TODO: Crea un nuevo pipeline que incluya:
# - Stages del pipeline anterior (feature_pipeline.stages)
# - StandardScaler
# - PCA

complete_pipeline_stages = list(feature_pipeline.stages) + [scaler, pca]

complete_pipeline = Pipeline(stages=complete_pipeline_stages)

print(f"\n✓ Pipeline completo con {len(complete_pipeline_stages)} stages:")
for i, stage in enumerate(complete_pipeline_stages):
    print(f"  {i+1}. {type(stage).__name__}")

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
# k_values = [5, 10, 15, 20]
# explained_vars = []
#
# for k in k_values:
#     pca_temp = PCA(k=min(k, num_features), inputCol="features_scaled", outputCol="temp_pca")
#     pca_temp_model = pca_temp.fit(df_scaled)
#     cumulative_var = sum(pca_temp_model.explainedVariance)
#     explained_vars.append(cumulative_var)
#     print(f"k={k}: {cumulative_var*100:.2f}% varianza explicada")
#
# # Graficar
# plt.figure(figsize=(8, 5))
# plt.plot(k_values, [v*100 for v in explained_vars], marker='o')
# plt.xlabel('Número de Componentes (k)')
# plt.ylabel('Varianza Explicada (%)')
# plt.title('PCA: Varianza Explicada vs Componentes')
# plt.grid(True)
# plt.savefig('/opt/spark-data/processed/pca_variance.png')
# print("Gráfico guardado en /opt/spark-data/processed/pca_variance.png")

# %%
# Seleccionar columna objetivo (label) para ML
# Asumimos que existe una columna con el valor del contrato

if "valor_del_contrato_num" in df_pca.columns:
    df_ml_ready = df_pca.select(
        "features_pca",
        col("valor_del_contrato_num").alias("label")
    )
else:
    print("ADVERTENCIA: No se encontró columna de valor. Usando todas las columnas.")
    df_ml_ready = df_pca

# %%
# Guardar dataset listo para ML
output_path = "/opt/spark-data/processed/secop_ml_ready.parquet"
df_ml_ready.write.mode("overwrite").parquet(output_path)
print(f"\n✓ Dataset ML-ready guardado en: {output_path}")

# %% [markdown]
# ## Preguntas de Reflexión
#
# 1. **¿Por qué StandardScaler usa withMean=False?**
#    Respuesta:
#
# 2. **¿Cuándo NO deberías usar PCA?**
#    Respuesta:
#
# 3. **Si tienes 100 features y aplicas PCA con k=10, ¿perdiste información?**
#    Respuesta:
#
# 4. **¿Qué ventaja tiene aplicar StandardScaler ANTES de PCA?**
#    Respuesta:

# %%
print("\n" + "="*60)
print("RESUMEN DE TRANSFORMACIONES")
print("="*60)
print(f"✓ Features normalizadas con StandardScaler")
print(f"✓ Dimensionalidad reducida: {num_features} → {k_components}")
print(f"✓ Varianza explicada: {sum(pca_model.explainedVariance)*100:.2f}%")
print(f"✓ Dataset listo para entrenar modelos")
print("="*60)

# %%
spark.stop()
