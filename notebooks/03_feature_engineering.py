# %% [markdown]
# # Notebook 03: Feature Engineering con Pipelines
#
# **Sección 13 - Spark ML**: Construcción de pipelines end-to-end
#
# **Objetivo**: Aplicar VectorAssembler y construir un pipeline de transformación.
#
# **Conceptos clave**:
# - **Transformer**: Aplica transformaciones (ej: StringIndexer)
# - **Estimator**: Aprende de los datos y genera un modelo
# - **Pipeline**: Encadena múltiples stages secuencialmente
#
# ## Actividades:
# 1. Crear StringIndexer para variables categóricas
# 2. Aplicar OneHotEncoder
# 3. Combinar features con VectorAssembler
# 4. Construir y ejecutar Pipeline

# %%
from pyspark.sql import SparkSession
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when, isnull

# %%
# Configurar SparkSession
spark = SparkSession.builder \
    .appName("SECOP_FeatureEngineering") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

# %%
# Cargar datos
df = spark.read.parquet("/opt/spark-data/processed/secop_eda.parquet")
print(f"Registros cargados: {df.count():,}")

# %%
# Explorar columnas disponibles
print("Columnas disponibles:")
for col_name in df.columns:
    print(f"  - {col_name}")

# %% [markdown]
# ## RETO 1: Selección de Features
#
# **Objetivo**: Identificar las mejores variables para predecir el valor del contrato.
#
# **Instrucciones**:
# 1. Analiza las columnas disponibles en el dataset
# 2. Selecciona al menos 3 variables categóricas relevantes
# 3. Selecciona al menos 2 variables numéricas relevantes
# 4. Justifica tu selección con un comentario

# %%
# TODO: Define tus variables categóricas
# Ejemplo: categorical_cols = ["departamento", "tipo_de_contrato", ...]
categorical_cols = [
    # TODO: Completa con tus columnas categóricas
]

# TODO: Define tus variables numéricas
# Ejemplo: numeric_cols = ["plazo_de_ejec_del_contrato", ...]
numeric_cols = [
    # TODO: Completa con tus columnas numéricas
]

# Verificar qué columnas existen realmente
available_cat = [c for c in categorical_cols if c in df.columns]
available_num = [c for c in numeric_cols if c in df.columns]

print(f"Categóricas seleccionadas: {available_cat}")
print(f"Numéricas seleccionadas: {available_num}")

# %% [markdown]
# ## RETO 2: Limpieza de Datos
#
# **Pregunta**: ¿Qué estrategia usarás para manejar valores nulos?
# - Opción A: Eliminar filas con nulos (dropna)
# - Opción B: Imputar valores (usar Imputer)
# - Opción C: Crear una categoría "DESCONOCIDO" para categóricas
#
# **Justifica tu decisión en un comentario**

# %%
# TODO: Implementa tu estrategia de limpieza de datos
# df_clean = ...

# SOLUCIÓN SUGERIDA (descomenta y adapta):
# df_clean = df.dropna(subset=available_cat + available_num)

print(f"Registros después de limpiar: {df_clean.count():,}")

# %% [markdown]
# ## PASO 1: StringIndexer para Variables Categóricas
#
# **Concepto**: StringIndexer convierte strings en índices numéricos.
# Por ejemplo: ["Bogotá", "Cali", "Bogotá"] → [0, 1, 0]

# %%
# TODO: Crea una lista de StringIndexers
# Pista: Usa list comprehension para crear un indexer por cada columna categórica
# Recuerda usar handleInvalid="keep" para manejar valores no vistos en test

indexers = [
    # TODO: Completa aquí
    # StringIndexer(inputCol=???, outputCol=???, handleInvalid="keep")
]

print("StringIndexers creados:")
for idx in indexers:
    print(f"  - {idx.getInputCol()} -> {idx.getOutputCol()}")

# %% [markdown]
# ## PASO 2: OneHotEncoder
#
# **Concepto**: OneHotEncoder convierte índices en vectores binarios.
# Por ejemplo: [0, 1, 0] → [[1,0], [0,1], [1,0]]

# %%
# TODO: Crea una lista de OneHotEncoders
# Pista: La columna de entrada debe ser la salida del StringIndexer (_idx)

encoders = [
    # TODO: Completa aquí
    # OneHotEncoder(inputCol=???, outputCol=???)
]

print("\nOneHotEncoders creados:")
for enc in encoders:
    print(f"  - {enc.getInputCol()} -> {enc.getOutputCol()}")

# %% [markdown]
# ## RETO 3: VectorAssembler
#
# **Objetivo**: Combinar todas las features en un solo vector.
#
# **Pregunta de reflexión**: ¿Por qué necesitamos combinar features numéricas
# y categóricas codificadas en un solo vector?

# %%
# TODO: Crea el VectorAssembler
# Pista: inputCols debe incluir:
#   1. Variables numéricas originales (available_num)
#   2. Variables categóricas codificadas (con sufijo "_vec")

feature_cols = [
    # TODO: Completa la lista de columnas a combinar
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

print(f"\nVectorAssembler combinará {len(feature_cols)} features:")
print(feature_cols)

# %% [markdown]
# ## RETO 4: Construir el Pipeline
#
# **Objetivo**: Encadenar todos los transformadores en un Pipeline.
#
# **Pregunta**: ¿Cuál es el orden correcto de los stages?
# Pista: Piensa en las dependencias (¿qué necesita qué?)

# %%
# TODO: Construye la lista de stages del pipeline
# Orden correcto: indexers → encoders → assembler

pipeline_stages = [
    # TODO: Completa aquí
]

pipeline = Pipeline(stages=pipeline_stages)

print(f"\nPipeline con {len(pipeline_stages)} stages:")
for i, stage in enumerate(pipeline_stages):
    print(f"  Stage {i+1}: {type(stage).__name__}")

# %% [markdown]
# ## PASO 5: Entrenar y Aplicar el Pipeline

# %%
# Entrenar el pipeline (fit)
print("\nEntrenando pipeline...")
pipeline_model = pipeline.fit(df_clean)
print("✓ Pipeline entrenado")

# Aplicar transformaciones (transform)
df_transformed = pipeline_model.transform(df_clean)

print(f"✓ Transformación completada")
print(f"Columnas después de transformar: {len(df_transformed.columns)}")

# %%
# Verificar el resultado
print("\nEsquema de features_raw:")
df_transformed.select("features_raw").printSchema()

# Ver dimensión del vector de features
sample_features = df_transformed.select("features_raw").first()[0]
print(f"Dimensión del vector de features: {len(sample_features)}")

# %% [markdown]
# ## RETO BONUS 1: ¿Cuántas features se generaron?
#
# **Pregunta**: Si tienes:
# - 2 variables numéricas
# - 3 variables categóricas con [10, 5, 3] categorías únicas
#
# ¿Cuántas features tendrás después de OneHotEncoding?
#
# **Calcula manualmente y verifica con el código**

# %%
# TODO: Calcula cuántas features hay por cada variable categórica
# Pista: Usa df_clean.select("columna").distinct().count()

for cat_col in available_cat:
    num_categorias = df_clean.select(cat_col).distinct().count()
    print(f"{cat_col}: {num_categorias} categorías únicas")

# TODO: Suma total de features = numéricas + (categóricas codificadas)
# ¿Coincide con len(sample_features)?

# %% [markdown]
# ## RETO BONUS 2: Feature Importance Manual
#
# **Objetivo**: Analizar la distribución de valores en el vector de features
#
# **Instrucciones**:
# 1. Toma una muestra de 1000 registros
# 2. Convierte el vector de features a una matriz de Pandas
# 3. Calcula la varianza de cada feature
# 4. Identifica las top 5 features con mayor varianza

# %%
# TODO: Implementa el análisis de varianza de features
# Pista: Usa .toPandas() y numpy para calcular varianza

# CÓDIGO SUGERIDO (descomentar y completar):
# import pandas as pd
# import numpy as np
#
# sample_df = df_transformed.select("features_raw").sample(0.01).limit(1000).toPandas()
# features_matrix = np.array([row['features_raw'].toArray() for row in sample_df])
#
# variances = np.var(features_matrix, axis=0)
# top_5_idx = np.argsort(variances)[-5:]
#
# print("Top 5 features con mayor varianza:")
# for idx in top_5_idx:
#     print(f"  Feature {idx}: varianza = {variances[idx]:.2f}")

# %%
# Guardar pipeline entrenado
pipeline_path = "/opt/spark-data/processed/feature_pipeline"
pipeline_model.write().overwrite().save(pipeline_path)
print(f"\n✓ Pipeline guardado en: {pipeline_path}")

# %%
# Guardar dataset transformado
output_path = "/opt/spark-data/processed/secop_features.parquet"
df_transformed.write.mode("overwrite").parquet(output_path)
print(f"✓ Dataset transformado guardado en: {output_path}")

# %% [markdown]
# ## Preguntas de Reflexión
#
# Responde en un comentario:
#
# 1. **¿Por qué usamos Pipeline en lugar de aplicar transformaciones individuales?**
#
# 2. **¿Qué pasaría si aplicamos OneHotEncoder antes de StringIndexer?**
#
# 3. **¿Cuándo usarías StandardScaler en el pipeline?**
#
# 4. **¿Qué ventaja tiene guardar el pipeline_model en lugar del DataFrame transformado?**

# %%
# TODO: Escribe tus respuestas aquí como comentarios
# 1. Pipeline:
# 2. Orden de transformaciones:
# 3. StandardScaler:
# 4. Guardar pipeline:

# %%
print("\n" + "="*60)
print("RESUMEN FEATURE ENGINEERING")
print("="*60)
print(f"✓ Variables categóricas procesadas: {len(available_cat)}")
print(f"✓ Variables numéricas: {len(available_num)}")
print(f"✓ Dimensión final del vector: {len(sample_features)}")
print(f"✓ Pipeline guardado y listo para usar")
print("="*60)

# %%
spark.stop()
