# ============================================================
# NOTEBOOK 03: Feature Engineering con Spark ML
# ============================================================

from pyspark.sql import SparkSession
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when, isnull
from delta import configure_spark_with_delta_pip

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
print("\n[2] Cargando dataset procesado para EDA / Feature Engineering...")

df = spark.read.parquet("/opt/spark-data/processed/secop_eda.parquet")
print(f"✓ Registros cargados: {df.count():,}")

print("\nColumnas disponibles en el dataset:")
for c in df.columns:
    print(f"  - {c}")

# ------------------------------------------------------------
# RETO 1: Selección de Features
# ------------------------------------------------------------
print("\n[3] RETO 1 – Selección de Features")
print("""
Objetivo:
Identificar variables relevantes para predecir el valor del contrato.
""")

categorical_cols = [
    "departamento",
    "tipo_de_contrato",
    "estado_contrato"
]

numeric_cols = [
    "plazo_de_ejecucion",
    "valor_contrato_num"
]

available_cat = [c for c in categorical_cols if c in df.columns]
available_num = [c for c in numeric_cols if c in df.columns]

print(f"Variables categóricas seleccionadas: {available_cat}")
print(f"Variables numéricas seleccionadas: {available_num}")

print("""
JUSTIFICACIÓN:
- Variables categóricas:
  Capturan contexto territorial, contractual y estado del proceso,
  factores que influyen directamente en el valor del contrato.

- Variables numéricas:
  Representan magnitud y duración, altamente correlacionadas con el valor.
""")

# ------------------------------------------------------------
# RETO 2: Limpieza de Datos
# ------------------------------------------------------------
print("\n[4] RETO 2 – Limpieza de Datos")
print("""
Estrategia de manejo de nulos:
- Categóricas  → Reemplazar por 'DESCONOCIDO'
- Numéricas    → Eliminar registros con nulos
""")

df_clean = df

for c in available_cat:
    df_clean = df_clean.withColumn(
        c,
        when(isnull(col(c)), "DESCONOCIDO").otherwise(col(c))
    )

df_clean = df_clean.dropna(subset=available_num)

print(f"✓ Registros después de limpieza: {df_clean.count():,}")

print("""
RAZÓN:
- Evitamos imputaciones arbitrarias en variables numéricas.
- Preservamos registros categóricos sin perder información.
""")

# ------------------------------------------------------------
# PASO 1: StringIndexer
# ------------------------------------------------------------
print("\n[5] PASO 1 – StringIndexer")

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=f"{c}_idx",
        handleInvalid="keep"
    )
    for c in available_cat
]

print("StringIndexers creados:")
for idx in indexers:
    print(f"  - {idx.getInputCol()} → {idx.getOutputCol()}")

# ------------------------------------------------------------
# PASO 2: OneHotEncoder
# ------------------------------------------------------------
print("\n[6] PASO 2 – OneHotEncoder")

encoders = [
    OneHotEncoder(
        inputCol=f"{c}_idx",
        outputCol=f"{c}_vec"
    )
    for c in available_cat
]

print("OneHotEncoders creados:")
for enc in encoders:
    print(f"  - {enc.getInputCol()} → {enc.getOutputCol()}")

# ------------------------------------------------------------
# RETO 3: VectorAssembler
# ------------------------------------------------------------
print("\n[7] RETO 3 – VectorAssembler")

print("""
¿Por qué combinar todas las features en un solo vector?
Porque los modelos de Spark ML esperan una única columna
llamada 'features' como entrada.
""")

feature_cols = available_num + [f"{c}_vec" for c in available_cat]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

print(f"✓ VectorAssembler combinará {len(feature_cols)} features:")
for f in feature_cols:
    print(f"  - {f}")

# ------------------------------------------------------------
# RETO 4: Construcción del Pipeline
# ------------------------------------------------------------
print("\n[8] RETO 4 – Construcción del Pipeline")

print("""
Orden correcto del Pipeline:
1. StringIndexer
2. OneHotEncoder
3. VectorAssembler
""")

pipeline_stages = indexers + encoders + [assembler]
pipeline = Pipeline(stages=pipeline_stages)

print(f"Pipeline construido con {len(pipeline_stages)} stages:")
for i, stage in enumerate(pipeline_stages, 1):
    print(f"  Stage {i}: {type(stage).__name__}")

# ------------------------------------------------------------
# Entrenamiento y Transformación
# ------------------------------------------------------------
print("\n[9] Entrenando pipeline...")
pipeline_model = pipeline.fit(df_clean)
print("✓ Pipeline entrenado correctamente")

df_transformed = pipeline_model.transform(df_clean)
print("✓ Dataset transformado")

sample_features = df_transformed.select("features_raw").first()[0]
print(f"Dimensión final del vector de features: {len(sample_features)}")

# ------------------------------------------------------------
# RETO BONUS 1 – Conteo de Features
# ------------------------------------------------------------
print("\n[10] RETO BONUS – Conteo de Features")

total_cat_features = 0
for c in available_cat:
    n = df_clean.select(c).distinct().count()
    print(f"{c}: {n} categorías únicas")
    total_cat_features += n

total_features = len(available_num) + total_cat_features

print(f"""
Resumen:
- Numéricas: {len(available_num)}
- Categóricas codificadas: {total_cat_features}
- TOTAL esperado: {total_features}
- Vector real: {len(sample_features)}
""")

# ------------------------------------------------------------
# Guardado de artefactos
# ------------------------------------------------------------
print("\n[11] Guardando pipeline y dataset transformado...")

pipeline_path = "/opt/spark-data/processed/feature_pipeline"
pipeline_model.write().overwrite().save(pipeline_path)
print(f"✓ Pipeline guardado en: {pipeline_path}")

output_path = "/opt/spark-data/processed/secop_features.parquet"
df_transformed.write.mode("overwrite").parquet(output_path)
print(f"✓ Dataset transformado guardado en: {output_path}")

# ------------------------------------------------------------
# Reflexiones finales
# ------------------------------------------------------------
print("\n[12] Reflexiones finales")
print("""
1. ¿Por qué usar Pipeline?
   - Garantiza reproducibilidad.
   - Evita data leakage.
   - Unifica entrenamiento e inferencia.

2. ¿Qué pasaría si aplicamos OneHotEncoder antes de StringIndexer?
   - Fallaría, porque OneHotEncoder requiere índices numéricos.

3. ¿Cuándo usar StandardScaler?
   - En modelos sensibles a escala: regresión, SVM, KMeans.

4. ¿Por qué guardar el pipeline?
   - Permite aplicar exactamente las mismas transformaciones
     a datos nuevos sin reentrenar ni reescribir lógica.
""")

print("\n" + "="*70)
print("FIN NOTEBOOK – FEATURE ENGINEERING COMPLETADO")
print("="*70)

spark.stop()
