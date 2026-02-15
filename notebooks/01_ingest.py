# =========================================
# 01_ingest_bronze.py
# Ingesta SECOP → Delta Bronze
# =========================================

from pyspark.sql import SparkSession
import os
import json
import re
from sodapy import Socrata
from delta import configure_spark_with_delta_pip

builder = (
    SparkSession.builder
    .appName("SECOP_Lakehouse")
    .master("spark://spark-master:7077")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark = configure_spark_with_delta_pip(builder).getOrCreate()
# -----------------------------------------
# Spark Session
# -----------------------------------------


print(f"Spark Version: {spark.version}")
print(f"Spark Master: {spark.sparkContext.master}")

# -----------------------------------------
# Paths
# -----------------------------------------
RAW_PATH = "/app/data/raw"
BRONZE_PATH = "/app/data/lakehouse/bronze/secop"
JSON_PATH = f"{RAW_PATH}/secop_contratos.json"

os.makedirs(RAW_PATH, exist_ok=True)

# -----------------------------------------
# Extracción desde Socrata (SECOP)
# -----------------------------------------
client = Socrata("www.datos.gov.co", None)

query = """
SELECT *
WHERE 
    departamento = "Distrito Capital de Bogotá"
AND
    fecha_de_firma > '2025-09-30T23:59:59'
AND 
    fecha_de_firma < '2026-01-01T00:00:00'
LIMIT 200000
"""

results = client.get("jbjy-vk9h", query=query)
print(f"Registros extraídos: {len(results)}")

# -----------------------------------------
# Guardar JSON línea a línea (raw)
# -----------------------------------------
with open(JSON_PATH, "w", encoding="utf-8") as f:
    for record in results:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Archivo RAW guardado en: {JSON_PATH}")

# -----------------------------------------
# Lectura Spark JSON
# -----------------------------------------
df_raw = spark.read.json(JSON_PATH)

print(f"Registros leídos: {df_raw.count()}")
print(f"Columnas: {len(df_raw.columns)}")

# -----------------------------------------
# Normalización de nombres de columnas
# (solo nombres, NO tipos)
# -----------------------------------------
def normalize_columns(df):
    for c in df.columns:
        new_c = re.sub(r"[ ,;{}()\n\t=]", "_", c.strip().lower())
        df = df.withColumnRenamed(c, new_c)
    return df

df_bronze = normalize_columns(df_raw)

# -----------------------------------------
# Escritura Delta Bronze
# -----------------------------------------
print("Escribiendo capa Bronze...")

df_bronze.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(BRONZE_PATH)

print("Ingesta Bronze completada correctamente ✅")
