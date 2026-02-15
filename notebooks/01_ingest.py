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

# -----------------------------------------
# Spark Session
# -----------------------------------------

builder = (
    SparkSession.builder
    .appName("SECOP_Lakehouse")
    .master("spark://spark-master:7077")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark = configure_spark_with_delta_pip(builder).getOrCreate()

print(f"Spark Version: {spark.version}")
print(f"Spark Master: {spark.sparkContext.master}")

# -----------------------------------------
# Paths
# -----------------------------------------
RAW_PATH = "/opt/spark-data/raw"
BRONZE_PATH = "/opt/spark-data/lakehouse/bronze/secop"
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
# -----------------------------------------
def normalize_columns(df):
    for c in df.columns:
        new_c = re.sub(r"[ ,;{}()\n\t=]", "_", c.strip().lower())
        df = df.withColumnRenamed(c, new_c)
    return df

df_bronze = normalize_columns(df_raw)

# -----------------------------------------
# Selección de columnas (MISMA SALIDA, MENOS RUIDO)
# -----------------------------------------
columnas_bronze = [
    # CORE (contrato / entidad)
    # ---------------------------
    "id_contrato",
    "referencia_del_contrato",
    "proceso_de_compra",
    "nombre_entidad",
    "nit_entidad",
    "sector",
    "rama",
    "departamento",
    "ciudad",
    "modalidad_de_contratacion",
    "tipo_de_contrato",
    "objeto_del_contrato",
    "estado_contrato",
    "fecha_de_firma",
    "fecha_de_inicio_del_contrato",
    "fecha_de_fin_del_contrato",
    "duraci_n_del_contrato",
    "proveedor_adjudicado",
    "tipodocproveedor",
    "codigo_proveedor",

    # ---------------------------
    # FINANCIERAS
    # ---------------------------
    "valor_del_contrato",
    "valor_pagado",
    "valor_pendiente_de_pago",
    "valor_pendiente_de_ejecucion",
    "valor_facturado",
    "valor_amortizado",
    "valor_de_pago_adelantado",
    "saldo_cdp",
    "saldo_vigencia",
    "presupuesto_general_de_la_nacion_pgn",
    "recursos_propios",
    "recursos_de_credito",
    "sistema_general_de_regal_as",
    "sistema_general_de_participaciones",
    "habilita_pago_adelantado",

    # ---------------------------
    # CONTEXTO / FEATURE ENGINEERING
    # ---------------------------
    "destino_gasto",
    "origen_de_los_recursos",
    "es_pyme",
    "es_grupo",
    "entidad_centralizada",
    "espostconflicto",
    "pilares_del_acuerdo",
    "puntos_del_acuerdo",
    "anno_bpin",
    "codigo_de_categoria_principal",
    "codigo_entidad",
    "localizaci_n"
]


# Mantener solo columnas existentes (robusto a cambios de esquema)
columnas_bronze = [c for c in columnas_bronze if c in df_bronze.columns]

df_bronze = df_bronze.select(*columnas_bronze)

# -----------------------------------------
# Escritura Delta Bronze (MISMA SALIDA)
# -----------------------------------------
print("Escribiendo capa Bronze...")

df_bronze.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(BRONZE_PATH)

print("Ingesta Bronze completada correctamente ✅")
