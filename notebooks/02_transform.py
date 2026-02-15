# =========================================
# 02_transformación_Silver.py
# Ingesta SECOP → Delta Bronze
# =========================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, regexp_replace,
    to_timestamp, concat_ws, current_timestamp
)
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DecimalType,
    LongType, TimestampType
)
from delta import *

# -----------------------------------------
# Spark Session
# -----------------------------------------

builder = SparkSession.builder \
    .appName("SECOP_Silver_Quality_Gate") \
    .master("spark://spark-master:7077") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

spark.conf.set("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")
spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")

# -----------------------------------------
# Leer Bronce
# -----------------------------------------
df_bronze = spark.read.format("delta") \
    .load("/app/data/lakehouse/bronze/secop")

# -----------------------------------------
# Esquema
# -----------------------------------------
silver_schema = StructType([
    StructField("nombre_entidad", StringType(), True),
    StructField("departamento", StringType(), True),
    StructField("ciudad", StringType(), True),

    # Montos
    StructField("valor_amortizado", DecimalType(18, 2), True),
    StructField("valor_de_pago_adelantado", DecimalType(18, 2), True),
    StructField("valor_del_contrato", DecimalType(18, 2), True),
    StructField("valor_facturado", DecimalType(18, 2), True),
    StructField("valor_pagado", DecimalType(18, 2), True),
    StructField("valor_pendiente_de", DecimalType(18, 2), True),
    StructField("valor_pendiente_de_ejecucion", DecimalType(18, 2), True),
    StructField("valor_pendiente_de_pago", DecimalType(18, 2), True),
    StructField("saldo_cdp", DecimalType(18, 2), True),
    StructField("saldo_vigencia", DecimalType(18, 2), True),

    # Otros numéricos
    StructField("dias_adicionados", LongType(), True),
    StructField("presupuesto_general_de_la_nacion_pgn", DecimalType(18, 2), True),
    StructField("sistema_general_de_participaciones", DecimalType(18, 2), True),
    StructField("sistema_general_de_regal_as", DecimalType(18, 2), True),
    StructField("recursos_de_credito", DecimalType(18, 2), True),
    StructField("recursos_propios", DecimalType(18, 2), True),

    StructField("codigo_entidad", LongType(), True),
    StructField("nit_entidad", LongType(), True),

    # Fechas
    StructField("fecha_de_firma", TimestampType(), True),
    StructField("fecha_de_inicio_del_contrato", TimestampType(), True),
    StructField("fecha_de_fin_del_contrato", TimestampType(), True),
    StructField("fecha_de_notificaci_n_de_prorrogaci_n", TimestampType(), True),
    StructField("ultima_actualizacion", TimestampType(), True),
    StructField("fecha_inicio_liquidacion", TimestampType(), True),
    StructField("fecha_fin_liquidacion", TimestampType(), True),
])

# -----------------------------------------
# Ajuste Esquema
# -----------------------------------------
decimal_cols = [
    "valor_amortizado",
    "valor_de_pago_adelantado",
    "valor_del_contrato",
    "valor_facturado",
    "valor_pagado",
    "valor_pendiente_de",
    "valor_pendiente_de_ejecucion",
    "valor_pendiente_de_pago",
    "saldo_cdp",
    "saldo_vigencia",
    "presupuesto_general_de_la_nacion_pgn",
    "sistema_general_de_participaciones",
    "sistema_general_de_regal_as",
    "recursos_de_credito",
    "recursos_propios"
]

long_cols = [
    "dias_adicionados",
    "codigo_entidad",
    "nit_entidad"
]

df_silver = df_bronze

# Decimales
for c in decimal_cols:
    df_silver = df_silver.withColumn(
        c,
        when(col(c).isNull() | (col(c) == ""), None)
        .otherwise(regexp_replace(col(c), ",", "").cast(DecimalType(18, 2)))
    )

# Long
for c in long_cols:
    df_silver = df_silver.withColumn(
        c,
        when(col(c).isNull() | (col(c) == ""), None)
        .otherwise(col(c).cast(LongType()))
    )

timestamp_cols = [
    "fecha_de_firma",
    "fecha_de_inicio_del_contrato",
    "fecha_de_fin_del_contrato",
    "fecha_de_notificaci_n_de_prorrogaci_n",
    "ultima_actualizacion",
    "fecha_inicio_liquidacion",
    "fecha_fin_liquidacion"
]

for c in timestamp_cols:
    df_silver = df_silver.withColumn(
        c,
        when(col(c).isNull() | (col(c) == ""), None)
        .otherwise(to_timestamp(col(c)))
    )

df_silver_base = df_silver.select([f.name for f in silver_schema.fields])

# -----------------------------------------
# Quality Gate
# -----------------------------------------
df_validated = df_silver_base.withColumn(
    "motivo_rechazo",
    concat_ws(", ",
        when(col("valor_del_contrato") <= 0, "Valor del contrato inválido"),
        when(col("fecha_de_firma").isNull(), "Fecha de firma nula")
    )
)

df_validated = df_validated.withColumn(
    "es_valido",
    col("motivo_rechazo") == ""
)

df_silver = (
    df_validated
    .filter(col("es_valido"))
    .drop("motivo_rechazo", "es_valido")
)

df_quarantine = (
    df_validated
    .filter(~col("es_valido"))
    .withColumn("fecha_cuarentena", current_timestamp())
)

# -----------------------------------------
# Escritura
# -----------------------------------------

df_silver.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save("/app/data/lakehouse/silver/secop")

df_quarantine.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save("/app/data/lakehouse/quarantine/secop_errors")


# -----------------------------------------
# Resultados
# -----------------------------------------
print(f"✅ Registros Silver: {df_silver.count()}")
print(f"❌ Registros en cuarentena: {df_quarantine.count()}")

df_silver.show(5, truncate=False)
df_quarantine.show(5, truncate=False)