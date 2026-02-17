# ============================================================
# NOTEBOOK 03: Analytics
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import (col, count, sum as spark_sum, avg, min as spark_min, max as spark_max, stddev, isnan, when, isnull, desc)
import os
import json
import re
from sodapy import Socrata
from delta import configure_spark_with_delta_pip

# ============================================================
# 1. INICIALIZACIÓN DE SPARK
# ============================================================

print("\n[1] Inicializando Spark para análisis analítico...")

builder = (
    SparkSession.builder
    .appName("SECOP_Analytics")
    .master("spark://spark-master:7077")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()

print("Spark inicializado correctamente")
print(f"  - Spark Version : {spark.version}")
print(f"  - Spark Master  : {spark.sparkContext.master}")


# ============================================================
# 2. LECTURA DE CAPA SILVER
# ============================================================

df_silver = spark.read.parquet("/opt/spark-data/processed/secop_eda.parquet")
print(f"Registros cargados: {df.count():,}")


total_rows = df_silver.count()

print(f"Registros cargados desde Silver: {total_rows:,}")
print(f"Columnas disponibles: {len(df_silver.columns)}")


# ============================================================
# 3. CALIDAD DE DATOS – COMPLETITUD
# ============================================================

print("\n[3] Evaluando completitud de variables clave...")

faltantes = df_silver.select(
    (count(when(col("valor_del_contrato").isNull(), 1)) / total_rows)
        .alias("valor_del_contrato_pct_null"),
    (count(when(col("valor_pagado").isNull(), 1)) / total_rows)
        .alias("valor_pagado_pct_null"),
    (count(when(col("saldo_cdp").isNull(), 1)) / total_rows)
        .alias("saldo_cdp_pct_null"),
    (count(when(col("fecha_de_firma").isNull(), 1)) / total_rows)
        .alias("fecha_firma_pct_null")
)

print("Porcentaje de valores faltantes:")
faltantes.show(truncate=False)


# ============================================================
# 4. ESTADÍSTICAS DESCRIPTIVAS BÁSICAS
# ============================================================

print("\n[4] Calculando estadísticas descriptivas básicas...")

stats_basicas = df_silver.select(
    "valor_del_contrato",
    "valor_pagado",
    "saldo_cdp"
).describe()

stats_basicas.show(truncate=False)


# ============================================================
# 5. MEDIANAS Y PERCENTILES
# ============================================================

print("\n[5] Calculando percentiles del valor del contrato...")

percentiles = df_silver.approxQuantile(
    "valor_del_contrato",
    [0.25, 0.5, 0.75],
    0.01
)

print(f"  - P25      : {percentiles[0]:,.2f}")
print(f"  - Mediana  : {percentiles[1]:,.2f}")
print(f"  - P75      : {percentiles[2]:,.2f}")


# ============================================================
# 6. DISTRIBUCIÓN POR ENTIDAD CONTRATANTE
# ============================================================

print("\n[6] Analizando distribución de contratos por entidad...")

dist_entidades = (
    df_silver
    .groupBy("nombre_entidad")
    .agg(
        count("*").alias("num_contratos"),
        spark_sum("valor_del_contrato").alias("total_contratado")
    )
    .orderBy(desc("total_contratado"))
)

dist_entidades.show(truncate=False)


# ============================================================
# 7. DETECCIÓN DE OUTLIERS (IQR)
# ============================================================

print("\n[7] Detectando outliers en valor_del_contrato (método IQR)...")

q1, q3 = df_silver.approxQuantile(
    "valor_del_contrato",
    [0.25, 0.75],
    0.01
)

iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(f"  - Límite inferior: {lower_bound:,.2f}")
print(f"  - Límite superior: {upper_bound:,.2f}")

outliers = df_silver.filter(
    (col("valor_del_contrato") < lower_bound) |
    (col("valor_del_contrato") > upper_bound)
)

print(f"Outliers detectados: {outliers.count():,}")

outliers.select(
    "nombre_entidad",
    "departamento",
    "valor_del_contrato"
).orderBy(desc("valor_del_contrato")).show(truncate=False)


# ============================================================
# 8. AGREGACIÓN DE NEGOCIO – TOP ENTIDADES
# ============================================================

print("\n[8] Construyendo agregación de negocio (Top 10 entidades)...")

df_gold = (
    df_silver
    .groupBy("nombre_entidad")
    .agg(
        spark_sum("valor_del_contrato").alias("total_contratado")
    )
    .orderBy(desc("total_contratado"))
    .limit(10)
)

df_gold.show(truncate=False)


# ============================================================
# 9. BASE ANALÍTICA TEMPORAL (KPIs)
# ============================================================

print("\n[9] Construyendo base analítica temporal (KPIs mensuales)...")

df_gold_base = df_silver.select(
    # Dimensiones
    col("nombre_entidad"),
    col("departamento"),
    col("ciudad"),

    # Tiempo
    col("fecha_de_firma"),
    year(col("fecha_de_firma")).alias("anio"),
    month(col("fecha_de_firma")).alias("mes"),

    # Métricas
    col("valor_del_contrato"),
    col("valor_pagado"),
    col("saldo_cdp"),
    col("valor_pendiente_de_pago"),
    col("valor_amortizado")
)

df_gold_kpis = (
    df_gold_base
    .groupBy("departamento", "anio", "mes")
    .agg(
        count("*").alias("num_contratos"),
        spark_sum("valor_del_contrato").alias("total_contratado"),
        spark_sum("valor_pagado").alias("total_pagado"),
        spark_sum("saldo_cdp").alias("saldo_total"),
        avg("valor_del_contrato").alias("promedio_contrato"),
        expr("percentile_approx(valor_del_contrato, 0.5)").alias("mediana_contrato")
    )
)

print("Resumen KPI mensual:")
df_gold_kpis.show(20, truncate=False)


# ============================================================
# 10. PERSISTENCIA CAPA GOLD
# ============================================================

print("\n[10] Persistiendo capa Gold en Delta Lake...")

GOLD_PATH = "/app/data/lakehouse/gold/top_entidades"

df_gold.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(GOLD_PATH)

print(f"Capa Gold guardada correctamente en: {GOLD_PATH}")


# ============================================================
# 11. VISUALIZACIÓN FINAL
# ============================================================

print("\n🏆 Top 10 Entidades por contratación pública:")
df_gold.show(truncate=False)

print("\nProceso de Analytics finalizado correctamente ✅")
print("="*70)