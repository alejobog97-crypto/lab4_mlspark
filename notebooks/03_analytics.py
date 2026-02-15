# %% [markdown]
# # 3. Analítica de Negocio (Capa Oro)
# Agregaciones estratégicas para toma de decisiones

# %%
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, when, col
from pyspark.sql.functions import sum as sum_, avg, count, expr, desc
from pyspark.sql.functions import year, month
from delta import *

# %%
builder = SparkSession.builder \
    .appName("Lab_SECOP_Gold") \
    .master("spark://spark-master:7077") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# %%
# Leer capa Plata (Silver)
df_silver = spark.read.format("delta").load(
    "/app/data/lakehouse/silver/secop"
)

# %%
#columnas numéricas clave: valor_del_contrato;valor_pagado;saldo_cdp

#Completitud de los datos  
total_rows = df_silver.count()

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

print("Datos Faltantes Variables Númericas")
faltantes.show(truncate=False)

#Estadisticas Basicas
stats_basicas = df_silver.select(
    "valor_del_contrato",
    "valor_pagado",
    "saldo_cdp"
).describe()

print("Estadísticas Basicas")
stats_basicas.show(truncate=False)

#Medianas y Percentiles
percentiles = df_silver.select(
    col("valor_del_contrato")
).approxQuantile(
    "valor_del_contrato",
    [0.25, 0.5, 0.75],
    0.01
)

print(f"P25: {percentiles[0]}")
print(f"Mediana (P50): {percentiles[1]}")
print(f"P75: {percentiles[2]}")

#Actividad 3. Distribución por departamento
dist_entidades = (
    df_silver
    .groupBy("nombre_entidad")
    .agg(
        count("*").alias("num_contratos"),
        sum_("valor_del_contrato").alias("total_contratado")
    )
    .orderBy(desc("total_contratado"))
)
print("Distribución por Entidades")
dist_entidades.show(truncate=False)


#Outliers
q1, q3 = df_silver.approxQuantile(
    "valor_del_contrato",
    [0.25, 0.75],
    0.01
)

iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(f"Límite inferior: {lower_bound}")
print(f"Límite superior: {upper_bound}")

outliers = df_silver.filter(
    (col("valor_del_contrato") < lower_bound) |
    (col("valor_del_contrato") > upper_bound)
)

print(f"Outliers detectados: {outliers.count()}")
outliers.select(
    "nombre_entidad",
    "departamento",
    "valor_del_contrato"
).orderBy(desc("valor_del_contrato")).show(truncate=False)


# Agregación de negocio (Top 10 departamentos por inversión)

df_gold = (
    df_silver
    .groupBy("nombre_entidad")
    .agg(
        sum_("valor_del_contrato").alias("total_contratado")
    )
    .orderBy(desc("total_contratado"))
    .limit(10)
)

df_gold.show(truncate=False)

# Base Final
df_gold_base = (
    df_silver
    .select(
        # Dimensiones
        col("nombre_entidad"),
        col("departamento"),
        col("ciudad"),

        # Tiempo
        col("fecha_de_firma"),
        year(col("fecha_de_firma")).alias("anio"),
        month(col("fecha_de_firma")).alias("mes"),

        # Métricas financieras
        col("valor_del_contrato"),
        col("valor_pagado"),
        col("saldo_cdp"),
        col("valor_pendiente_de_pago"),
        col("valor_amortizado")
    )
)

df_gold_kpis = (
    df_gold_base
    .groupBy("departamento", "anio", "mes")
    .agg(
        count("*").alias("num_contratos"),
        sum_("valor_del_contrato").alias("total_contratado"),
        sum_("valor_pagado").alias("total_pagado"),
        sum_("saldo_cdp").alias("saldo_total"),
        avg("valor_del_contrato").alias("promedio_contrato"),
        expr("percentile_approx(valor_del_contrato, 0.5)").alias("mediana_contrato")
    )
)
print("Summary")
df_gold_kpis.show(20, truncate=False)

# %%
# Persistir capa Oro
df_gold.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save("/app/data/lakehouse/gold/top_deptos")

# %%
# Visualizar resultados
print("🏆 Top 10 Entidades por contratación pública:")
df_gold.show(truncate=False)
