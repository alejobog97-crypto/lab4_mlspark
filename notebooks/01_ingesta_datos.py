# %% [markdown]
# # Notebook 01: Ingesta de Datos
#
# **Objetivo**: Descargar y cargar datos de contratos públicos desde la API de Datos Abiertos Colombia (SECOP II).
#
# **Dataset**: SECOP II - Contratos Electrónicos
#
# **Fuente**: https://www.datos.gov.co/Gastos-Gubernamentales/SECOP-II-Contratos-Electr-nicos/jbjy-vk9h
#
# ## Actividades:
# 1. Configurar SparkSession con Delta Lake
# 2. Descargar dataset usando API Socrata (opcional) o leer CSV local
# 3. Explorar esquema inicial
# 4. Guardar en formato Parquet optimizado

# %%
# Importar librerías
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month
import os

# %%
# Configurar SparkSession
# Conectamos al cluster Spark Master
spark = SparkSession.builder \
    .appName("SECOP_Ingesta") \
    .master("local[*]") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "1g") \
    .getOrCreate()

print(f"Spark Version: {spark.version}")
print(f"Spark Master: {spark.sparkContext.master}")

# %%
# OPCIÓN 1: Descargar datos desde API de Datos Abiertos Colombia
# Nota: Este dataset puede ser muy grande (varios GB), por lo que limitamos a 100k registros
# Para producción, considera descargar el CSV completo manualmente

# Dataset ID: jbjy-vk9h
# API Endpoint: https://www.datos.gov.co/resource/jbjy-vk9h.json

print("Descargando datos desde API Socrata...")
print("Nota: Limitamos a 100,000 registros para el ejercicio práctico")

# Usando requests para descargar datos (alternativa a sodapy)
import requests
import json

# URL de la API con límite de registros
api_url = "https://www.datos.gov.co/resource/jbjy-vk9h.json?$limit=1000"

response = requests.get(api_url)
data = response.json()

print(f"Registros descargados: {len(data)}")

# Guardar JSON localmente
json_path = "/opt/spark-data/raw/secop_contratos.json"
os.makedirs(os.path.dirname(json_path), exist_ok=True)

with open(json_path, 'w', encoding='utf-8') as f:
    for record in data:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"Datos guardados en: {json_path}")

# %%
# OPCIÓN 2: Leer desde JSON descargado
print("Leyendo datos desde JSON...")
df_raw = spark.read.json(json_path)

print(f"Total de registros: {df_raw.count()}")
print(f"Total de columnas: {len(df_raw.columns)}")

# %%
# Explorar esquema del dataset
print("\n=== ESQUEMA DEL DATASET ===")
df_raw.printSchema()

# %%
# Mostrar primeras filas
print("\n=== PRIMERAS 5 FILAS ===")
df_raw.show(5, truncate=False)

# %%
# Mostrar nombres de columnas
print("\n=== COLUMNAS DISPONIBLES ===")
for col_name in df_raw.columns:
    print(f"- {col_name}")

# %%
# Estadísticas básicas del dataset
print("\n=== INFORMACIÓN DEL DATASET ===")
print(f"Registros totales: {df_raw.count():,}")
print(f"Columnas totales: {len(df_raw.columns)}")

# %%
# Seleccionar columnas clave para el análisis de ML

columnas_clave = [
    "referencia_del_contrato",
    "nit_entidad",
    "nombre_entidad",
    "departamento",
    "ciudad",
    "tipo_de_contrato",
    "valor_del_contrato",
    "fecha_de_firma",
    "plazo",
    "plazo_de_ejec_del_contrato",
    "nombre_del_proveedor",
    "estado_contrato"
]

# Verificar qué columnas existen realmente
columnas_disponibles = [col for col in columnas_clave if col in df_raw.columns]
print(f"\n=== COLUMNAS SELECCIONADAS ({len(columnas_disponibles)}) ===")
for col in columnas_disponibles:
    print(f"- {col}")

# %%
# Filtrar columnas disponibles
if columnas_disponibles:
    df_clean = df_raw.select(*columnas_disponibles)
else:
    # Si no encontramos las columnas esperadas, usamos todas
    print("ADVERTENCIA: No se encontraron las columnas esperadas. Usando todas las columnas.")
    df_clean = df_raw

# %%
# Guardar en formato Parquet optimizado
output_path = "/opt/spark-data/raw/secop_contratos.parquet"
print(f"\n=== GUARDANDO EN FORMATO PARQUET ===")
print(f"Ruta: {output_path}")

df_clean.write \
    .mode("overwrite") \
    .parquet(output_path)

print("Datos guardados exitosamente en formato Parquet")

# %%
# Verificar que el archivo se guardó correctamente
print("\n=== VERIFICACIÓN ===")
df_verificacion = spark.read.parquet(output_path)
print(f"Registros en Parquet: {df_verificacion.count():,}")
print(f"Columnas en Parquet: {len(df_verificacion.columns)}")

# %%
# Resumen final
print("\n" + "="*60)
print("RESUMEN DE INGESTA")
print("="*60)
print(f"✓ Datos descargados desde API Socrata")
print(f"✓ Registros procesados: {df_clean.count():,}")
print(f"✓ Formato de salida: Parquet")
print(f"✓ Ubicación: {output_path}")
print("="*60)

# %%
# Detener SparkSession
spark.stop()
print("SparkSession finalizada")