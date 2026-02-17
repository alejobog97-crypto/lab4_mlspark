# %% [markdown]
# # Notebook 12: Inferencia en Producción
#
# **Sección 16 - MLOps**: Despliegue y predicciones batch
#
# **Objetivo**: Simular un pipeline de producción para generar predicciones
#
# ## Conceptos clave:
# - **Batch Inference**: Predicciones sobre grandes volúmenes de datos
# - **Model Loading**: Cargar modelo desde MLflow Registry
# - **Monitoring**: Verificar calidad de predicciones
# - **Output Formats**: Guardar resultados para consumo (Parquet, CSV)
#
# ## Actividades:
# 1. Cargar modelo desde Model Registry (Production)
# 2. Aplicar transformaciones del pipeline
# 3. Generar predicciones batch sobre datos nuevos
# 4. Monitorear y guardar resultados

# ============================================================
# NOTEBOOK 12: INFERENCIA EN PRODUCCIÓN
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, current_timestamp, min as spark_min, max as spark_max,
    avg, stddev, count, when, log1p, exp
)
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from datetime import datetime

# ============================================================
# INICIAR SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("SECOP_Produccion") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

print("Spark inicializado correctamente")

# ============================================================
# RETO 1: CARGAR MODELO EN PRODUCCIÓN
# ============================================================

print("\n" + "="*60)
print("RETO 1: CARGAR MODELO EN PRODUCCIÓN")
print("="*60)

# Configurar MLflow
mlflow.set_tracking_uri("http://mlflow:5000")
client = MlflowClient()

# Definir nombre del modelo
model_name = "secop_prediccion_contratos"
model_uri = f"models:/{model_name}/Production"

print(f"\nCargando modelo desde: {model_uri}")

# Intentar cargar modelo con fallback
try:
    production_model = mlflow.spark.load_model(model_uri)
    print(f"Modelo cargado exitosamente por stage")
    print(f"Tipo: {type(production_model)}")
    carga_exitosa = True
except Exception as e:
    print(f"NOTA: No se pudo cargar por stage: {type(e).__name__}")
    print("Intentando método alternativo por run_id...")
    
    try:
        # Obtener run_id del modelo en Production
        production_versions = client.get_latest_versions(model_name, stages=["Production"])
        if production_versions:
            run_id = production_versions[0].run_id
            model_uri_alt = f"runs:/{run_id}/model"
            
            print(f"Cargando desde: {model_uri_alt}")
            production_model = mlflow.spark.load_model(model_uri_alt)
            print(f"Modelo cargado exitosamente por run_id")
            print(f"Tipo: {type(production_model)}")
            carga_exitosa = True
        else:
            print("No se encontró modelo en Production")
            carga_exitosa = False
    except Exception as e2:
        print(f"NOTA: Error en método alternativo: {type(e2).__name__}")
        print("\nUsaremos simulación para demostrar el flujo de producción")
        carga_exitosa = False

print("\n¿Por qué cargar el modelo desde el Registry y no desde una ruta fija?")

print("""
En producción no queremos depender de archivos estáticos como 
/models/v2/model.pkl. Eso nos obliga a cambiar código cada vez 
que el modelo cambia.

El Model Registry nos permite desacoplar el código del modelo.

¿Qué significa eso en la práctica?

• Siempre se carga el modelo que esté marcado como "Production".
  No importa si es v2, v5 o v10. El código no cambia.

• Si algo sale mal, se puede hacer rollback simplemente 
  cambiando el stage en MLflow. No hay que redeplegar nada.

• Queda trazabilidad: quién promovió el modelo, cuándo y por qué.
  Esto es clave en entornos regulados o corporativos.

• El mismo código funciona en distintos ambientes:
  - Dev puede apuntar a Staging
  - Prod apunta a Production
  Solo cambia el stage, no la lógica.

Comparación rápida:

Ruta fija:
    load_model("/models/v2/model.pkl")
    → versión hardcodeada
    → difícil hacer rollback
    → requiere redeploy

Model Registry:
    mlflow.spark.load_model("models:/nombre/Production")
    → siempre toma la versión activa
    → rollback inmediato
    → más limpio y mantenible

¿Qué pasa si no hay modelo en Production?

Puede ocurrir en el primer despliegue o si alguien archivó la versión por error.
En ese caso, conviene manejarlo explícitamente.
""")


# ============================================================
# RETO 2: CARGAR Y PREPARAR DATOS NUEVOS
# ============================================================

print("\n" + "="*60)
print("RETO 2: CARGAR Y PREPARAR DATOS NUEVOS")
print("="*60)

# Cargar datos para predicción
df_new = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")

# Renombrar para coincidir con modelo
df_new = df_new.withColumnRenamed("features_pca", "features")

# Guardar label real para comparación posterior (simulamos que no la conocemos aún)
df_actual = df_new.select("label").withColumn("row_id", col("label"))

# Simular datos nuevos (sin label)
df_new_no_label = df_new.drop("label")

print(f"\nContratos para predecir: {df_new_no_label.count():,}")
print(f"Columnas disponibles: {df_new_no_label.columns}")

print("\nEn un sistema real, ¿de dónde vendrían los datos nuevos?")

print("""
En producción los datos no aparecen en un parquet mágico.
Siempre vienen de algún sistema operativo o analítico.

Algunas fuentes típicas:

1) Base de datos transaccional  
   Lo más común es leer directamente desde PostgreSQL, MySQL u Oracle.
   Por ejemplo, contratos que se firmaron en las últimas 24 horas.

   df_new = spark.read \
       .format("jdbc") \
       .option("url", "jdbc:postgresql://db:5432/secop") \
       .option("dbtable", "contratos_pendientes") \
       .option("user", "readonly") \
       .option("password", "***") \
       .load()

   Esto normalmente se ejecuta en batch diario.

2) Archivos en S3 o Azure Blob  
   Muchos sistemas exportan datos como CSV o Parquet y los dejan en un bucket.
   El pipeline simplemente procesa la carpeta del día.

   df_new = spark.read.parquet("s3://datos-secop/nuevos/2025-02-14/")

   Es un patrón muy común en arquitecturas data lake.

3) Streaming (Kafka, Kinesis)  
   Si los eventos llegan en tiempo real, se usa Spark Structured Streaming.
   Por ejemplo, cada vez que se registra un contrato nuevo.

   df_stream = spark.readStream \
       .format("kafka") \
       .option("kafka.bootstrap.servers", "kafka:9092") \
       .option("subscribe", "contratos_nuevos") \
       .load()

   Aquí ya estamos hablando de inferencia casi en tiempo real.

4) API REST  
   Para predicción individual (por ejemplo, desde un portal web),
   el backend recibe los datos, los convierte en DataFrame y llama al modelo.

   @app.post("/predict")
   def predict(contrato: ContratoInput):
       df = create_spark_df(contrato)
       prediction = model.transform(df)
       return prediction.toJSON()

   Esto es típico cuando el modelo se expone como microservicio.

5) Data Warehouse  
   También es común leer desde Snowflake, BigQuery o Redshift,
   especialmente para procesos batch sobre tablas consolidadas.

   df_new = spark.read \
       .format("snowflake") \
       .options(**snowflake_options) \
       .option("dbtable", "analytics.contratos_activos") \
       .load()


En el caso de SECOP, lo más razonable sería:
- Consumir la API de Datos Abiertos
- Ejecutar un batch diario
- Procesar los contratos nuevos cada noche
- Guardar predicciones para análisis o monitoreo

La fuente depende del nivel de madurez del sistema,
pero el patrón general es siempre el mismo:
extraer → transformar → predecir → almacenar resultados.
""")


# ============================================================
# RETO 3: GENERAR PREDICCIONES BATCH
# ============================================================

print("\n" + "="*60)
print("RETO 3: GENERAR PREDICCIONES BATCH")
print("="*60)

if carga_exitosa:
    print("\nGenerando predicciones con modelo de Production...")
    predictions_batch = production_model.transform(df_new_no_label)
    prediccion_real = True
else:
    print("\nSIMULACIÓN: Generando predicciones sintéticas...")
    print("(En producción real, esto vendría del modelo)")
    
    # Simular predicciones basadas en patrones realistas
    from pyspark.sql.functions import rand, round as spark_round
    predictions_batch = df_new_no_label.withColumn(
        "prediction",
        spark_round(10 + rand() * 3, 4)  # Simula predicciones en escala log
    )
    prediccion_real = False

# Agregar timestamp y metadata
predictions_batch = predictions_batch.withColumn(
    "prediction_timestamp", 
    current_timestamp()
).withColumn(
    "model_version",
    col("prediction") * 0 + 2  # Versión 2 del modelo
).withColumn(
    "prediction_id",
    col("prediction") * 0  # Se reemplazará con ID único
)

print(f"\nPredicciones generadas: {predictions_batch.count():,}")
print("\nPrimeras predicciones:")
predictions_batch.select(
    "prediction", 
    "prediction_timestamp", 
    "model_version"
).show(10, truncate=False)

print("\n¿Por qué agregar timestamp a las predicciones?")

print("""
En desarrollo una predicción es solo un número.
En producción, sin contexto, ese número no sirve de mucho.

El timestamp permite saber exactamente cuándo se generó la predicción.
Si más adelante alguien pregunta por qué un contrato fue estimado en cierto valor,
podemos revisar qué modelo estaba activo en ese momento y bajo qué condiciones.

También es clave para monitoreo.  
Si con el tiempo empezamos a notar que las predicciones cambian de patrón,
el timestamp nos permite analizar ese comportamiento por día, semana o mes
y detectar posibles problemas como drift o cambios en los datos de entrada.

Desde el punto de vista operativo, ayuda en varios frentes:
- auditoría (qué modelo tomó la decisión y cuándo),
- debugging (reproducir una predicción específica),
- análisis de desempeño (tiempos de respuesta, comportamiento por batch).

Además del timestamp, normalmente se guarda algo de metadata adicional.

Del modelo:
- versión utilizada
- tipo de modelo
- fecha de entrenamiento
- métrica de referencia al momento de entrenar

De los datos:
- fuente
- versión del pipeline de preprocesamiento
- algún indicador de calidad si existe

De la predicción:
- identificador único (prediction_id)
- batch_id si fue procesamiento masivo
- latencia de inferencia
- score de confianza (si aplica)

Y cuando ya conocemos el valor real, podemos cerrar el ciclo:
- actual_value
- error
- fecha en que se validó

Un ejemplo sencillo en Spark sería:

predictions = predictions \
    .withColumn("prediction_id", uuid_udf()) \
    .withColumn("prediction_timestamp", current_timestamp()) \
    .withColumn("model_version", lit(MODEL_VERSION)) \
    .withColumn("model_name", lit(MODEL_NAME)) \
    .withColumn("batch_id", lit(BATCH_ID)) \
    .withColumn("data_source", lit("API_SECOP"))

Con esto después podemos:
- calcular error por versión de modelo
- comparar batches
- analizar comportamiento en el tiempo
- alimentar dashboards de monitoreo

En resumen, el timestamp no es un detalle técnico menor.
Es lo que convierte una predicción aislada en un evento trazable.
""")


# ============================================================
# RETO 4: MONITOREO DE PREDICCIONES
# ============================================================

print("\n" + "="*60)
print("RETO 4: MONITOREO DE PREDICCIONES")
print("="*60)

# Calcular estadísticas de predicciones
stats = predictions_batch.select(
    spark_min("prediction").alias("min_pred"),
    spark_max("prediction").alias("max_pred"),
    avg("prediction").alias("avg_pred"),
    stddev("prediction").alias("std_pred"),
    count("*").alias("total")
).collect()[0]

print("\n=== ESTADÍSTICAS DE PREDICCIONES (ESCALA LOG) ===")
print(f"Total: {stats['total']:,}")
print(f"Mínimo: {stats['min_pred']:.4f}")
print(f"Máximo: {stats['max_pred']:.4f}")
print(f"Promedio: {stats['avg_pred']:.4f}")
print(f"Desviación Estándar: {stats['std_pred']:.4f}")

# Convertir a escala original (des-log)
import math

print("\n=== ESTADÍSTICAS EN ESCALA ORIGINAL (COP) ===")
print(f"Mínimo: ${math.exp(stats['min_pred']):,.0f}")
print(f"Máximo: ${math.exp(stats['max_pred']):,.0f}")
print(f"Promedio: ${math.exp(stats['avg_pred']):,.0f}")

# Analizar distribución por rangos
print("\n=== DISTRIBUCIÓN POR RANGOS DE VALOR ===")
prediction_ranges = predictions_batch.select(
    count(when(exp(col("prediction")) < 10000000, True)).alias("< 10M"),
    count(when((exp(col("prediction")) >= 10000000) & (exp(col("prediction")) < 100000000), True)).alias("10M-100M"),
    count(when((exp(col("prediction")) >= 100000000) & (exp(col("prediction")) < 1000000000), True)).alias("100M-1B"),
    count(when(exp(col("prediction")) >= 1000000000, True)).alias("> 1B")
)
prediction_ranges.show()

# Identificar anomalías
anomalias_negativas = predictions_batch.filter(col("prediction") < 0).count()
anomalias_extremas = predictions_batch.filter(exp(col("prediction")) > 10000000000).count()

print(f"\n=== DETECCIÓN DE ANOMALÍAS ===")
print(f"Predicciones negativas (error): {anomalias_negativas}")
print(f"Predicciones extremas (> $10B): {anomalias_extremas}")

# Alertas
print("\n=== SISTEMA DE ALERTAS ===")
alerta_activada = False

if anomalias_negativas > 0:
    print(f" ALERTA CRÍTICA: {anomalias_negativas} predicciones negativas detectadas")
    alerta_activada = True

if anomalias_extremas > stats['total'] * 0.05:
    print(f" ALERTA: Más del 5% de predicciones son extremadamente altas")
    alerta_activada = True

if stats['std_pred'] > 2.0:
    print(f" ADVERTENCIA: Alta varianza en predicciones (std={stats['std_pred']:.2f})")

if not alerta_activada:
    print(" OK: Predicciones dentro de rangos esperados")

print("\n¿Cómo detectarías data drift?")

print("""
Data drift significa que lo que está llegando hoy no se parece a lo que
usamos para entrenar el modelo.

Puede pasar por varias razones.

A veces cambia la relación entre las variables y el target
(concept drift). Por ejemplo, si hay inflación fuerte,
la relación histórica entre tipo de contrato y valor puede dejar de ser válida.
Normalmente lo detectamos porque el error empieza a subir con el tiempo.

Otras veces no cambia la relación, sino la distribución de los datos de entrada.
Empiezan a aparecer tipos de contratos nuevos o rangos de valores que
no estaban en el entrenamiento. Eso es data drift en sentido estricto.

También puede cambiar la distribución del target.
Por ejemplo, si ahora hay muchos más contratos grandes que antes.

¿Cómo lo detectamos en la práctica?

Una forma es estadística.
Comparar la distribución de entrenamiento contra la de producción.

Por ejemplo, con un test de Kolmogorov-Smirnov:

from scipy import stats

ks_stat, p_value = stats.ks_2samp(
    train_predictions,
    prod_predictions
)

if p_value < 0.05:
    print("Posible drift: las distribuciones son diferentes")

Otra métrica común es el PSI (Population Stability Index).
Si supera cierto umbral (por ejemplo 0.2), ya es una señal fuerte.

También se puede monitorear algo más simple:
media, varianza o percentiles por semana.

Si el promedio actual se desvía más de, digamos, 15% frente a la baseline,
vale la pena investigar.

Más allá de números, las visualizaciones ayudan mucho:
- histogramas superpuestos (train vs prod)
- serie temporal del RMSE
- distribución de cada feature importante

Hay una técnica interesante llamada adversarial validation.
Se entrena un modelo para distinguir datos de entrenamiento
vs datos de producción.
Si ese modelo logra separarlos con AUC alto,
significa que efectivamente las distribuciones son distintas.

Ejemplo conceptual:

train_df['is_train'] = 1
prod_df['is_train'] = 0
combined = pd.concat([train_df, prod_df])

model = RandomForest()
model.fit(combined[features], combined['is_train'])

Si el modelo distingue fácilmente ambos conjuntos,
probablemente hay drift.

¿Y qué hacemos si detectamos drift?

Primero validar que no sea un problema de calidad de datos.
Luego analizar la causa.
Si es real, lo normal es reentrenar con datos recientes
y actualizar la versión en el registry.

Después del reentrenamiento, se establece una nueva baseline
y se sigue monitoreando.

En producción, el drift no es una excepción.
Es algo que eventualmente va a pasar.
La clave es detectarlo antes de que impacte decisiones críticas.
""")


# ============================================================
# RETO 5: GUARDAR RESULTADOS
# ============================================================

print("\n" + "="*60)
print("RETO 5: GUARDAR RESULTADOS")
print("="*60)

predictions_output = "/opt/spark-data/processed/predictions_produccion"

# Guardar en Parquet
print("\nGuardando predicciones en Parquet...")
predictions_batch.write.mode("overwrite").parquet(predictions_output + "/parquet")
print(f" Parquet guardado en: {predictions_output}/parquet")

# Guardar en CSV
print("\nGuardando resumen en CSV...")
predictions_batch.select(
    "prediction", 
    "prediction_timestamp", 
    "model_version"
).write.mode("overwrite") \
    .option("header", "true") \
    .csv(predictions_output + "/csv")
print(f" CSV guardado en: {predictions_output}/csv")

# Verificar
print("\nVerificando archivos guardados...")
parquet_count = spark.read.parquet(predictions_output + "/parquet").count()
print(f" Registros en Parquet: {parquet_count:,}")

print("\n¿Qué formato usar según el caso de uso?")

print("""
La elección del formato realmente depende de quién va a usar los datos después.
No es lo mismo guardar predicciones para un dashboard que enviarlas a una API
o generar un reporte para gerencia.

Si los datos van para analítica interna (por ejemplo Tableau o Power BI),
lo más lógico es usar Parquet. Es eficiente, comprimido y funciona muy bien
cuando se trabaja con grandes volúmenes. Además, como es columnar,
permite leer solo lo que se necesita y eso acelera mucho las consultas.

predictions.write \
    .partitionBy("prediction_date", "model_version") \
    .parquet("s3://analytics/predictions/")

En cambio, si el objetivo es compartir un reporte,
por ejemplo con alguien que lo va a abrir en Excel,
ahí lo más práctico es CSV o XLSX.
No buscamos eficiencia distribuida, sino facilidad de uso.

predictions.select(
    "contrato_id",
    "valor_predicho",
    "fecha_prediccion"
).coalesce(1) \
 .write.option("header", "true") \
 .csv("reportes/predicciones_febrero.csv")

Si las predicciones son input para otro sistema o una API,
normalmente usaría JSON porque es fácil de consumir
desde cualquier lenguaje.  
Si el volumen es alto o necesito algo más estructurado,
AVRO es mejor opción.

predictions.select(
    struct(
        col("prediction").alias("valor_estimado"),
        col("prediction_timestamp").alias("fecha"),
        col("model_version").alias("version")
    ).alias("payload")
).write.json("api_input/")

Cuando el destino es un Data Warehouse como BigQuery o Snowflake,
lo ideal es usar el conector nativo o formatos como Parquet,
que ya están optimizados para ese tipo de motor.

predictions.write \
    .format("bigquery") \
    .option("table", "analytics.predictions") \
    .save()

Si lo que queremos es almacenamiento histórico para auditoría,
me interesa que sea inmutable y trazable.
Ahí Parquet también funciona bien,
pero acompañado de metadata (fecha, versión del modelo, número de registros, etc.).

En escenarios de streaming (Kafka, por ejemplo),
lo mejor suele ser AVRO o Protobuf.
Son más compactos que JSON y funcionan mejor cuando el tráfico es alto.

En resumen, no hay un formato único correcto.
Depende del consumidor final:

- Para analítica: Parquet.
- Para humanos: CSV o Excel.
- Para APIs: JSON.
- Para alto volumen o streaming: AVRO o Protobuf.

La decisión siempre es contextual.
""")


# ============================================================
# RETO 6: DISEÑAR PIPELINE DE PRODUCCIÓN
# ============================================================

print("\n" + "="*60)
print("""
DISEÑO DEL PIPELINE DE PRODUCCIÓN PARA SECOP

1. FRECUENCIA

Decidimos ejecutar el pipeline una vez al día, a las 2:00 AM.

Lo pensamos así porque los contratos se publican durante el día,
pero no es un caso donde necesitemos reaccionar en tiempo real.
Procesarlo en la madrugada nos permite:

- No afectar horarios laborales.
- Aprovechar menor carga de infraestructura.
- Tener las predicciones listas a primera hora.

Evaluamos hacerlo en tiempo real, pero era innecesariamente complejo.
Semanal era muy lento.
Y bajo demanda no nos daba buena planificación de recursos.

En Airflow sería algo como:
schedule_interval = '0 2 * * *'


2. ORQUESTADOR

Elegimos Airflow porque nos da visibilidad y control.

Nos permite:
- Ver cada etapa del proceso.
- Manejar reintentos automáticos.
- Configurar alertas.
- Tener trazabilidad de cada ejecución.

El flujo que planteamos sería:

- Verificar que hay datos nuevos.
- Extraer desde la API.
- Validar calidad.
- Cargar el modelo desde MLflow.
- Generar predicciones con Spark.
- Validar resultados.
- Guardar en el warehouse.
- Actualizar dashboard.
- Notificar al equipo.

Consideramos cron, pero se quedaba corto.
Streaming tampoco tenía sentido para un batch diario.


3. MONITOREO

No basta con que el job termine sin error.
Necesitamos monitorear datos, modelo e infraestructura.

A) Datos:
- Volumen esperado.
- Valores nulos.
- Duplicados.
- Valores fuera de rango.

Si algo no cumple reglas mínimas, el proceso se detiene
y se genera alerta.

B) Modelo:
- Seguimiento del RMSE.
- Cambios en la distribución (drift).
- Comparación contra baseline.

Si vemos degradación significativa,
evaluamos reentrenamiento.

C) Infraestructura:
- Tiempo de ejecución.
- Uso de recursos.
- Errores recurrentes.

Para esto usaríamos:
- Airflow UI para ejecución.
- MLflow para métricas del modelo.
- Herramientas como Prometheus o Grafana para métricas técnicas.


4. REENTRENAMIENTO

No lo dejaríamos solo calendarizado.

Planteamos un enfoque mixto:

- Reentrenamiento mensual programado.
- Reentrenamiento anticipado si detectamos drift
  o degradación de métricas.

Las condiciones podrían ser:
- Drift por encima de un umbral definido.
- RMSE empeora más de cierto porcentaje.
- Cambio significativo en el volumen de datos.

Antes de pasar un modelo nuevo a producción,
debe superar al baseline en validación.

Haríamos primero despliegue en staging,
y luego promoción a producción si todo es estable.


5. ALERTAS

Definimos niveles para no saturar al equipo.

INFO:
Eventos normales, solo registro.

WARNING:
Situaciones atípicas pero no críticas.
Notificación por Slack o correo.

ERROR:
Fallo en ejecución o métricas fuera de rango.
Notificación inmediata.

CRITICAL:
Modelo no carga o pipeline completamente caído.
Escalamiento urgente.

La idea es reaccionar rápido cuando sea necesario,
pero sin generar ruido innecesario.


RESUMEN

Nuestro flujo sería:

Airflow (2 AM)
→ Extraer datos
→ Validar calidad
→ Cargar modelo (MLflow)
→ Generar predicciones (Spark)
→ Validar resultados
→ Guardar en warehouse
→ Monitorear métricas
→ Notificar

Priorizamos estabilidad, trazabilidad y capacidad de reacción.
Más que complejidad, buscamos que el sistema sea confiable
y fácil de mantener en el tiempo.


RESUMEN ARQUITECTURA:

```
┌─────────────────────────────────────────────────────────┐
│                    AIRFLOW DAG (2 AM)                   │
└───────────────────┬─────────────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
    ┌────▼────┐         ┌─────▼──────┐
    │ Extract │         │ Load Model │
    │  Data   │         │  (MLflow)  │
    └────┬────┘         └─────┬──────┘
         │                    │
         └──────────┬─────────┘
                    │
              ┌─────▼──────┐
              │  Predict   │
              │  (Spark)   │
              └─────┬──────┘
                    │
         ┌──────────┼──────────┐
         │          │          │
    ┌────▼────┐ ┌──▼───┐ ┌───▼────┐
    │ Monitor │ │ Save │ │ Alert  │
    └─────────┘ └──────┘ └────────┘
```

""")

# ============================================================
# RETO BONUS: SIMULACIÓN DE SCORING CONTINUO
# ============================================================

print("\n" + "="*60)
print("RETO BONUS: SIMULACIÓN DE SCORING CONTINUO")
print("="*60)

# Dividir datos en lotes
batches = df_new_no_label.randomSplit([0.33, 0.33, 0.34], seed=42)

print("\nProcesando datos por lotes...")
print(f"Total de lotes: {len(batches)}")

batch_stats = []

for i, batch in enumerate(batches):
    batch_size = batch.count()
    
    if carga_exitosa:
        preds = production_model.transform(batch)
    else:
        # Simulación
        from pyspark.sql.functions import rand
        preds = batch.withColumn("prediction", 10 + rand() * 3)
    
    avg_pred = preds.select(avg("prediction")).collect()[0][0]
    std_pred = preds.select(stddev("prediction")).collect()[0][0]
    
    batch_info = {
        'lote': i + 1,
        'registros': batch_size,
        'prediccion_promedio': avg_pred,
        'desviacion_std': std_pred
    }
    batch_stats.append(batch_info)
    
    print(f"\nLote {i+1}:")
    print(f"  Registros: {batch_size:,}")
    print(f"  Predicción promedio (log): {avg_pred:.4f}")
    print(f"  Predicción promedio (COP): ${math.exp(avg_pred):,.0f}")
    print(f"  Desviación estándar: {std_pred:.4f}")

# Análisis de consistencia
print("\n=== ANÁLISIS DE CONSISTENCIA ENTRE LOTES ===")

promedios = [b['prediccion_promedio'] for b in batch_stats]
varianza_entre_lotes = sum((p - sum(promedios)/len(promedios))**2 for p in promedios) / len(promedios)

print(f"Varianza entre lotes: {varianza_entre_lotes:.6f}")

if varianza_entre_lotes < 0.01:
    print(" CONSISTENCIA ALTA: Predicciones muy similares entre lotes")
elif varianza_entre_lotes < 0.05:
    print(" CONSISTENCIA MODERADA: Algunas diferencias entre lotes")
else:
    print(" CONSISTENCIA BAJA: Diferencias significativas entre lotes")
    print("   Posible causa: Datos no homogéneos o modelo inestable")

# ============================================================
# PREGUNTAS DE REFLEXIÓN
# ============================================================

print("\n" + "="*60)
print("PREGUNTAS")
print("="*60)

print("""
1. ¿Qué pasa si los datos nuevos tienen un esquema diferente al de entrenamiento?

Si el esquema cambia, el riesgo es alto. El modelo fue entrenado con una
estructura específica, y cualquier variación puede romper el pipeline
o generar predicciones incorrectas.

Puede pasar que:
- Falten columnas obligatorias.
- Aparezcan columnas nuevas.
- Cambien los tipos de datos.
- Cambien los nombres de los campos.

Las consecuencias pueden ser:
- Errores en tiempo de ejecución.
- Predicciones mal calculadas.
- Fallas completas del pipeline.

Nosotros lo manejaríamos en cuatro niveles:

Primero, validación estricta de schema antes de predecir.
Si falta una columna crítica o el tipo no coincide, detenemos el proceso.

Segundo, capa de adaptación controlada.
Mapeamos nombres alternativos conocidos,
agregamos columnas faltantes con valores por defecto,
y convertimos tipos cuando sea seguro hacerlo.

Tercero, versionamiento de esquema.
El modelo guarda el schema con el que fue entrenado.
Cuando llega nueva data, comparamos contra esa versión.

Cuarto, monitoreo.
Si el schema cambia frecuentemente,
activamos alerta porque puede indicar cambio estructural en la fuente.

La regla que seguimos es:
adaptar cuando es seguro,
detener cuando es riesgoso.


2. ¿Cómo implementaríamos A/B testing en producción?

La idea es probar un modelo nuevo sin reemplazar el actual inmediatamente.

Haríamos un split de tráfico, por ejemplo:
80% modelo actual (A)
20% modelo nuevo (B)

Es importante que el mismo usuario siempre vea el mismo modelo,
para mantener consistencia.

Durante el experimento:
- Registramos métricas de ambos.
- Medimos desempeño técnico.
- Medimos impacto en métricas de negocio.
- Ejecutamos pruebas estadísticas.

Después de 1 o 2 semanas,
si el modelo B demuestra mejora significativa y estable,
lo promovemos gradualmente a producción completa.

Nunca reemplazamos directamente sin evidencia.


3. ¿Cuándo deberíamos retirar un modelo de producción?

Un modelo no se retira por intuición.
Se retira por evidencia.

Las principales razones serían:

Degradación técnica:
- Performance cae más de un umbral definido.
- Drift persistente.
- Latencia inaceptable.
- Fallas recurrentes.

Cambios de negocio:
- Cambian los objetivos.
- Nuevas métricas de éxito.
- Regulaciones nuevas.

Problemas estructurales de datos:
- Fuente deja de existir.
- Cambio radical en distribución.
- Calidad deteriorada permanentemente.

También si existe una alternativa claramente superior.

El proceso que seguiríamos sería:

1. Evaluación formal y documentación.
2. Comunicación a stakeholders.
3. Transición gradual con monitoreo intensivo.
4. Deprecación oficial.
5. Archivo y respaldo completo.

Nunca retiraríamos un modelo sin:
- Alternativa funcional.
- Plan de rollback.
- Notificación previa.
- Backup completo.


4. ¿Qué métricas son más importantes para nuestro caso?

Como estamos prediciendo valores de contratos SECOP,
nos enfocamos en cuatro categorías.

Métricas del modelo:
- RMSE semanal para ver error global.
- MAE segmentado por rango de valor.
- Tasa de predicciones extremas.
- Drift (por ejemplo PSI).

Métricas de datos:
- Volumen diario esperado.
- Completitud de campos críticos.
- Cambios fuertes en distribución.

Métricas operacionales:
- Latencia del batch.
- Uptime del pipeline.
- Tasa de éxito de ejecución.

Métricas de negocio:
- Precisión en auditorías priorizadas.
- Cobertura de detección.
- Impacto económico estimado.

Nuestro enfoque no es solo que el modelo funcione,
sino que genere valor y sea estable en el tiempo.

En resumen,
monitoreamos modelo, datos, sistema e impacto.
Si cualquiera de esas capas falla,
actuamos antes de que el problema escale.
""")


# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n" + "="*60)
print("RESUMEN INFERENCIA EN PRODUCCIÓN")
print("="*60)
print("Verifica que hayas completado:")
print("  [X] Modelo cargado desde MLflow Registry")
print("  [X] Predicciones batch generadas")
print("  [X] Estadísticas de predicciones calculadas")
print("  [X] Anomalías detectadas")
print("  [X] Resultados guardados (Parquet y CSV)")
print("  [X] Pipeline de producción diseñado")
print("  [X] Sistema de monitoreo especificado")
print("  [X] Estrategia de alertas definida")
print("  [X] Política de reentrenamiento establecida")
print("\nPróximos pasos sugeridos:")
print("  1. Implementar pipeline en Airflow")
print("  2. Configurar monitoreo con Prometheus + Grafana")
print("  3. Implementar A/B testing framework")
print("  4. Automatizar reentrenamiento periódico")
print("  5. Crear alertas en Slack/PagerDuty")
print("  6. Documentar runbooks para incidentes")
print("="*60)
print("\nFELICITACIONES!")
print("Has completado el taller de MLOps con Spark ML y MLflow")
print("="*60)

spark.stop()
print("Proceso finalizado correctamente")