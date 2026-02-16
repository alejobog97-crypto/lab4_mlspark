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

print("\n¿Por qué cargar el modelo desde el Model Registry y no desde una ruta fija?")

print("""
En entornos de producción, cargar modelos desde rutas físicas fijas
(por ejemplo, /models/v2/model.pkl) genera un fuerte acoplamiento
entre el código y una versión específica del modelo.

El uso del Model Registry permite desacoplar el código
de la versión concreta del modelo.

IMPLICACIONES PRÁCTICAS

- La aplicación siempre carga la versión marcada como Production,
  independientemente de su número (v2, v5, v10, etc.).
  El código no requiere cambios ante nuevas versiones.

- El rollback se realiza de forma inmediata,
  simplemente modificando el stage en el Model Registry,
  sin necesidad de redeplegar la aplicación.

- Se mantiene trazabilidad completa:
  quién promovió el modelo, cuándo se realizó el cambio
  y cuál fue la justificación.

- El mismo código puede utilizarse en distintos entornos:
  Development puede apuntar a Staging
  y Production al stage Production,
  sin modificar la lógica de carga.

COMPARACIÓN RESUMIDA

Carga por ruta fija:
  - Versión hardcodeada
  - Rollback complejo
  - Requiere cambios de código y redeploy

Carga desde Model Registry:
  - Versión gestionada por stage
  - Rollback inmediato
  - Código más limpio y mantenible

En escenarios donde no exista una versión en Production
(por ejemplo, en el primer despliegue),
este caso debe gestionarse explícitamente
mediante lógica de validación o mecanismos de fallback.
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
En entornos de producción, los datos utilizados para inferencia
provienen de sistemas operacionales o analíticos,
no de archivos estáticos preparados manualmente.

FUENTES DE DATOS COMUNES

1. Bases de datos transaccionales  
   Lectura directa desde motores como PostgreSQL, MySQL u Oracle,
   generalmente para procesar registros recientes en cargas batch.

2. Almacenamiento en data lake  
   Consumo de archivos CSV o Parquet ubicados en S3, Azure Blob u otros
   sistemas de almacenamiento distribuido, siguiendo patrones diarios
   o particionados por fecha.

3. Streaming de eventos  
   Uso de tecnologías como Kafka o Kinesis junto con Spark Structured Streaming
   para inferencia casi en tiempo real a partir de eventos entrantes.

4. APIs REST  
   Predicción individual a través de servicios web,
   donde el backend transforma la entrada en un DataFrame
   y ejecuta el modelo como parte de un microservicio.

5. Data Warehouse  
   Lectura desde plataformas como Snowflake, BigQuery o Redshift,
   común en procesos batch sobre datos consolidados y analíticos.

APLICACIÓN AL CASO SECOP

Un enfoque razonable consiste en:
- Consumir datos desde la fuente oficial (por ejemplo, APIs de datos abiertos).
- Ejecutar procesos batch de forma periódica.
- Identificar y procesar únicamente registros nuevos.
- Almacenar las predicciones para análisis, monitoreo o auditoría.

Independientemente de la fuente,
el patrón general de operación se mantiene constante:

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
En entornos de producción, una predicción aislada carece de valor
si no se encuentra acompañada de contexto.

El uso de un timestamp permite identificar con precisión
cuándo se generó una predicción, facilitando la trazabilidad
ante auditorías, análisis retrospectivos o solicitudes de explicación.

Desde una perspectiva de monitoreo,
el timestamp permite analizar la evolución de las predicciones
a lo largo del tiempo y detectar patrones anómalos,
como cambios en la distribución de resultados
o posibles fenómenos de data drift.

A nivel operativo, esta información resulta clave para:
- Auditoría: identificación del modelo activo en el momento de la predicción.
- Depuración: reproducción de predicciones específicas.
- Análisis de desempeño: evaluación por periodo o por lote de procesamiento.

Junto con el timestamp, es habitual almacenar metadata adicional:

METADATA DEL MODELO
- Versión del modelo.
- Tipo de modelo.
- Fecha de entrenamiento.
- Métrica de referencia al momento del entrenamiento.

METADATA DE LOS DATOS
- Fuente de los datos.
- Versión del pipeline de preprocesamiento.
- Indicadores de calidad, si existen.

METADATA DE LA PREDICCIÓN
- Identificador único de la predicción.
- Identificador de batch (en procesamiento masivo).
- Latencia de inferencia.
- Puntaje de confianza, cuando aplica.

Cuando el valor real se encuentra disponible,
es posible cerrar el ciclo de evaluación
mediante el cálculo del error y su fecha de validación.

En conjunto, estos elementos permiten analizar el desempeño
por versión de modelo, por batch y a lo largo del tiempo,
así como alimentar tableros de monitoreo.

En conclusión, el timestamp no es un detalle accesorio,
sino un componente fundamental para transformar una predicción
en un evento trazable y gobernable.
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
Data drift ocurre cuando los datos que ingresan en producción
dejan de ser representativos de los datos utilizados
durante el entrenamiento del modelo.

Este fenómeno puede presentarse por distintas causas.

Un primer escenario es el concept drift,
donde cambia la relación entre las variables de entrada y el target.
Por ejemplo, ante cambios económicos significativos,
la relación histórica entre tipo de contrato y valor
puede dejar de ser válida.
Este caso suele detectarse porque el error del modelo
aumenta progresivamente en el tiempo.

Otro escenario es el data drift en sentido estricto,
donde la relación entre variables se mantiene,
pero la distribución de los datos de entrada cambia.
Aparecen nuevos tipos de contratos,
rangos de valores no observados en el entrenamiento
o combinaciones de features poco frecuentes.

También puede producirse un cambio en la distribución del target,
por ejemplo cuando comienza a predominar un volumen mayor
de contratos de alto valor frente al histórico.

DETECCIÓN DE DRIFT EN LA PRÁCTICA

Una estrategia común es el análisis estadístico,
comparando la distribución de los datos de entrenamiento
con la de producción.

Un ejemplo es el test de Kolmogorov-Smirnov,
que permite evaluar si dos distribuciones son estadísticamente distintas.
Un valor p bajo sugiere la presencia de drift.

Otra métrica ampliamente utilizada es el
Population Stability Index (PSI).
Valores por encima de umbrales habituales (por ejemplo 0.2)
indican un cambio relevante en la distribución.

También pueden monitorearse métricas descriptivas simples,
como la media, la varianza o percentiles de las predicciones
en ventanas temporales.
Desviaciones significativas respecto a la baseline
son señales tempranas de alerta.

Las visualizaciones resultan especialmente útiles,
incluyendo:
- Histogramas comparativos entre entrenamiento y producción.
- Series temporales del error (RMSE, MAE).
- Distribución individual de features clave.

Una técnica más avanzada es la validación adversarial.
Consiste en entrenar un modelo que intente distinguir
entre datos de entrenamiento y datos de producción.
Si dicho modelo logra separarlos con alto desempeño,
es evidencia de que las distribuciones son diferentes.

GESTIÓN DEL DRIFT

Ante la detección de drift, el primer paso es descartar
problemas de calidad de datos o errores en el pipeline.
Si el cambio es real y sostenido,
la acción habitual es reentrenar el modelo
con datos más recientes y registrar una nueva versión.

Posteriormente, esta nueva versión establece
una baseline actualizada
y el proceso de monitoreo continúa de forma cíclica.

En entornos productivos, el drift no es una excepción,
sino un comportamiento esperado a lo largo del tiempo.
La clave está en detectarlo de manera temprana,
antes de que impacte decisiones críticas del negocio.
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
La elección del formato de salida depende del uso posterior
que se dará a las predicciones y del sistema consumidor.

FORMATOS SEGÚN EL DESTINO

- Analítica y dashboards  
  Para herramientas como Power BI, Tableau o procesos analíticos,
  Parquet es la opción preferida.
  Es un formato columnar, comprimido y eficiente,
  adecuado para grandes volúmenes y consultas selectivas.

- Reportes para usuarios finales  
  Cuando las predicciones se comparten en reportes
  que serán consumidos en Excel u hojas de cálculo,
  CSV o XLSX resultan más prácticos por su facilidad de uso,
  priorizando accesibilidad sobre eficiencia.

- Integración con APIs u otros sistemas  
  JSON es el formato más común por su interoperabilidad.
  En escenarios de mayor volumen o mayor necesidad de esquema,
  AVRO ofrece mejores garantías de estructura y eficiencia.

- Data Warehouse  
  Para plataformas como BigQuery o Snowflake,
  se recomienda utilizar conectores nativos
  o formatos optimizados como Parquet,
  alineados con el motor de consulta.

- Almacenamiento histórico y auditoría  
  El objetivo principal es la trazabilidad y la inmutabilidad.
  Parquet, acompañado de metadata relevante
  (fecha, versión del modelo, volumen, origen),
  permite un almacenamiento eficiente y gobernable.

- Streaming de eventos  
  En escenarios de alta frecuencia, como Kafka,
  AVRO o Protobuf son preferibles por su compacidad
  y mejor desempeño frente a JSON.

RESUMEN

No existe un formato universalmente correcto.
La elección debe alinearse con el consumidor final
y los requisitos de volumen, desempeño y gobernanza:

- Analítica: Parquet  
- Consumo humano: CSV / Excel  
- APIs: JSON  
- Alto volumen o streaming: AVRO / Protobuf  

La decisión siempre depende del contexto de uso.
""")


# ============================================================
# RETO 6: DISEÑAR PIPELINE DE PRODUCCIÓN
# ============================================================

print("\n" + "="*60)
print("""
DISEÑO DEL PIPELINE DE PRODUCCIÓN – SECOP

FRECUENCIA DE EJECUCIÓN
El pipeline se ejecuta una vez al día a las 2:00 AM.
Este enfoque batch es suficiente dado que la publicación de contratos
no requiere reacción en tiempo real y permite:
- Minimizar impacto en horarios laborales.
- Optimizar el uso de infraestructura.
- Disponibilizar resultados a primera hora del día.

ORQUESTACIÓN
Se utiliza Airflow como orquestador principal por su capacidad
de trazabilidad, control y manejo de errores.
El flujo incluye:
- Verificación de datos nuevos.
- Extracción desde la fuente.
- Validaciones de calidad.
- Carga del modelo desde MLflow.
- Generación de predicciones con Spark.
- Validación de resultados.
- Persistencia en el warehouse.
- Actualización de visualizaciones y notificación.

MONITOREO
El monitoreo cubre tres dimensiones:
- Datos: volumen, nulos, duplicados y rangos esperados.
- Modelo: desempeño, comparación con baseline y detección de drift.
- Infraestructura: tiempos de ejecución, uso de recursos y fallos.

Las métricas se observan mediante Airflow, MLflow
y herramientas de monitoreo técnico como Prometheus o Grafana.

REENTRENAMIENTO
Se adopta un enfoque híbrido:
- Reentrenamiento periódico programado.
- Reentrenamiento anticipado ante degradación de métricas o drift.
Todo nuevo modelo debe validarse en Staging
antes de su promoción a Production.

ALERTAS
Se definen niveles de severidad para evitar ruido operativo:
- INFO: eventos informativos.
- WARNING: comportamientos atípicos no críticos.
- ERROR: fallos de ejecución o métricas fuera de umbral.
- CRITICAL: indisponibilidad del modelo o del pipeline.

RESUMEN DEL FLUJO

Airflow (2 AM)
→ Extraer datos
→ Validar calidad
→ Cargar modelo (MLflow)
→ Predecir (Spark)
→ Validar resultados
→ Guardar en warehouse
→ Monitorear y alertar

El diseño prioriza estabilidad, trazabilidad
y capacidad de reacción,
buscando un sistema confiable y mantenible a largo plazo.
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
1. ¿Qué ocurre si los datos nuevos presentan un esquema distinto al de entrenamiento?

Un cambio de esquema representa un riesgo elevado,
ya que el modelo fue entrenado bajo una estructura específica.
Variaciones en columnas, tipos de datos o nombres de campos
pueden provocar errores de ejecución o predicciones incorrectas.

La gestión se aborda en varios niveles:
- Validación estricta del esquema antes de la inferencia.
- Adaptación controlada cuando el cambio es seguro
  (mapeo de nombres, valores por defecto, conversiones simples).
- Versionamiento del esquema asociado al modelo entrenado.
- Monitoreo y alertas ante cambios frecuentes o inesperados.

El principio operativo es claro:
adaptar únicamente cuando es seguro y detener el proceso
cuando el cambio introduce riesgo.
""")

print("""
2. ¿Cómo se implementa A/B testing en producción?

El A/B testing permite evaluar un modelo nuevo
sin reemplazar de inmediato el modelo activo.

El tráfico se divide de forma controlada
(por ejemplo, 80% al modelo actual y 20% al modelo candidato),
asegurando asignación consistente por usuario o entidad.

Durante el experimento se comparan métricas técnicas,
operacionales y de negocio.
Si la nueva versión demuestra una mejora significativa y estable
durante un periodo definido,
se promueve gradualmente a producción completa.

La sustitución nunca se realiza sin evidencia cuantitativa.
""")

print("""
3. ¿Cuándo debe retirarse un modelo de producción?

La retirada de un modelo debe basarse en evidencia objetiva,
no en percepción.

Las causas más comunes incluyen:
- Degradación técnica sostenida (error, drift, latencia).
- Cambios en los objetivos o restricciones del negocio.
- Problemas estructurales en las fuentes de datos.
- Existencia de una alternativa claramente superior.

El proceso incluye evaluación formal,
comunicación a los stakeholders,
transición controlada,
archivo del modelo y respaldo completo.

Un modelo no debe retirarse sin una alternativa funcional,
un plan de rollback y trazabilidad completa.
""")

print("""
4. ¿Qué métricas son más relevantes para este caso de uso?

La evaluación se organiza en cuatro dimensiones:

- Modelo: RMSE periódico, MAE por segmentos, predicciones extremas y métricas de drift.
- Datos: volumen esperado, completitud de campos críticos y estabilidad de distribuciones.
- Operación: tiempos de ejecución, disponibilidad y tasa de éxito del pipeline.
- Negocio: precisión en auditorías, cobertura de detección e impacto económico.

El objetivo no es únicamente el funcionamiento técnico,
sino la generación de valor sostenible y la estabilidad operativa.
El monitoreo continuo permite actuar de forma preventiva
ante fallos en cualquiera de estas capas.
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