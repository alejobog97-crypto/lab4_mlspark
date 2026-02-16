FROM apache/spark:3.5.0


USER root

# -----------------------------
# Instalar Python + pip
# -----------------------------
RUN apt-get update && apt-get install -y python3 python3-pip curl

# -----------------------------
# Instalar librerías Python
# -----------------------------

RUN pip3 install --no-cache-dir \
    pyspark==3.5.0 \
    delta-spark==3.2.0 \
    sodapy \
    mlflow \
    jupyterlab

# -----------------------------
# Instalar DELTA LAKE (JAR JVM)
# -----------------------------
RUN curl -L -o /opt/spark/jars/delta-core.jar \
    https://repo1.maven.org/maven2/io/delta/delta-spark_2.12/3.2.0/delta-spark_2.12-3.2.0.jar

RUN curl -L -o /opt/spark/jars/delta-storage.jar \
    https://repo1.maven.org/maven2/io/delta/delta-storage/3.2.0/delta-storage-3.2.0.jar


# -----------------------------
# Variables Spark (Delta)
# -----------------------------
ENV SPARK_EXTRA_CLASSPATH=/opt/spark/jars/*

# -----------------------------
# Crear HOME de spark para Jupyter
# -----------------------------
RUN mkdir -p /home/spark/.local/share/jupyter/runtime \
    && chown -R spark:spark /home/spark

# DELTA jars
ENV SPARK_EXTRA_CLASSPATH=/opt/spark/jars/*

# Crear home y permisos
RUN mkdir -p /home/spark/.local/share/jupyter/runtime \
    && chown -R spark:spark /home/spark

USER spark
