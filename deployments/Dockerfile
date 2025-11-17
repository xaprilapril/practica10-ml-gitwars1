# Usar imagen base oficial de Python
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar SOLO requirements de producción (api/requirements.txt)
# Nota: No usar requirements-dev.txt en producción para evitar dependencias innecesarias
COPY api/requirements.txt /app/api/requirements.txt

# Instalar dependencias del sistema necesarias para compilación
# gfortran: Requerido para compilar scipy (dependencia de scikit-optimize)
# libopenblas-dev, liblapack-dev: Bibliotecas de álgebra lineal optimizadas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python en orden específico
# 1. Actualizar pip, setuptools y wheel
# 2. Instalar numpy primero (requerido por scipy)
# 3. Instalar scipy explícitamente (dejando que pip elija versión compatible)
# 4. Instalar resto de dependencias de api/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.26.4 && \
    pip install --no-cache-dir scipy && \
    pip install --no-cache-dir -r /app/api/requirements.txt

# Copiar todo el proyecto al contenedor
COPY . /app

# Crear directorio para el modelo si no existe
RUN mkdir -p /app/model

# Variable de entorno para Python
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Exponer puerto dinámico (configurable por $PORT)
EXPOSE $PORT

# Healthcheck ajustable al puerto
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/health')" || exit 1

# Comando por defecto usando variable de entorno PORT
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
