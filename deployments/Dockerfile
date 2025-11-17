# Usar imagen base oficial de Python
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de requirements primero (para aprovechar cache de Docker)
COPY api/requirements.txt /app/api/requirements.txt

# Instalar dependencias del sistema necesarias para scikit-optimize
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/api/requirements.txt

# Copiar todo el proyecto al contenedor
COPY . /app

# Crear directorio para el modelo si no existe
RUN mkdir -p /app/model

# Variable de entorno para Python
ENV PYTHONUNBUFFERED=1

# Exponer puertos (8000 para local, 10000 para Render)
EXPOSE 8000
EXPOSE 10000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Comando por defecto para ejecución local (puerto 8000)
# Render sobrescribirá esto con el puerto 10000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
