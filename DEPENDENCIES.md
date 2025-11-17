# Gestión de Dependencias

Este proyecto utiliza diferentes archivos de dependencias para diferentes propósitos:

## 📦 Archivos de Dependencias

### `api/requirements.txt` (Producción)
**Uso:** Deploy en Render y ejecución en Docker
**Contiene:** Solo las dependencias necesarias para ejecutar la API en producción
- FastAPI, Uvicorn
- Scikit-learn, NumPy, Pandas
- Scikit-optimize (incluye scipy como dependencia)

**Instalación:**
```bash
pip install -r api/requirements.txt
```

### `requirements-dev.txt` (Desarrollo)
**Uso:** Desarrollo local, notebooks, experimentación
**Contiene:** Todas las dependencias incluyendo:
- Dependencias de producción (api/requirements.txt)
- Jupyter, Notebook, IPykernel
- Matplotlib, Seaborn (visualización)
- Pytest (testing)
- GPy (optimización avanzada)

**Instalación:**
```bash
pip install -r requirements-dev.txt
```

## 🚀 Despliegue

Render utiliza Docker, y el `Dockerfile` instala únicamente `api/requirements.txt` para mantener la imagen ligera y evitar problemas de compilación con dependencias de desarrollo.

## ⚠️ Notas Importantes

- **scipy** NO está listado explícitamente en `api/requirements.txt` porque viene incluido como dependencia de `scikit-optimize`
- Esto evita conflictos de versiones y problemas de compilación en entornos sin compilador Fortran
- El Dockerfile instala las herramientas de compilación necesarias (gcc, g++, gfortran) solo para construir la imagen
