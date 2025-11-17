# Makefile para automatización del proyecto GitWars Superheroes
# Compatible con Windows PowerShell y Linux/Mac

# Variables de configuración
IMAGE_NAME = gitwars-superheroes
CONTAINER_NAME = gitwars-api
PORT = 8000
TEAM_NAME = equipo_gitwars

# Detectar sistema operativo
ifeq ($(OS),Windows_NT)
    SHELL := powershell.exe
    .SHELLFLAGS := -NoProfile -Command
    RM = Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
    MKDIR = New-Item -ItemType Directory -Force -Path
    PWD = $(shell Get-Location | Select-Object -ExpandProperty Path)
    TAR = tar
else
    SHELL := /bin/bash
    RM = rm -rf
    MKDIR = mkdir -p
    PWD = $(shell pwd)
    TAR = tar
endif

.PHONY: help build run status stop clean package test-api logs restart train-model

# Comando por defecto: mostrar ayuda
help:
	@echo "GitWars Superheroes - Makefile Commands"
	@echo "========================================="
	@echo "make build       - Construir la imagen Docker"
	@echo "make run         - Ejecutar el contenedor en segundo plano"
	@echo "make status      - Mostrar el estado del contenedor"
	@echo "make stop        - Detener y eliminar el contenedor"
	@echo "make clean       - Limpiar recursos de Docker"
	@echo "make package     - Generar tarball para entrega"
	@echo "make test-api    - Probar endpoints de la API"
	@echo "make logs        - Ver logs del contenedor"
	@echo "make restart     - Reiniciar el contenedor"
	@echo "make train-model - Entrenar el modelo óptimo con Optimización Bayesiana"

# Entrenar modelo óptimo
train-model:
	@echo "Entrenando modelo óptimo con Optimización Bayesiana..."
ifeq ($(OS),Windows_NT)
	python model/create_model.py
else
	python3 model/create_model.py
endif
	@echo "Modelo entrenado exitosamente"

# Construir la imagen Docker
build:
	@echo "Construyendo imagen Docker..."
	docker build -t $(IMAGE_NAME) -f deployments/Dockerfile .
	@echo "Imagen construida exitosamente"

# Ejecutar el contenedor en segundo plano
run:
	@echo "Iniciando contenedor..."
ifeq ($(OS),Windows_NT)
	-docker stop $(CONTAINER_NAME) 2>$$null
	-docker rm $(CONTAINER_NAME) 2>$$null
else
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true
endif
	docker run -d --name $(CONTAINER_NAME) -p $(PORT):8000 $(IMAGE_NAME)
	@echo "Contenedor iniciado en http://localhost:$(PORT)"
	@echo "Documentación disponible en http://localhost:$(PORT)/docs"

# Mostrar estado del contenedor
status:
	@echo "Estado del contenedor:"
ifeq ($(OS),Windows_NT)
	docker ps -a --filter "name=$(CONTAINER_NAME)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
else
	@docker ps -a --filter "name=$(CONTAINER_NAME)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "Contenedor no encontrado"
endif

# Detener y eliminar el contenedor
stop:
	@echo "Deteniendo contenedor..."
ifeq ($(OS),Windows_NT)
	-docker stop $(CONTAINER_NAME) 2>$$null
	-docker rm $(CONTAINER_NAME) 2>$$null
else
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true
endif
	@echo "Contenedor detenido y eliminado"

# Limpiar recursos de Docker
clean: stop
	@echo "Limpiando recursos de Docker..."
ifeq ($(OS),Windows_NT)
	-docker rmi $(IMAGE_NAME) 2>$$null
else
	-docker rmi $(IMAGE_NAME) 2>/dev/null || true
endif
	docker system prune -f
	@echo "Recursos limpiados"

# Generar tarball para entrega
package:
	@echo "Generando tarball para entrega..."
ifeq ($(OS),Windows_NT)
	$(TAR) -czf $(TEAM_NAME).tar.gz --exclude=.git --exclude=.github --exclude=__pycache__ --exclude=**/__pycache__ --exclude=.pytest_cache --exclude=.venv --exclude=venv --exclude=env --exclude=*.tar.gz .
else
	$(TAR) -czf $(TEAM_NAME).tar.gz --exclude=.git --exclude=.github --exclude=__pycache__ --exclude=**/__pycache__ --exclude=.pytest_cache --exclude=.venv --exclude=venv --exclude=env --exclude=*.tar.gz .
endif
	@echo "Tarball generado: $(TEAM_NAME).tar.gz"

# Ver logs del contenedor
logs:
	@echo "Mostrando logs del contenedor..."
	docker logs $(CONTAINER_NAME) --tail 100 -f

# Reiniciar el contenedor
restart: stop run

# Probar endpoints de la API
test-api:
	@echo "Probando endpoints de la API..."
	@echo ""
	@echo "1. Testing /health"
ifeq ($(OS),Windows_NT)
	curl -s http://localhost:$(PORT)/health
else
	@curl -s http://localhost:$(PORT)/health | python3 -m json.tool || echo "Error en /health"
endif
	@echo ""
	@echo ""
	@echo "2. Testing /info"
ifeq ($(OS),Windows_NT)
	curl -s http://localhost:$(PORT)/info
else
	@curl -s http://localhost:$(PORT)/info | python3 -m json.tool || echo "Error en /info"
endif
	@echo ""
	@echo ""
	@echo "3. Testing /predict"
ifeq ($(OS),Windows_NT)
	curl -s -X POST http://localhost:$(PORT)/predict -H "Content-Type: application/json" -d "{\"features\": {\"intelligence\": 50, \"strength\": 80, \"speed\": 60, \"durability\": 70, \"combat\": 55, \"height\": 185, \"weight_kg\": 90}}"
else
	@curl -s -X POST http://localhost:$(PORT)/predict -H "Content-Type: application/json" -d '{"features": {"intelligence": 50, "strength": 80, "speed": 60, "durability": 70, "combat": 55, "height": 185, "weight_kg": 90}}' | python3 -m json.tool || echo "Error en /predict"
endif
	@echo ""
	@echo ""
	@echo "Pruebas completadas"
