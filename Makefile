.PHONY: help install train export validate package-torchserve \
        build-fastapi build-torchserve build docker-up docker-down \
        k8s-deploy k8s-delete k8s-status test clean docker-logs \
        registry-upload registry-download registry-ui

help:
	@echo "Fusion MLOps Demo - Production Flow"
	@echo "====================================="
	@echo "Setup:"
	@echo "  make install          Install Python dependencies"
	@echo ""
	@echo "ML Pipeline:"
	@echo "  make train            Train model"
	@echo "  make export           Export to ONNX"
	@echo "  make validate         Validate ONNX model"
	@echo ""
	@echo "Model Registry (Production):"
	@echo "  make registry-upload  Upload model to MLflow registry"
	@echo "  make registry-download Download model from registry"
	@echo "  make registry-ui      Start MLflow UI"
	@echo ""
	@echo "Docker:"
	@echo "  make build            Build all Docker images"
	@echo "  make docker-up        Start with docker-compose"
	@echo "  make docker-down      Stop docker-compose"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make k8s-deploy       Deploy to Kubernetes"
	@echo "  make k8s-delete       Delete from Kubernetes"
	@echo "  make k8s-status       Check deployment status"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run unit tests"
	@echo "  make ui               Start test UI"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove artifacts"

# Setup
install:
	pip install -r requirements.txt

# ML Pipeline
train:
	python models/train_defect_classifier.py

export:
	python models/export_to_onnx.py

validate:
	python models/validate_onnx.py

# Model Registry Operations (Production)
registry-upload:
	@echo "Uploading model to MLflow registry..."
	python scripts/model_registry.py upload --stage staging
	@echo ""
	@echo "To promote to production:"
	@echo "  python scripts/model_registry.py upload --stage production"

registry-download:
	@echo "Downloading model from registry..."
	python scripts/model_registry.py download --stage production || \
	python scripts/model_registry.py download --stage staging

registry-ui:
	@echo "Starting MLflow UI..."
	@echo "Open: http://localhost:5000"
	mlflow ui

# TorchServe
package-torchserve:
	cd torchserve && ./package_model.sh

# Docker
build: build-fastapi build-torchserve

build-fastapi:
	@echo "Downloading model for Docker build..."
	@$(MAKE) registry-download || (echo "⚠️  No model in registry, training..." && $(MAKE) train export)
	docker build -t fastapi-defect-classifier:latest -f Dockerfile.fastapi .

build-torchserve:
	@echo "Downloading model for Docker build..."
	@$(MAKE) registry-download || (echo "⚠️  No model in registry, training..." && $(MAKE) train export)
	docker build -t torchserve-defect-classifier:latest -f Dockerfile.torchserve .

docker-up:
	@$(MAKE) registry-download || (echo "⚠️  No model in registry, training..." && $(MAKE) train export)
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-test:
	@echo "Testing FastAPI..."
	@curl -s http://localhost:8000/health | jq || echo "FastAPI not responding"
	@echo "\nTesting TorchServe..."
	@curl -s http://localhost:8080/ping | jq || echo "TorchServe not responding"

# Kubernetes
k8s-deploy:
	kubectl apply -f k8s/fastapi-deployment.yaml
	kubectl apply -f k8s/fastapi-service.yaml
	kubectl apply -f k8s/torchserve-deployment.yaml
	kubectl apply -f k8s/torchserve-service.yaml

k8s-delete:
	kubectl delete -f k8s/

k8s-status:
	kubectl get pods,svc,deployments

k8s-forward:
	@echo "Starting port forwards..."
	@echo "FastAPI: http://localhost:8000"
	@echo "TorchServe: http://localhost:8080"
	kubectl port-forward service/fastapi-service 8000:8000 & \
	kubectl port-forward service/torchserve-service 8080:8080 &

# Testing
test:
	pytest tests/ -v

ui:
	python serve_ui.py

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	docker compose down -v 2>/dev/null || true
	rm -rf mlruns/ mlartifacts/ 2>/dev/null || true