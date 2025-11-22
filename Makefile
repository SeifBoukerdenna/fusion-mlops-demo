.PHONY: help install train export validate package-torchserve \
        build-fastapi build-torchserve build docker-up docker-down \
        k8s-deploy k8s-delete k8s-status test clean

help:
	@echo "Fusion MLOps Demo - Makefile Commands"
	@echo "======================================"
	@echo "Setup:"
	@echo "  make install          Install Python dependencies"
	@echo ""
	@echo "ML Pipeline:"
	@echo "  make train            Train model"
	@echo "  make export           Export to ONNX"
	@echo "  make validate         Validate ONNX model"
	@echo "  make package-torchserve  Package TorchServe .mar"
	@echo ""
	@echo "Docker:"
	@echo "  make build            Build all Docker images"
	@echo "  make build-fastapi    Build FastAPI image"
	@echo "  make build-torchserve Build TorchServe image"
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
	@echo "  make benchmark        Run load tests"
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

package-torchserve:
	cd torchserve && ./package_model.sh

# Docker
build: build-fastapi build-torchserve

build-fastapi:
	docker build -t fastapi-defect-classifier:latest -f Dockerfile.fastapi .

build-torchserve:
	docker build -t torchserve-defect-classifier:latest -f Dockerfile.torchserve .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# Kubernetes
k8s-deploy:
	kubectl apply -f k8s/fastapi-deployment.yaml
	kubectl apply -f k8s/fastapi-service.yaml
	kubectl apply -f k8s/torchserve-deployment.yaml
	kubectl apply -f k8s/torchserve-service.yaml

k8s-delete:
	kubectl delete -f k8s/

k8s-status:
	kubectl get pods,svc

# Testing
test:
	pytest tests/ -v

benchmark:
	locust -f load_testing/locustfile.py --headless -u 10 -r 2 -t 30s

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	docker-compose down -v