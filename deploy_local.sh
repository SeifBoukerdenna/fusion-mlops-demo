#!/bin/bash
# FREE Local Kubernetes with Minikube

set -e

echo "🆓 Setting up FREE Local Kubernetes"

# Install minikube if needed
if ! command -v minikube &> /dev/null; then
    echo "Installing minikube..."
    brew install minikube
fi

# Start minikube
echo "Starting minikube cluster..."
minikube start --driver=docker --cpus=4 --memory=8192

# Build images directly in minikube
echo "Building images in minikube..."
eval $(minikube docker-env)

docker build -t fusion-fastapi:latest -f Dockerfile.fastapi .
docker build -t fusion-torchserve:latest -f Dockerfile.torchserve .
docker build -t fusion-frontend:latest -f Dockerfile.frontend .

# Update manifests for local images
cat > k8s-local.yaml << 'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: fusion-mlops
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi
  namespace: fusion-mlops
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fastapi
  template:
    metadata:
      labels:
        app: fastapi
    spec:
      containers:
      - name: fastapi
        image: fusion-fastapi:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: fastapi
  namespace: fusion-mlops
spec:
  type: NodePort
  selector:
    app: fastapi
  ports:
  - port: 8000
    nodePort: 30000
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: torchserve
  namespace: fusion-mlops
spec:
  replicas: 1
  selector:
    matchLabels:
      app: torchserve
  template:
    metadata:
      labels:
        app: torchserve
    spec:
      containers:
      - name: torchserve
        image: fusion-torchserve:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: torchserve
  namespace: fusion-mlops
spec:
  type: NodePort
  selector:
    app: torchserve
  ports:
  - port: 8080
    nodePort: 30080
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: fusion-mlops
spec:
  replicas: 1
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: fusion-frontend:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 3000
---
apiVersion: v1
kind: Service
metadata:
  name: frontend
  namespace: fusion-mlops
spec:
  type: NodePort
  selector:
    app: frontend
  ports:
  - port: 3000
    nodePort: 30300
EOF

# Deploy
echo "Deploying to minikube..."
kubectl apply -f k8s-local.yaml

# Wait for pods
echo "Waiting for pods..."
kubectl wait --for=condition=ready pod -l app=fastapi -n fusion-mlops --timeout=120s
kubectl wait --for=condition=ready pod -l app=torchserve -n fusion-mlops --timeout=180s

# Get URLs
echo ""
echo "✅ Deployment complete!"
echo ""
echo "Access services:"
minikube service fastapi -n fusion-mlops --url
minikube service torchserve -n fusion-mlops --url
minikube service frontend -n fusion-mlops --url

echo ""
echo "Or use these commands:"
echo "  minikube service fastapi -n fusion-mlops"
echo "  minikube service torchserve -n fusion-mlops"
echo "  minikube service frontend -n fusion-mlops"