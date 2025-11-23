#!/usr/bin/env python3
"""
Upload trained model to MLflow model registry
Production setup: Use remote MLflow server (AWS/GCP/Azure)
"""
import mlflow
import mlflow.onnx
import onnx
import os
import sys
import json
from pathlib import Path


def upload_model_to_registry(
    model_path="models/model_artifacts/resnet18_neu.onnx",
    metadata_path="models/model_artifacts/resnet18_neu_metadata.json",
    model_name="defect-classifier",
    stage="staging"
):
    """Upload model to MLflow registry"""

    mlflow.set_tracking_uri("file:./mlruns")

    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Uploading model: {model_path}")

    # Load ONNX model object (not just path)
    onnx_model = onnx.load(model_path)

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load training history
    history_path = Path(model_path).parent / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
            best_epoch = max(history, key=lambda x: x['val_acc'])
            metadata['best_val_acc'] = best_epoch['val_acc']
            metadata['best_epoch'] = best_epoch['epoch']

    # Start MLflow run
    with mlflow.start_run(run_name=f"upload-{metadata['model_version']}"):

        # Log model (pass ONNX model object, not path)
        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            artifact_path="model",
            registered_model_name=model_name
        )

        # Log metadata
        mlflow.log_params({
            "model_version": metadata['model_version'],
            "num_classes": metadata['num_classes'],
            "input_shape": str(metadata['input_shape']),
            "opset_version": metadata['opset_version']
        })

        # Log metrics
        if 'best_val_acc' in metadata:
            mlflow.log_metric("val_accuracy", metadata['best_val_acc'])
            mlflow.log_metric("best_epoch", metadata['best_epoch'])

        run_id = mlflow.active_run().info.run_id
        print(f"✓ Model uploaded to MLflow (run_id: {run_id})")

    # Transition to stage
    client = mlflow.tracking.MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = max([int(v.version) for v in model_versions])

    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage=stage.capitalize()
    )

    print(f"✓ Model transitioned to '{stage}' stage")
    print(f"  Model: {model_name}")
    print(f"  Version: {latest_version}")
    print(f"  Stage: {stage}")

    return run_id, latest_version


def download_model_from_registry(
    model_name="defect-classifier",
    stage="production",
    output_dir="models/model_artifacts"
):
    """Download model from MLflow registry"""

    mlflow.set_tracking_uri("file:./mlruns")

    print(f"Downloading model: {model_name} (stage: {stage})")

    # Get model URI
    model_uri = f"models:/{model_name}/{stage}"

    # Download model
    model_path = mlflow.onnx.load_model(model_uri)

    print(f"✓ Model downloaded")
    return model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLflow model registry operations")
    parser.add_argument("action", choices=["upload", "download"])
    parser.add_argument("--model-path", default="models/model_artifacts/resnet18_neu.onnx")
    parser.add_argument("--model-name", default="defect-classifier")
    parser.add_argument("--stage", default="staging", choices=["staging", "production"])

    args = parser.parse_args()

    if args.action == "upload":
        if not os.path.exists(args.model_path):
            print(f"Error: Model not found at {args.model_path}")
            print("Run: make train && make export")
            sys.exit(1)

        upload_model_to_registry(
            model_path=args.model_path,
            model_name=args.model_name,
            stage=args.stage
        )

    elif args.action == "download":
        download_model_from_registry(
            model_name=args.model_name,
            stage=args.stage
        )