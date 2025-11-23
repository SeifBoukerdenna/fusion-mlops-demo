#!/usr/bin/env python3
"""Push trained models to Hugging Face Hub (free model registry)"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
import argparse

def push_to_hub(
    model_name="defect-classifier-resnet18",
    repo_id=None,  # format: "username/model-name"
    token=None,
    private=False
):
    """
    Push models to Hugging Face Hub

    Setup:
    1. Create account at https://huggingface.co
    2. Get token from https://huggingface.co/settings/tokens
    3. Set environment variable: export HF_TOKEN=your_token_here
    """

    print("=" * 60)
    print("Pushing Models to Hugging Face Hub")
    print("=" * 60)

    # Get token from env or parameter
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        print("\n❌ No token found!")
        print("Get your token from: https://huggingface.co/settings/tokens")
        print("Then run: export HF_TOKEN=your_token_here")
        return False

    api = HfApi()

    # Get username
    user_info = api.whoami(token=token)
    username = user_info['name']

    # Create repo ID
    if not repo_id:
        repo_id = f"{username}/{model_name}"

    print(f"\nRepository: {repo_id}")
    print(f"Visibility: {'Private' if private else 'Public'}")

    # Create or get repo
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✓ Repository ready")
    except Exception as e:
        print(f"⚠️  {e}")

    # Files to upload
    artifacts_dir = Path("models/model_artifacts")
    files_to_upload = [
        ("resnet18_neu.pth", "pytorch_model.pth"),
        ("resnet18_neu.onnx", "model.onnx"),
        ("resnet18_neu_metadata.json", "metadata.json"),
        ("training_history.json", "training_history.json"),
    ]

    print("\nUploading files...")
    for local_name, hub_name in files_to_upload:
        local_path = artifacts_dir / local_name
        if local_path.exists():
            try:
                upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=hub_name,
                    repo_id=repo_id,
                    token=token,
                )
                print(f"  ✓ {hub_name}")
            except Exception as e:
                print(f"  ✗ {hub_name}: {e}")
        else:
            print(f"  ⚠️  {local_name} not found, skipping")

    # Create model card
    model_card = f"""---
tags:
- computer-vision
- defect-detection
- pytorch
- onnx
library_name: pytorch
---

# Defect Classifier (ResNet18)

Steel surface defect classification model trained on NEU-DET dataset.

## Model Details

- **Architecture**: ResNet18
- **Classes**: 6 defect types (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches)
- **Input**: 224x224 RGB images
- **Formats**: PyTorch (.pth), ONNX (.onnx)

## Files

- `pytorch_model.pth`: PyTorch checkpoint
- `model.onnx`: ONNX format for deployment
- `metadata.json`: Model configuration
- `training_history.json`: Training metrics

## Usage

### PyTorch

```python
import torch
from torchvision import models
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(repo_id="{repo_id}", filename="pytorch_model.pth")

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 6)
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### ONNX

```python
import onnxruntime as ort
from huggingface_hub import hf_hub_download

# Download ONNX model
model_path = hf_hub_download(repo_id="{repo_id}", filename="model.onnx")

# Load with ONNX Runtime
session = ort.InferenceSession(model_path)
```

## Training

See training history in `training_history.json` for detailed metrics.
"""

    readme_path = Path("README_model.md")
    readme_path.write_text(model_card)

    try:
        upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
        print(f"  ✓ README.md")
        readme_path.unlink()
    except Exception as e:
        print(f"  ✗ README.md: {e}")

    print("\n" + "=" * 60)
    print("✓ Upload Complete!")
    print(f"View at: https://huggingface.co/{repo_id}")
    print("=" * 60)

    return True


def download_from_hub(repo_id, output_dir="models/model_artifacts"):
    """Download models from Hugging Face Hub"""

    from huggingface_hub import hf_hub_download

    print(f"Downloading from {repo_id}...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        ("pytorch_model.pth", "resnet18_neu.pth"),
        ("model.onnx", "resnet18_neu.onnx"),
        ("metadata.json", "resnet18_neu_metadata.json"),
        ("training_history.json", "training_history.json"),
    ]

    for hub_name, local_name in files:
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=hub_name,
                cache_dir=None,
            )
            # Copy to output dir
            import shutil
            shutil.copy(downloaded, output_dir / local_name)
            print(f"  ✓ {local_name}")
        except Exception as e:
            print(f"  ✗ {hub_name}: {e}")

    print(f"✓ Models downloaded to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push/pull models to Hugging Face Hub")
    parser.add_argument("action", choices=["push", "pull"], help="push or pull models")
    parser.add_argument("--repo-id", help="Repository ID (username/model-name)")
    parser.add_argument("--model-name", default="defect-classifier-resnet18", help="Model name")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--token", help="HuggingFace token (or set HF_TOKEN env var)")

    args = parser.parse_args()

    if args.action == "push":
        push_to_hub(
            model_name=args.model_name,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private
        )
    else:  # pull
        if not args.repo_id:
            print("Error: --repo-id required for pull")
        else:
            download_from_hub(args.repo_id)