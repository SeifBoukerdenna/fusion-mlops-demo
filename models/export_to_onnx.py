import torch
import torch.nn as nn
from torchvision import models
import os
import json

def load_trained_model(checkpoint_path='models/model_artifacts/resnet18_neu.pth'):
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 6)  # 6 classes

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Training accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")

    return model, checkpoint.get('config', {})


def export_to_onnx(
    checkpoint_path='models/model_artifacts/resnet18_neu.pth',
    onnx_path='models/model_artifacts/resnet18_neu.onnx',
    opset_version=11
):

    print("=" * 60)
    print("Exporting Model to ONNX")
    print("=" * 60)

    model, config = load_trained_model(checkpoint_path)

    dummy_input = torch.randn(1, 3, 224, 224)

    print("\nExporting to ONNX...")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output classes: 6")
    print(f"  Opset version: {opset_version}")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    metadata = {
        'model_version': config.get('model_version', 'resnet18-neu-v1.0'),
        'num_classes': 6,
        'class_names': ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'],
        'input_shape': [1, 3, 224, 224],
        'input_mean': [0.485, 0.456, 0.406],
        'input_std': [0.229, 0.224, 0.225],
        'opset_version': opset_version
    }

    metadata_path = onnx_path.replace('.onnx', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB

    print("\n" + "=" * 60)
    print("✓ Export successful!")
    print(f"  ONNX model: {onnx_path} ({file_size:.2f} MB)")
    print(f"  Metadata: {metadata_path}")
    print("=" * 60)

    return onnx_path


if __name__ == '__main__':
    export_to_onnx()