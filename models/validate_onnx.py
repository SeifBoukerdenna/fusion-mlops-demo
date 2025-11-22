import onnxruntime as ort
import numpy as np
import json
import os
import sys

def validate_onnx_model(
    onnx_path='models/model_artifacts/resnet18_neu.onnx',
    metadata_path='models/model_artifacts/resnet18_neu_metadata.json'
):
    """
    Validate ONNX model can be loaded and performs inference correctly.
    This is the ML-specific CI gate.
    """

    print("=" * 60)
    print("ONNX Model Validation (ML CI Gate)")
    print("=" * 60)

    if not os.path.exists(onnx_path):
        print(f"✗ ONNX model not found: {onnx_path}")
        return False

    print(f"\n1. Checking file: {onnx_path}")
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"   File size: {file_size:.2f} MB")

    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"   Model version: {metadata.get('model_version', 'N/A')}")

    print("\n2. Loading ONNX model with ONNXRuntime...")
    try:
        ort_session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False

    print("\n3. Verifying model inputs/outputs...")

    inputs = ort_session.get_inputs()
    if len(inputs) != 1:
        print(f"   ✗ Expected 1 input, got {len(inputs)}")
        return False

    input_name = inputs[0].name
    input_shape = inputs[0].shape
    print(f"   Input name: {input_name}")
    print(f"   Input shape: {input_shape}")

    outputs = ort_session.get_outputs()
    if len(outputs) != 1:
        print(f"   ✗ Expected 1 output, got {len(outputs)}")
        return False

    output_name = outputs[0].name
    output_shape = outputs[0].shape
    print(f"   Output name: {output_name}")
    print(f"   Output shape: {output_shape}")

    expected_num_classes = metadata.get('num_classes', 6)
    if output_shape[-1] != expected_num_classes:
        print(f"   ✗ Expected {expected_num_classes} output classes, got {output_shape[-1]}")
        return False

    print("   ✓ Input/Output shapes validated")

    print("\n4. Running sample inference...")
    try:
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

        outputs = ort_session.run(
            [output_name],
            {input_name: dummy_input}
        )

        logits = outputs[0]
        print(f"   Output shape: {logits.shape}")
        print(f"   Output dtype: {logits.dtype}")

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        predicted_class = np.argmax(probs, axis=1)[0]
        confidence = probs[0][predicted_class]

        print(f"   Predicted class: {predicted_class}")
        print(f"   Confidence: {confidence:.4f}")

        # Sanity checks
        if not (0 <= predicted_class < expected_num_classes):
            print(f"   ✗ Invalid predicted class: {predicted_class}")
            return False

        if not (0 <= confidence <= 1):
            print(f"   ✗ Invalid confidence value: {confidence}")
            return False

        if not np.isclose(np.sum(probs), 1.0, atol=1e-5):
            print(f"   ✗ Probabilities don't sum to 1: {np.sum(probs)}")
            return False

        print("   ✓ Inference successful")

    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All validation checks passed!")
    print("  Model is ready for deployment")
    print("=" * 60)

    return True


if __name__ == '__main__':
    success = validate_onnx_model()

    sys.exit(0 if success else 1)