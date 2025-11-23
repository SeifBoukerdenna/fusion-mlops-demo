"""
Unit tests for ONNX model inference
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def test_model_files_exist():
    """Test that required model files exist"""
    model_path = Path(__file__).parent.parent / "models" / "model_artifacts" / "resnet18_neu.onnx"
    metadata_path = Path(__file__).parent.parent / "models" / "model_artifacts" / "resnet18_neu_metadata.json"

    assert model_path.exists(), f"ONNX model not found at {model_path}"
    assert metadata_path.exists(), f"Metadata not found at {metadata_path}"


def test_onnx_model_loads():
    """Test that ONNX model can be loaded"""
    try:
        import onnxruntime as ort
        model_path = Path(__file__).parent.parent / "models" / "model_artifacts" / "resnet18_neu.onnx"

        if not model_path.exists():
            pytest.skip("ONNX model not found")

        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        assert session is not None

        # Check inputs
        inputs = session.get_inputs()
        assert len(inputs) == 1
        assert inputs[0].shape == [1, 3, 224, 224] or inputs[0].shape == ['batch_size', 3, 224, 224]

        # Check outputs
        outputs = session.get_outputs()
        assert len(outputs) == 1

    except ImportError:
        pytest.skip("onnxruntime not installed")


def test_onnx_inference():
    """Test that ONNX model can perform inference"""
    try:
        import onnxruntime as ort
        model_path = Path(__file__).parent.parent / "models" / "model_artifacts" / "resnet18_neu.onnx"

        if not model_path.exists():
            pytest.skip("ONNX model not found")

        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])

        # Create dummy input
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        outputs = session.run([output_name], {input_name: dummy_input})

        logits = outputs[0]

        # Validate output shape
        assert logits.shape == (1, 6), f"Expected shape (1, 6), got {logits.shape}"

        # Validate output range (logits can be any value)
        assert logits.dtype == np.float32

    except ImportError:
        pytest.skip("onnxruntime not installed")


def test_metadata_format():
    """Test that metadata JSON has correct format"""
    import json

    metadata_path = Path(__file__).parent.parent / "models" / "model_artifacts" / "resnet18_neu_metadata.json"

    if not metadata_path.exists():
        pytest.skip("Metadata file not found")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Check required fields
    assert 'model_version' in metadata
    assert 'num_classes' in metadata
    assert 'class_names' in metadata
    assert 'input_shape' in metadata

    # Validate values
    assert metadata['num_classes'] == 6
    assert len(metadata['class_names']) == 6
    assert metadata['input_shape'] == [1, 3, 224, 224]

    # Validate class names
    expected_classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    assert metadata['class_names'] == expected_classes


def test_training_history_format():
    """Test that training history has correct format"""
    import json

    history_path = Path(__file__).parent.parent / "models" / "model_artifacts" / "training_history.json"

    if not history_path.exists():
        pytest.skip("Training history not found")

    with open(history_path, 'r') as f:
        history = json.load(f)

    assert isinstance(history, list)
    assert len(history) > 0

    # Check first epoch format
    first_epoch = history[0]
    assert 'epoch' in first_epoch
    assert 'train_loss' in first_epoch
    assert 'train_acc' in first_epoch
    assert 'val_loss' in first_epoch
    assert 'val_acc' in first_epoch

    # Check values are reasonable
    assert 0 <= first_epoch['train_acc'] <= 100
    assert 0 <= first_epoch['val_acc'] <= 100


def test_class_names_consistency():
    """Test that class names are consistent across files"""
    import json

    # Expected class names
    expected_classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

    # Check metadata
    metadata_path = Path(__file__).parent.parent / "models" / "model_artifacts" / "resnet18_neu_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        assert metadata['class_names'] == expected_classes

    # Could also check handler.py, onnx_infer.py, etc.
    # For now, just metadata is sufficient


if __name__ == '__main__':
    pytest.main([__file__, '-v'])