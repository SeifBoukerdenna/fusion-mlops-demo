import onnxruntime as ort
import numpy as np
import json
import base64
from PIL import Image
import io
from pathlib import Path
from typing import Dict, Tuple

class ONNXDefectClassifier:
    """ONNX Runtime inference for defect classification"""

    def __init__(self, model_path: str, metadata_path: str):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.class_names = self.metadata['class_names']
        self.input_mean = np.array(self.metadata['input_mean']).reshape(1, 3, 1, 1)
        self.input_std = np.array(self.metadata['input_std']).reshape(1, 3, 1, 1)
        self.input_size = self.metadata['input_shape'][2]  # 224

        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider']
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"✓ Loaded ONNX model: {self.model_path}")
        print(f"  Model version: {self.metadata['model_version']}")
        print(f"  Classes: {self.class_names}")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess PIL Image to model input format"""
        image = image.convert('RGB')
        image = image.resize((self.input_size, self.input_size))

        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, 0)

        # Critical: ensure float32
        img_array = (img_array - self.input_mean.astype(np.float32)) / self.input_std.astype(np.float32)

        return img_array.astype(np.float32)

    def predict(self, image: Image.Image) -> Tuple[int, float, Dict[str, float]]:
        """
        Run inference on a PIL Image

        Returns:
            predicted_class_id: int
            confidence: float
            all_probabilities: dict mapping class names to probabilities
        """
        # Preprocess
        input_tensor = self.preprocess(image)

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )

        logits = outputs[0][0]  # Remove batch dimension

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # Get prediction
        predicted_class_id = int(np.argmax(probs))
        confidence = float(probs[predicted_class_id])

        # Build probability dict
        all_probs = {
            class_name: float(probs[i])
            for i, class_name in enumerate(self.class_names)
        }

        return predicted_class_id, confidence, all_probs

    def decode_base64_image(self, base64_string: str) -> Image.Image:
        """Decode base64 string to PIL Image"""
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        return image

    @property
    def model_version(self) -> str:
        return self.metadata['model_version']