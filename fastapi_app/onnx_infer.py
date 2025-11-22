import onnxruntime as ort
import numpy as np
import json
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple

class ONNXDefectClassifier:
    def __init__(self, model_path: str, metadata_path: str):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.class_names = self.metadata['class_names']
        self.input_mean = np.array(self.metadata['input_mean'], dtype=np.float32).reshape(1, 3, 1, 1)
        self.input_std = np.array(self.metadata['input_std'], dtype=np.float32).reshape(1, 3, 1, 1)
        self.input_size = self.metadata['input_shape'][2]

        self.session = ort.InferenceSession(str(self.model_path), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"✓ Loaded ONNX model: {self.model_path}")
        print(f"  Model version: {self.metadata['model_version']}")
        print(f"  Classes: {self.class_names}")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert('RGB')
        image = image.resize((self.input_size, self.input_size))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, 0)
        img_array = (img_array - self.input_mean) / self.input_std
        return img_array.astype(np.float32)

    def predict(self, image: Image.Image) -> Tuple[int, float, Dict[str, float]]:
        input_tensor = self.preprocess(image)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        predicted_class_id = int(np.argmax(probs))
        confidence = float(probs[predicted_class_id])
        all_probs = {class_name: float(probs[i]) for i, class_name in enumerate(self.class_names)}
        return predicted_class_id, confidence, all_probs

    @property
    def model_version(self) -> str:
        return self.metadata['model_version']