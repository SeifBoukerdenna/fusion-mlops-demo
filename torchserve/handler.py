import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
import json
import base64
import logging

logger = logging.getLogger(__name__)

class DefectClassifierHandler:
    def __init__(self):
        self.model = None
        self.device = None
        self.initialized = False
        self.class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def initialize(self, context):
        properties = context.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        model_dir = properties.get("model_dir")
        model_pt_path = f"{model_dir}/resnet18_neu.pth"

        logger.info(f"Loading model from {model_pt_path}")

        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 6)

        checkpoint = torch.load(model_pt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.initialized = True
        logger.info("Model initialized successfully")

    def preprocess(self, requests):
        images = []

        for request in requests:
            data = request.get("body") or request.get("data")

            if data is None:
                continue

            if isinstance(data, dict):
                image_data = data.get("image")
                if image_data:
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                else:
                    raise ValueError("No 'image' field")
            elif isinstance(data, (bytearray, bytes)):
                image = Image.open(io.BytesIO(data)).convert('RGB')
            else:
                raise ValueError(f"Unsupported type: {type(data)}")

            images.append(self.transform(image))

        if not images:
            return None

        return torch.stack(images).to(self.device)

    def inference(self, model_input):
        if model_input is None:
            return None

        with torch.no_grad():
            outputs = self.model(model_input)
            probs = F.softmax(outputs, dim=1)
        return probs

    def postprocess(self, inference_output):
        if inference_output is None:
            return [{"status": "ok"}]

        responses = []
        for probs in inference_output:
            probs_cpu = probs.cpu().numpy()
            predicted_class_id = int(probs_cpu.argmax())
            confidence = float(probs_cpu[predicted_class_id])

            responses.append({
                "predicted_class": self.class_names[predicted_class_id],
                "predicted_class_id": predicted_class_id,
                "confidence": confidence,
                "all_probabilities": {self.class_names[i]: float(probs_cpu[i]) for i in range(len(self.class_names))},
                "model_version": "resnet18-neu-v1.0"
            })

        return responses

    def handle(self, data, context):
        try:
            model_input = self.preprocess(data)
            model_output = self.inference(model_input)
            return self.postprocess(model_output)
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return [{"error": str(e)}]

_service = DefectClassifierHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
    return _service.handle(data, context)