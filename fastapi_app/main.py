from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
from pathlib import Path
import sys
from PIL import Image
import io

sys.path.append(str(Path(__file__).parent))

from schemas import PredictionRequest, PredictionResponse, HealthResponse
from onnx_infer import ONNXDefectClassifier

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_path = Path(__file__).parent.parent / "models" / "model_artifacts" / "resnet18_neu.onnx"
    metadata_path = Path(__file__).parent.parent / "models" / "model_artifacts" / "resnet18_neu_metadata.json"

    if model_path.exists():
        model = ONNXDefectClassifier(str(model_path), str(metadata_path))
        print("✓ FastAPI server ready")
    else:
        print(f"⚠️  Model not found at {model_path}")
    yield

app = FastAPI(
    title="Defect Classification API",
    description="ONNX-powered defect classification",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"service": "Defect Classification API", "version": "1.0.0", "status": "running"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model else "model_not_loaded",
        model_loaded=model is not None,
        model_version=model.model_version if model else "unknown"
    )

@app.post("/v1/predict/multipart", response_model=PredictionResponse)
async def predict_multipart(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(503, "Model not loaded")

    try:
        start = time.time()
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        pred_id, conf, probs = model.predict(image)

        return PredictionResponse(
            predicted_class=model.class_names[pred_id],
            predicted_class_id=pred_id,
            confidence=conf,
            all_probabilities=probs,
            model_version=model.model_version,
            inference_time_ms=round((time.time() - start) * 1000, 2)
        )
    except Exception as e:
        raise HTTPException(400, f"Inference failed: {e}")

@app.get("/v1/classes")
async def get_classes():
    if not model:
        raise HTTPException(503, "Model not loaded")
    return {"classes": model.class_names, "num_classes": len(model.class_names)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)