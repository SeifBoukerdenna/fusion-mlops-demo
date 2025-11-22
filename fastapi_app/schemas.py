from pydantic import BaseModel, Field, ConfigDict

class PredictionRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image string")

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    predicted_class: str
    predicted_class_id: int
    confidence: float
    all_probabilities: dict
    model_version: str
    inference_time_ms: float

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_version: str