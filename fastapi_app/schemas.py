from pydantic import BaseModel, Field, ConfigDict
from typing import List

class PredictionRequest(BaseModel):
    """Request schema for defect prediction"""
    image: str = Field(..., description="Base64-encoded image string")

class PredictionResponse(BaseModel):
    """Response schema for defect prediction"""
    model_config = ConfigDict(protected_namespaces=())

    predicted_class: str
    predicted_class_id: int
    confidence: float
    all_probabilities: dict
    model_version: str
    inference_time_ms: float

class HealthResponse(BaseModel):
    """Health check response"""
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: str