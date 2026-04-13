"""Pydantic request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    version: str


{% if cookiecutter.framework == "pytorch" and cookiecutter.task_type == "classification" -%}
class PredictionResponse(BaseModel):
    pred: int = Field(..., description="Argmax class index")
    probs: list[float]
{% elif cookiecutter.framework == "pytorch" and cookiecutter.task_type == "segmentation" -%}
class SegmentationResponse(BaseModel):
    mask_base64: str = Field(..., description="PNG mask, base64-encoded")
{% elif cookiecutter.framework == "pytorch" and cookiecutter.task_type == "keypoints" -%}
class KeypointDetection(BaseModel):
    bbox: list[float]
    keypoints: list[list[float]]
    score: float


class DetectionResponse(BaseModel):
    detections: list[KeypointDetection]
{% elif cookiecutter.task_type == "nlp" -%}
class TextRequest(BaseModel):
    text: str


class ClassificationResponse(BaseModel):
    pred: int
    top_k: list[dict]
{% else -%}
class FeaturesRequest(BaseModel):
    features: dict


class TabularResponse(BaseModel):
    pred: int
    proba: list[float]
{% endif -%}
