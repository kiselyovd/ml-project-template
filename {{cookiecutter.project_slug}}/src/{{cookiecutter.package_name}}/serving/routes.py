"""FastAPI routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, UploadFile

from .. import __version__
from ..inference.predict import predict
from .dependencies import get_model
from .errors import InferenceError
from .schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_model()
        loaded = True
    except Exception:
        loaded = False
    return HealthResponse(
        status="ok" if loaded else "degraded", model_loaded=loaded, version=__version__,
    )


{% if cookiecutter.framework == "pytorch" and cookiecutter.task_type in ["classification", "segmentation", "keypoints"] -%}
@router.post("/predict")
async def predict_endpoint(file: UploadFile, model=Depends(get_model)) -> dict:
    import tempfile

    suffix = "." + (file.filename or "input.bin").split(".")[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        return predict(model, tmp_path)
    except Exception as exc:
        raise InferenceError(str(exc)) from exc
{% elif cookiecutter.task_type == "nlp" -%}
from .schemas import ClassificationResponse, TextRequest


@router.post("/classify", response_model=ClassificationResponse)
def classify(req: TextRequest, model=Depends(get_model)) -> ClassificationResponse:
    result = predict(model, req.text)
    return ClassificationResponse(**result)
{% else -%}
from .schemas import FeaturesRequest, TabularResponse


@router.post("/predict", response_model=TabularResponse)
def predict_endpoint(req: FeaturesRequest, model=Depends(get_model)) -> TabularResponse:
    result = predict(model, req.features)
    return TabularResponse(**result)
{% endif -%}
