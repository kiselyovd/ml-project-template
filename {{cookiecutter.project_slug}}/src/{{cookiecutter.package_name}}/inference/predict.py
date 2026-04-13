"""Inference CLI — load a checkpoint and predict on input(s)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import configure_logging, get_logger

log = get_logger(__name__)


{% if cookiecutter.framework == "pytorch" -%}
def load_model(checkpoint_path: str | Path):
    """Load a Lightning module from checkpoint."""
{%- if cookiecutter.task_type == "classification" %}
    from ..models import ClassificationModule
    return ClassificationModule.load_from_checkpoint(str(checkpoint_path))
{%- elif cookiecutter.task_type == "segmentation" %}
    from ..models import SegmentationModule
    return SegmentationModule.load_from_checkpoint(str(checkpoint_path))
{%- elif cookiecutter.task_type == "nlp" %}
    from ..models import NLPModule
    return NLPModule.load_from_checkpoint(str(checkpoint_path))
{%- elif cookiecutter.task_type == "keypoints" %}
    from ultralytics import YOLO
    return YOLO(str(checkpoint_path))
{%- endif %}


def predict(model, input_path: str | Path):
    """Run a single prediction. Returns a task-specific result dict."""
{%- if cookiecutter.task_type == "classification" %}
    import torch
    from PIL import Image

    from ..data.transforms import build_eval_transforms

    model.eval()
    tf = build_eval_transforms()
    img = Image.open(input_path).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = model._forward_logits(x) if hasattr(model, "_forward_logits") else model(x)
        probs = logits.softmax(-1).squeeze(0).tolist()
    return {"probs": probs, "pred": int(max(range(len(probs)), key=probs.__getitem__))}
{%- elif cookiecutter.task_type == "keypoints" %}
    results = model(str(input_path))
    return [r.tojson() for r in results]
{%- else %}
    raise NotImplementedError("Override predict() per project")
{%- endif %}
{% else -%}
import joblib


def load_model(path: str | Path):
    return joblib.load(path)


def predict(model, features: dict) -> dict:
    import pandas as pd

    X = pd.DataFrame([features])
    proba = model.predict_proba(X)[0].tolist()
    pred = int(model.predict(X)[0])
    return {"pred": pred, "proba": proba}
{% endif -%}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    configure_logging()
    model = load_model(args.checkpoint)
    result = predict(model, args.input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
