"""Models layer."""
from __future__ import annotations

{% if cookiecutter.framework == "pytorch" -%}
from .factory import build_model
{% if cookiecutter.task_type == "classification" -%}
from .lightning_module import ClassificationModule
{% endif -%}
{% if cookiecutter.task_type == "segmentation" -%}
from .lightning_module import SegmentationModule
{% endif -%}
{% if cookiecutter.task_type == "nlp" -%}
from .lightning_module import NLPModule
{% endif -%}
{% else -%}
from .sklearn_pipeline import build_pipeline
{% endif -%}
