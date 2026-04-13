"""Data layer."""
from __future__ import annotations

{% if cookiecutter.framework == "pytorch" -%}
{% if cookiecutter.task_type in ["classification", "segmentation", "keypoints"] -%}
from .dataset import ImageDataset
from .datamodule import ImageDataModule
{% endif -%}
{% if cookiecutter.task_type == "nlp" -%}
from .dataset import TextDataset
{% endif -%}
{% else -%}
from .dataset import load_dataset
{% endif -%}
