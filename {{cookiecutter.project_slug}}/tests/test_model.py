"""Model smoke tests (forward pass, output shape)."""
from __future__ import annotations

{% if cookiecutter.framework == "pytorch" and cookiecutter.task_type == "classification" -%}
import torch

from {{ cookiecutter.package_name }}.models import build_model


def test_resnet50_forward():
    model = build_model("resnet50", num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 3)
{% elif cookiecutter.framework == "sklearn" -%}
from {{ cookiecutter.package_name }}.models import build_pipeline


def test_lgbm_pipeline_builds():
    pipe = build_pipeline("lgbm", n_estimators=10)
    assert pipe.steps[-1][0] == "clf"
{% elif cookiecutter.task_type == "nlp" -%}
import pytest


@pytest.mark.skip(reason="Requires network model download; enable locally")
def test_nlp_factory():
    from {{ cookiecutter.package_name }}.models import build_model

    m = build_model("prajjwal1/bert-tiny", num_labels=3)
    assert m is not None
{% else -%}
def test_placeholder():
    assert True
{% endif -%}
