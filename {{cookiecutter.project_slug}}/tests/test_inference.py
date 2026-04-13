"""Inference smoke."""
from __future__ import annotations

{% if cookiecutter.framework == "sklearn" -%}
import numpy as np
import pandas as pd

from {{ cookiecutter.package_name }}.inference.predict import predict
from {{ cookiecutter.package_name }}.models import build_pipeline


def test_predict_returns_shape():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] > 0).astype(int)
    pipe = build_pipeline("lgbm", n_estimators=10).fit(X, y)
    result = predict(pipe, dict(X.iloc[0]))
    assert "pred" in result
    assert "proba" in result
{% else -%}
import pytest


@pytest.mark.skip(reason="Requires trained checkpoint; enable after first training")
def test_placeholder():
    pass
{% endif -%}
