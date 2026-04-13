"""Training smoke — one-batch overfit."""
from __future__ import annotations

{% if cookiecutter.framework == "pytorch" and cookiecutter.task_type == "classification" -%}
import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from {{ cookiecutter.package_name }}.models import ClassificationModule, build_model


def test_overfit_on_batch():
    torch.manual_seed(0)
    model = build_model("resnet50", num_classes=3, pretrained=False)
    lit = ClassificationModule(model, num_classes=3, lr=1e-3)
    x = torch.randn(4, 3, 64, 64)
    y = torch.tensor([0, 1, 2, 0])
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4)
    trainer = L.Trainer(
        max_epochs=3,
        overfit_batches=1,
        logger=False,
        enable_checkpointing=False,
        accelerator="cpu",
    )
    trainer.fit(lit, loader, loader)
    assert trainer.callback_metrics["train/loss_epoch"].item() < 2.0
{% elif cookiecutter.framework == "sklearn" -%}
import numpy as np
import pandas as pd

from {{ cookiecutter.package_name }}.models import build_pipeline


def test_fit_predict_tiny():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] > 0).astype(int)
    pipe = build_pipeline("lgbm", n_estimators=10)
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (20,)
{% else -%}
def test_placeholder():
    assert True
{% endif -%}
