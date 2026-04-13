"""Data layer smoke tests."""
from __future__ import annotations

{% if cookiecutter.framework == "pytorch" and cookiecutter.task_type in ["classification", "segmentation"] -%}
from {{ cookiecutter.package_name }}.data import ImageDataModule


def test_datamodule_instantiates(sample_data_dir):
    dm = ImageDataModule(data_dir=str(sample_data_dir), batch_size=2, num_workers=0)
    assert dm.hparams.batch_size == 2
{% elif cookiecutter.task_type == "nlp" -%}
import pandas as pd

from {{ cookiecutter.package_name }}.data import TextDataset


def test_text_dataset_loads(tmp_path):
    p = tmp_path / "sample.csv"
    pd.DataFrame({"text": ["hello", "world"], "label": [0, 1]}).to_csv(p, index=False)
    ds = TextDataset(p)
    assert len(ds) == 2
    assert ds[0]["label"] == 0
{% else -%}
import pandas as pd

from {{ cookiecutter.package_name }}.data import load_dataset


def test_load_dataset(tmp_path):
    p = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(p, index=False)
    df = load_dataset(p)
    assert len(df) == 2
{% endif -%}
