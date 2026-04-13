"""Dataset implementations."""
from __future__ import annotations

{% if cookiecutter.framework == "pytorch" -%}
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

{% if cookiecutter.task_type in ["classification", "segmentation", "keypoints"] -%}


class ImageDataset(Dataset):
    """Generic image dataset with class-subdir layout."""

    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for cls, idx in self.class_to_idx.items():
            for ext in extensions:
                self.samples.extend(
                    (p, idx) for p in (self.root / cls).glob(f"**/*{ext}")
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
{% endif -%}
{% if cookiecutter.task_type == "nlp" -%}


class TextDataset(Dataset):
    """CSV-backed text classification dataset."""

    def __init__(
        self,
        csv_path: Path | str,
        text_col: str = "text",
        label_col: str = "label",
        tokenizer: Callable | None = None,
        max_length: int = 512,
    ) -> None:
        import pandas as pd

        self.df = pd.read_csv(csv_path)
        self.text_col = text_col
        self.label_col = label_col
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        item = {"text": str(row[self.text_col]), "label": int(row[self.label_col])}
        if self.tokenizer is not None:
            enc = self.tokenizer(
                item["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            item.update({k: v.squeeze(0) for k, v in enc.items()})
        return item
{% endif -%}
{% else -%}
from pathlib import Path

import pandas as pd


def load_dataset(csv_path: Path | str) -> pd.DataFrame:
    """Load a CSV into a dataframe."""
    return pd.read_csv(csv_path)
{% endif -%}
