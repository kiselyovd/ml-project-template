"""Training entrypoint (Hydra-powered)."""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ..utils import configure_logging, get_logger, seed_everything

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging(level=cfg.get("log_level", "INFO"))
    seed_everything(cfg.get("seed", 42))
    log.info("train.start", config=OmegaConf.to_container(cfg, resolve=True))
{%- if cookiecutter.framework == "pytorch" %}

    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import MLFlowLogger

    from ..models import build_model
{%- if cookiecutter.task_type == "classification" %}
    from ..data import ImageDataModule
    from ..models import ClassificationModule

    dm = ImageDataModule(**cfg.data)
    net = build_model(cfg.model.name, num_classes=cfg.model.num_classes)
    lit = ClassificationModule(
        net,
        num_classes=cfg.model.num_classes,
        lr=cfg.model.lr,
        model_name=cfg.model.name,
    )
{%- endif %}
{%- if cookiecutter.task_type == "segmentation" %}
    from ..data import ImageDataModule
    from ..models import SegmentationModule

    dm = ImageDataModule(**cfg.data)
    net = build_model(cfg.model.name, num_classes=cfg.model.num_classes)
    lit = SegmentationModule(
        net,
        num_classes=cfg.model.num_classes,
        lr=cfg.model.lr,
        model_name=cfg.model.name,
    )
{%- endif %}
{%- if cookiecutter.task_type == "nlp" %}
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    from ..data import TextDataset
    from ..models import NLPModule

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    train_ds = TextDataset(cfg.data.train_csv, tokenizer=tok, max_length=cfg.data.max_length)
    val_ds = TextDataset(cfg.data.val_csv, tokenizer=tok, max_length=cfg.data.max_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size)
    net = build_model(cfg.model.name, num_labels=cfg.model.num_labels)
    lit = NLPModule(
        net,
        num_labels=cfg.model.num_labels,
        lr=cfg.model.lr,
        model_name=cfg.model.name,
    )
{%- endif %}
{%- if cookiecutter.task_type == "keypoints" %}
    from ultralytics import YOLO

    model = YOLO(cfg.model.name + "-pose.pt")
    model.train(
        data=cfg.data.yaml_path,
        epochs=cfg.trainer.max_epochs,
        imgsz=cfg.data.image_size,
        batch=cfg.data.batch_size,
        project=cfg.trainer.output_dir,
        name=cfg.experiment_name,
    )
    log.info("train.done")
    return
{%- endif %}
{%- if cookiecutter.task_type != "keypoints" %}

    out_dir = Path(cfg.trainer.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=out_dir / "checkpoints",
            filename="best",
            monitor=cfg.trainer.monitor,
            mode=cfg.trainer.monitor_mode,
            save_top_k=1,
        ),
        EarlyStopping(
            monitor=cfg.trainer.monitor,
            mode=cfg.trainer.monitor_mode,
            patience=cfg.trainer.patience,
        ),
    ]
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.experiment_name, tracking_uri=cfg.trainer.tracking_uri
    )
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=callbacks,
        logger=mlflow_logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic="warn",
    )
{%- if cookiecutter.task_type == "nlp" %}
    trainer.fit(lit, train_loader, val_loader)
{%- else %}
    trainer.fit(lit, dm)
{%- endif %}
    log.info("train.done", ckpt=str(out_dir / "checkpoints" / "best.ckpt"))
{%- endif %}
{%- else %}

    import json

    import joblib
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split

    from ..data import load_dataset
    from ..models import build_pipeline

    df = load_dataset(cfg.data.csv_path)
    target = cfg.data.target_col
    X = df.drop(columns=[target])
    y = df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.seed, stratify=y,
    )
    pipe = build_pipeline(cfg.model.name, **OmegaConf.to_container(cfg.model.params, resolve=True))
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    metrics = {"report": classification_report(y_te, y_pred, output_dict=True)}
    if hasattr(pipe, "predict_proba") and len(pipe.classes_) == 2:
        y_proba = pipe.predict_proba(X_te)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_te, y_proba))
    out_dir = Path(cfg.trainer.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("train.done", out=str(out_dir))
{%- endif %}


if __name__ == "__main__":
    main()
