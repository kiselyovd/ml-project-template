"""Export trained model to HuggingFace-native format (safetensors + config.json).

Run BEFORE publish_to_hf.py so the HF repo gets proper pipeline pills / Inference
Providers instead of just a raw Lightning .ckpt.

Usage:
    python scripts/export_hf_native.py \\
        --checkpoint artifacts/checkpoints/best.ckpt \\
        --out artifacts/hf_export \\
{% if cookiecutter.task_type != "tabular" %}        --base-model <HF_BASE_MODEL_ID>
{% endif %}"""
from __future__ import annotations

import argparse
from pathlib import Path

{% if cookiecutter.task_type == "segmentation" %}

def main() -> None:
    p = argparse.ArgumentParser(description="Export segmentation model to HF-native format.")
    p.add_argument("--checkpoint", default="artifacts/checkpoints/best.ckpt")
    p.add_argument("--out", default="artifacts/hf_export")
    p.add_argument(
        "--base-model",
        default=None,
        help="HF base model ID to copy image processor from "
             "(e.g. nvidia/segformer-b2-finetuned-ade-512-512)",
    )
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from {{ cookiecutter.package_name }}.inference.predict import load_model

    lit = load_model(args.checkpoint)
    backbone = lit.model

    if not hasattr(backbone, "save_pretrained"):
        raise SystemExit(
            "Backbone is not transformers-compatible; cannot export natively. "
            "Wrap your model in a transformers PreTrainedModel subclass first."
        )

    backbone.save_pretrained(out)
    print(f"Saved model weights + config to {out}")

    if args.base_model:
        from transformers import AutoImageProcessor

        AutoImageProcessor.from_pretrained(args.base_model).save_pretrained(out)
        print(f"Copied image processor from {args.base_model} to {out}")

    print(f"HF-native export complete: {out}")


{% elif cookiecutter.task_type == "image_classification" or cookiecutter.task_type == "classification" %}

def main() -> None:
    p = argparse.ArgumentParser(description="Export image classification model to HF-native format.")
    p.add_argument("--checkpoint", default="artifacts/checkpoints/best.ckpt")
    p.add_argument("--out", default="artifacts/hf_export")
    p.add_argument(
        "--base-model",
        default=None,
        help="HF base model ID to copy image processor from "
             "(e.g. google/vit-base-patch16-224)",
    )
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from {{ cookiecutter.package_name }}.inference.predict import load_model

    lit = load_model(args.checkpoint)
    backbone = lit.model

    if not hasattr(backbone, "save_pretrained"):
        raise SystemExit(
            "Backbone is not transformers-compatible; cannot export natively. "
            "Wrap your model in a transformers PreTrainedModel subclass first."
        )

    backbone.save_pretrained(out)
    print(f"Saved model weights + config to {out}")

    if args.base_model:
        from transformers import AutoImageProcessor

        AutoImageProcessor.from_pretrained(args.base_model).save_pretrained(out)
        print(f"Copied image processor from {args.base_model} to {out}")

    print(f"HF-native export complete: {out}")


{% elif cookiecutter.task_type == "nlp" %}

def main() -> None:
    p = argparse.ArgumentParser(description="Export NLP model to HF-native format.")
    p.add_argument("--checkpoint", default="artifacts/checkpoints/best.ckpt")
    p.add_argument("--out", default="artifacts/hf_export")
    p.add_argument(
        "--base-model",
        default=None,
        help="HF base model ID to copy tokenizer from "
             "(e.g. bert-base-uncased)",
    )
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from {{ cookiecutter.package_name }}.inference.predict import load_model

    lit = load_model(args.checkpoint)
    backbone = lit.model

    if not hasattr(backbone, "save_pretrained"):
        raise SystemExit(
            "Backbone is not transformers-compatible; cannot export natively. "
            "Wrap your model in a transformers PreTrainedModel subclass first."
        )

    backbone.save_pretrained(out)
    print(f"Saved model weights + config to {out}")

    if args.base_model:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(args.base_model).save_pretrained(out)
        print(f"Copied tokenizer from {args.base_model} to {out}")

    print(f"HF-native export complete: {out}")


{% elif cookiecutter.task_type == "tabular" %}

def main() -> None:
    raise SystemExit(
        "Tabular models have no HF native export; upload joblib directly via publish_to_hf. "
        "Place your serialized model at artifacts/model.joblib and run publish_to_hf.py normally."
    )


{% else %}

def main() -> None:
    p = argparse.ArgumentParser(description="Export model to HF-native format.")
    p.add_argument("--checkpoint", default="artifacts/checkpoints/best.ckpt")
    p.add_argument("--out", default="artifacts/hf_export")
    p.add_argument("--base-model", default=None, help="HF base model ID for processor/tokenizer.")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from {{ cookiecutter.package_name }}.inference.predict import load_model

    lit = load_model(args.checkpoint)
    backbone = lit.model

    if not hasattr(backbone, "save_pretrained"):
        raise SystemExit(
            "Backbone is not transformers-compatible; cannot export natively."
        )

    backbone.save_pretrained(out)
    print(f"HF-native export complete: {out}")


{% endif %}

if __name__ == "__main__":
    main()
