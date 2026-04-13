"""Model factory — returns a torch.nn.Module by name."""
from __future__ import annotations

{% if cookiecutter.framework == "pytorch" -%}
from torch import nn

{% if cookiecutter.task_type == "classification" -%}
def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "convnextv2_tiny":
        from transformers import ConvNextV2ForImageClassification

        return ConvNextV2ForImageClassification.from_pretrained(
            "facebook/convnextv2-tiny-22k-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    if name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown model: {name}")
{% endif -%}

{% if cookiecutter.task_type == "segmentation" -%}
def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "segformer_b2":
        from transformers import SegformerForSemanticSegmentation

        return SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    if name == "unet":
        from torchvision.models.segmentation import deeplabv3_resnet50

        return deeplabv3_resnet50(num_classes=num_classes, weights_backbone=None)
    raise ValueError(f"Unknown model: {name}")
{% endif -%}

{% if cookiecutter.task_type == "keypoints" -%}
def build_model(name: str, num_keypoints: int, pretrained: bool = True) -> nn.Module:
    if name.startswith("yolo26"):
        from ultralytics import YOLO

        return YOLO(f"{name}-pose.pt" if pretrained else f"{name}-pose.yaml")
    raise ValueError(f"Unknown model: {name}")
{% endif -%}

{% if cookiecutter.task_type == "nlp" -%}
def build_model(name: str, num_labels: int) -> nn.Module:
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
{% endif -%}
{% endif -%}
