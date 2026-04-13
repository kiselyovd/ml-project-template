"""Reference cookiecutter contexts for parameterized template tests."""
from __future__ import annotations

CLASSIFICATION = {
    "project_name": "Demo Classifier",
    "project_slug": "demo-classifier",
    "package_name": "demo_classifier",
    "task_type": "classification",
    "framework": "pytorch",
    "has_serving": "true",
    "python_version": "3.13",
}

SEGMENTATION = {
    "project_name": "Demo Segmenter",
    "project_slug": "demo-segmenter",
    "package_name": "demo_segmenter",
    "task_type": "segmentation",
    "framework": "pytorch",
    "has_serving": "true",
    "python_version": "3.13",
}

KEYPOINTS = {
    "project_name": "Demo Keypoints",
    "project_slug": "demo-keypoints",
    "package_name": "demo_keypoints",
    "task_type": "keypoints",
    "framework": "pytorch",
    "has_serving": "true",
    "python_version": "3.13",
}

TABULAR = {
    "project_name": "Demo Tabular",
    "project_slug": "demo-tabular",
    "package_name": "demo_tabular",
    "task_type": "tabular",
    "framework": "sklearn",
    "has_serving": "true",
    "python_version": "3.13",
}

NLP = {
    "project_name": "Demo NLP",
    "project_slug": "demo-nlp",
    "package_name": "demo_nlp",
    "task_type": "nlp",
    "framework": "pytorch",
    "has_serving": "true",
    "python_version": "3.13",
}

ALL_CONTEXTS = [
    ("classification", CLASSIFICATION),
    ("segmentation", SEGMENTATION),
    ("keypoints", KEYPOINTS),
    ("tabular", TABULAR),
    ("nlp", NLP),
]
