# ml-project-template

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-grade cookiecutter template for ML projects with PyTorch Lightning / scikit-learn, Hydra, MLflow, DVC, FastAPI serving, Docker, CI/CD, and HuggingFace Hub publishing.

**Russian version:** [README.ru.md](README.ru.md)

## Quick Start

```bash
pipx install cookiecutter
cookiecutter gh:kiselyovd/ml-project-template
cd my-new-project
make setup
make test
```

## Supported task types

| task_type | framework | notes |
|---|---|---|
| classification | pytorch | Image classification |
| segmentation | pytorch | Semantic segmentation |
| keypoints | pytorch / ultralytics | Keypoint detection |
| tabular | sklearn + lightgbm | No Lightning, simpler scaffold |
| nlp | pytorch + transformers | Text classification |

See [cookiecutter.json](cookiecutter.json) for all variables.

## License

MIT — see [LICENSE](LICENSE).
