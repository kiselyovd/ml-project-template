# ml-project-template

[![CI](https://img.shields.io/github/actions/workflow/status/kiselyovd/ml-project-template/ci.yml?branch=main&label=CI&style=for-the-badge&logo=github)](https://github.com/kiselyovd/ml-project-template/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Cookiecutter](https://img.shields.io/badge/scaffold-cookiecutter-D4AA00?style=for-the-badge&logo=cookiecutter&logoColor=white)](https://www.cookiecutter.io/)

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
