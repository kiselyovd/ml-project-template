# {{ cookiecutter.project_name }}

[![CI](https://github.com/{{ cookiecutter.github_user }}/{{ cookiecutter.project_slug }}/actions/workflows/ci.yml/badge.svg)](https://github.com/{{ cookiecutter.github_user }}/{{ cookiecutter.project_slug }}/actions/workflows/ci.yml)
[![License: {{ cookiecutter.license }}](https://img.shields.io/badge/License-{{ cookiecutter.license }}-yellow.svg)](LICENSE)
[![Python {{ cookiecutter.python_version }}+](https://img.shields.io/badge/python-{{ cookiecutter.python_version }}+-blue.svg)](https://www.python.org/)

{{ cookiecutter.project_description }}

**Russian version:** [README.ru.md](README.ru.md)

## Task

Task type: `{{ cookiecutter.task_type }}` · Framework: `{{ cookiecutter.framework }}`.

## Dataset

Document dataset source, size, splits. Link to Kaggle / HF dataset page.

## Results

Fill in after training. Include metrics table with main model vs baseline.

| Model | Metric 1 | Metric 2 |
|---|---|---|
| Main | — | — |
| Baseline | — | — |

## Quick Start

```bash
uv sync --all-groups
make data
make train
make evaluate
{% if cookiecutter.has_serving == "true" -%}
make serve
docker compose up
{% endif -%}
```

## Project Structure

```
src/{{ cookiecutter.package_name }}/
├── data/
├── models/
├── training/
├── evaluation/
├── inference/
{% if cookiecutter.has_serving == "true" -%}
├── serving/
{% endif -%}
└── utils/
```

## License

{{ cookiecutter.license }} — see [LICENSE](LICENSE).
