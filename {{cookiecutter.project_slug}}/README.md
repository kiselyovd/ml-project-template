# {{ cookiecutter.project_name }}

[![CI](https://img.shields.io/github/actions/workflow/status/{{ cookiecutter.github_user }}/{{ cookiecutter.project_slug }}/ci.yml?branch=main&label=CI&style=for-the-badge&logo=github)](https://github.com/{{ cookiecutter.github_user }}/{{ cookiecutter.project_slug }}/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-526CFE?style=for-the-badge&logo=materialformkdocs&logoColor=white)](https://{{ cookiecutter.github_user }}.github.io/{{ cookiecutter.project_slug }}/)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/{{ cookiecutter.github_user }}/{{ cookiecutter.project_slug }}/badges/coverage.json&style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/{{ cookiecutter.github_user }}/{{ cookiecutter.project_slug }}/actions/workflows/ci.yml)
[![License: {{ cookiecutter.license }}](https://img.shields.io/badge/License-{{ cookiecutter.license }}-yellow?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python {{ cookiecutter.python_version }}+](https://img.shields.io/badge/Python-{{ cookiecutter.python_version }}%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![HF Hub](https://img.shields.io/badge/🤗%20HF%20Hub-model-FFD21E?style=for-the-badge)](https://huggingface.co/{{ cookiecutter.hf_user }}/{{ cookiecutter.project_slug }})

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
