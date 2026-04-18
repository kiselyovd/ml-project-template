# {{ cookiecutter.project_name }}

[![CI](https://img.shields.io/github/actions/workflow/status/{{ cookiecutter.github_user }}/{{ cookiecutter.project_slug }}/ci.yml?branch=main&label=CI&style=for-the-badge&logo=github)](https://github.com/{{ cookiecutter.github_user }}/{{ cookiecutter.project_slug }}/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-526CFE?style=for-the-badge&logo=materialformkdocs&logoColor=white)](https://{{ cookiecutter.github_user }}.github.io/{{ cookiecutter.project_slug }}/)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/{{ cookiecutter.github_user }}/{{ cookiecutter.project_slug }}/badges/coverage.json&style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/{{ cookiecutter.github_user }}/{{ cookiecutter.project_slug }}/actions/workflows/ci.yml)
[![License: {{ cookiecutter.license }}](https://img.shields.io/badge/License-{{ cookiecutter.license }}-yellow?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python {{ cookiecutter.python_version }}+](https://img.shields.io/badge/Python-{{ cookiecutter.python_version }}%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![HF Hub](https://img.shields.io/badge/🤗%20HF%20Hub-model-FFD21E?style=for-the-badge)](https://huggingface.co/{{ cookiecutter.hf_user }}/{{ cookiecutter.project_slug }})

{{ cookiecutter.project_description }}

**English:** [README.md](README.md)

## Задача

Тип задачи: `{{ cookiecutter.task_type }}` · Фреймворк: `{{ cookiecutter.framework }}`.

## Датасет

Укажите источник датасета, размер, разбиение. Ссылка на Kaggle / HF.

## Результаты

Заполняется после обучения. Таблица метрик: основная модель vs baseline.

| Модель | Метрика 1 | Метрика 2 |
|---|---|---|
| Основная | — | — |
| Baseline | — | — |

## Быстрый старт

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

## Структура проекта

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

## Лицензия

{{ cookiecutter.license }} — см. [LICENSE](LICENSE).
