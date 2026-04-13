# {{ cookiecutter.project_name }}

[![CI](https://github.com/{{ cookiecutter.github_user }}/{{ cookiecutter.project_slug }}/actions/workflows/ci.yml/badge.svg)](https://github.com/{{ cookiecutter.github_user }}/{{ cookiecutter.project_slug }}/actions/workflows/ci.yml)
[![License: {{ cookiecutter.license }}](https://img.shields.io/badge/License-{{ cookiecutter.license }}-yellow.svg)](LICENSE)
[![Python {{ cookiecutter.python_version }}+](https://img.shields.io/badge/python-{{ cookiecutter.python_version }}+-blue.svg)](https://www.python.org/)

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
