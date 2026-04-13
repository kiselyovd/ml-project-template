# Serving

{% if cookiecutter.has_serving == "true" -%}
```bash
make serve
# or
docker compose up
```
{% else -%}
Serving is not enabled for this project.
{% endif -%}
