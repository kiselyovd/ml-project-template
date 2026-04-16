"""Meta-tests: verify the cookiecutter template generates correctly."""

from __future__ import annotations


def test_template_generates(cookies, cookiecutter_context):
    """Template must scaffold without errors for every task type."""
    result = cookies.bake(extra_context=cookiecutter_context)
    assert result.exit_code == 0, f"bake failed: {result.exception}"
    assert result.exception is None
    assert result.project_path.is_dir()
    assert result.project_path.name == cookiecutter_context["project_slug"]


def test_generated_has_core_files(cookies, cookiecutter_context):
    """Every generated project has the core files."""
    result = cookies.bake(extra_context=cookiecutter_context)
    root = result.project_path
    for expected in [
        "README.md",
        "README.ru.md",
        "LICENSE",
        "pyproject.toml",
        "Makefile",
        "Dockerfile",
        ".github/workflows/ci.yml",
        "configs/config.yaml",
    ]:
        assert (root / expected).exists(), f"missing: {expected}"


def test_tabular_prunes_lightning_files(cookies):
    """Tabular projects must not contain Lightning modules."""
    from tests.fixtures.contexts import TABULAR

    result = cookies.bake(extra_context=TABULAR)
    pkg = result.project_path / "src" / TABULAR["package_name"]
    assert not (pkg / "models" / "lightning_module.py").exists()
    assert not (pkg / "models" / "factory.py").exists()
    assert (pkg / "models" / "sklearn_pipeline.py").exists()


def test_no_serving_prunes_serving(cookies):
    """has_serving=false removes serving module."""
    from tests.fixtures.contexts import CLASSIFICATION

    ctx = dict(CLASSIFICATION)
    ctx["has_serving"] = "false"
    result = cookies.bake(extra_context=ctx)
    pkg = result.project_path / "src" / ctx["package_name"]
    assert not (pkg / "serving").exists()
    assert not (result.project_path / "docker-compose.yml").exists()
