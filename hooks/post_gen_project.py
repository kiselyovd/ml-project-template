"""Post-generation cleanup and bootstrapping."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

TASK_TYPE = "{{ cookiecutter.task_type }}"
FRAMEWORK = "{{ cookiecutter.framework }}"
HAS_SERVING = "{{ cookiecutter.has_serving }}" == "true"
PACKAGE = "{{ cookiecutter.package_name }}"

PROJECT_ROOT = Path.cwd()
SRC_PKG = PROJECT_ROOT / "src" / PACKAGE


def rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def prune_for_tabular() -> None:
    rm(SRC_PKG / "models" / "factory.py")
    rm(SRC_PKG / "models" / "lightning_module.py")
    rm(SRC_PKG / "models" / "metrics.py")
    rm(SRC_PKG / "data" / "datamodule.py")
    rm(SRC_PKG / "data" / "transforms.py")


def prune_for_non_tabular() -> None:
    rm(SRC_PKG / "models" / "sklearn_pipeline.py")


def prune_serving() -> None:
    if HAS_SERVING:
        return
    rm(SRC_PKG / "serving")
    rm(PROJECT_ROOT / "tests" / "test_serving.py")
    rm(PROJECT_ROOT / "docker-compose.yml")


def try_init_git() -> None:
    try:
        subprocess.run(["git", "init", "-b", "main"], check=True, cwd=PROJECT_ROOT)
        subprocess.run(["git", "add", "."], check=True, cwd=PROJECT_ROOT)
        subprocess.run(
            ["git", "commit", "-m", "chore: initial scaffold from ml-project-template"],
            check=True,
            cwd=PROJECT_ROOT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"WARN: git init skipped: {exc}", file=sys.stderr)


def try_init_dvc() -> None:
    """Initialise DVC so `dvc add` works out of the box.

    Creates `.dvc/config` + `.dvc/.gitignore`, registers them with git so
    downstream `dvc add` produces sidecar `.dvc` files that stay tracked.
    """
    try:
        subprocess.run(["dvc", "init", "--no-scm"], check=True, cwd=PROJECT_ROOT)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"WARN: dvc init skipped: {exc}", file=sys.stderr)


def main() -> None:
    if FRAMEWORK == "sklearn" or TASK_TYPE == "tabular":
        prune_for_tabular()
    else:
        prune_for_non_tabular()
    prune_serving()
    try_init_dvc()
    try_init_git()
    print(f"Project {PROJECT_ROOT.name} ready. Next: cd {PROJECT_ROOT.name} && make setup")


if __name__ == "__main__":
    main()
