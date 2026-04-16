"""Validate cookiecutter inputs before project generation."""

from __future__ import annotations

import re
import sys

SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{1,49}$")
PACKAGE_RE = re.compile(r"^[a-z][a-z0-9_]{1,49}$")

project_slug = "{{ cookiecutter.project_slug }}"
package_name = "{{ cookiecutter.package_name }}"
task_type = "{{ cookiecutter.task_type }}"
framework = "{{ cookiecutter.framework }}"


def fail(msg: str) -> None:
    sys.stderr.write(f"ERROR: {msg}\n")
    sys.exit(1)


if not SLUG_RE.match(project_slug):
    fail(
        f"project_slug '{project_slug}' must be lowercase, start with a letter, "
        "and contain only letters, digits, and hyphens (2-50 chars)."
    )

if not PACKAGE_RE.match(package_name):
    fail(
        f"package_name '{package_name}' must be lowercase, start with a letter, "
        "and contain only letters, digits, and underscores (2-50 chars)."
    )

if task_type == "tabular" and framework == "pytorch":
    sys.stderr.write(
        "WARN: task_type=tabular with framework=pytorch is unusual. "
        "Proceeding, but you likely want framework=sklearn.\n"
    )

print(f"Pre-gen validation passed for {project_slug} (task={task_type}, fw={framework}).")
