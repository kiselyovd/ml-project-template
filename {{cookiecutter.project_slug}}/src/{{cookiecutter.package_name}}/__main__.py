"""CLI entrypoint: python -m {{ cookiecutter.package_name }}"""
from __future__ import annotations

import sys


def main() -> int:
    print("{{ cookiecutter.project_name }} — use make train / make evaluate / make serve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
