"""Shared pytest fixtures for template tests."""

from __future__ import annotations

import pytest

from tests.fixtures.contexts import ALL_CONTEXTS


@pytest.fixture(params=ALL_CONTEXTS, ids=[name for name, _ in ALL_CONTEXTS])
def cookiecutter_context(request: pytest.FixtureRequest) -> dict:
    _, ctx = request.param
    return ctx
