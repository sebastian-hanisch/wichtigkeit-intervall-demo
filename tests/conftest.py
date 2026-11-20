"""Gemeinsame Fixtures: das eingefrorene Raster und das Beispielmodell."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wik_evaluation as E  # noqa: E402


@pytest.fixture(scope="session")
def grid():
    return E.load_grid()


@pytest.fixture(scope="session")
def example():
    return E.load_example()
