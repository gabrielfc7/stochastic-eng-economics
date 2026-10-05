"""Pytest fixtures and configuration for stochastic engineering economics tests."""

import pytest
import numpy as np


@pytest.fixture
def rng():
    return np.random.default_rng(42)
