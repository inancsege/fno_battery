import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_data():
    """Return sample data for testing."""
    return np.array([1, 2, 3, 4, 5])

@pytest.fixture
def sample_2d_data():
    """Return 2D sample data for testing."""
    return np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])

@pytest.fixture
def sample_3d_data():
    """Return 3D sample data for testing."""
    return np.array([[[1], [2]], [[3], [4]], [[5], [6]]])

@pytest.fixture
def sample_multi_input_data():
    """Return a dictionary of multi-input data."""
    num_samples = 10
    return {
        'voltage': np.ones((num_samples, 1)),
        'current': np.ones((num_samples, 1)) * 2,
        'temperature': np.ones((num_samples, 1)) * 3,
        'capacity': np.ones((num_samples, 1)) * 4
    }, np.arange(num_samples) 