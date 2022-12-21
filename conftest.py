import sys
import pytest
import numpy as np

sys.path.append('src/')

from physfields import ScalarField, VectorField


@pytest.fixture
def unit_square():
    x = np.linspace(-1, 1, 201)
    return np.meshgrid(x, x)

@pytest.fixture
def scalar_zero():
    return ScalarField()

@pytest.fixture
def radial():
    return ScalarField(lambda x, y: x + y)

@pytest.fixture
def trough():
    return ScalarField(lambda x, y: x**2 + y)

@pytest.fixture
def rotating_disk():
    return VectorField(lambda x, y: (y, -x))

@pytest.fixture
def source():
    return ScalarField(lambda x, y: (x, y))

