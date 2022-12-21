import sys
import pytest
import numpy as np

sys.path.append('src/')

from physfields import ScalarField, VectorField


class TestVector():
    def test_scalar_times_vector(self, trough, rotating_disk, unit_square):
        assert isinstance((trough * rotating_disk), VectorField)

