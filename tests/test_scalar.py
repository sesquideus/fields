import sys
import pytest
import numpy as np

from physfields.scalarfield import ScalarField


class TestScalar():
    def test_zero(self, scalar_zero):
        assert scalar_zero(1, 2) == 0

    def test_zero_zero(self, scalar_zero):
        assert scalar_zero(0, 0) == 0

    def test_sum(self, radial, trough):
        assert (radial + trough)(-1, -3) == -6

    def test_distributivity_plus_1(self, radial, trough):
        xy = (2, 3)
        assert (radial + trough)(*xy) == radial(*xy) + trough(*xy)

    def test_distributivity_plus_2(self, radial, trough):
        xy = (7, -4)
        assert (radial + trough)(*xy) == radial(*xy) + trough(*xy)

    def test_distributivity_plus_3(self, radial, trough):
        xy = (0.235, np.pi * 0.75)
        assert (radial + trough)(*xy) == radial(*xy) + trough(*xy)

    def test_unary_minus(self, trough, unit_square):
        assert ((-trough)(*unit_square) == -trough(*unit_square)).all()

    def test_minus_is_zero(self, trough, scalar_zero, unit_square):
        assert ((trough - trough)(*unit_square) == scalar_zero(*unit_square)).all()

    def test_distributivity_multiply(self, radial, trough, unit_square):
        assert ((2 * trough)(*unit_square) == 2 * trough(*unit_square)).all()

    def test_multiply(self, radial, trough, unit_square):
        assert ((2 * trough)(*unit_square) == trough(*unit_square) + trough(*unit_square)).all()

    def test_scalar_times_scalar(self, radial, trough):
        assert isinstance(radial * trough, ScalarField)

    def test_scalar_times_bull(self, radial, trough):
        with pytest.raises(TypeError):
            _ = radial * None
