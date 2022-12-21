import sys
import pytest
import numpy as np

sys.path.append('src/')

from physfields import Field, ScalarField, VectorField
from physfields.zernike import Zernike, ZernikeVector


class TestZernikeScalar():
    def test_create(self):
        assert isinstance(Zernike(0, 0), ScalarField)

    def test_create_3m3(self):
        assert isinstance(Zernike(3, -3), ScalarField)

    def test_create_wrong_n(self):
        with pytest.raises(ValueError):
            _ = Zernike(-2, 2)

    def test_create_wrong_l(self):
        with pytest.raises(ValueError):
            _ = Zernike(3, -2)

    @staticmethod
    def eval(v1, v2, field):
        x, y = field
        z = np.ma.masked_where(Field.UnitDiskMask(x, y), (v1 * v2)(x, y))
        return np.sum(z) / z.count()

    def test_ortho_1(self, unit_square):
        v1 = Zernike(3, 1)
        v2 = Zernike(5, -3)
        assert self.eval(v1, v2, unit_square) == pytest.approx(0, abs=0.002)

    def test_ortho_2(self, unit_square):
        v1 = Zernike(1, -1)
        v2 = Zernike(5, 5)
        assert self.eval(v1, v2, unit_square) == pytest.approx(0, abs=0.002)

    def test_ortho_3(self, unit_square):
        v1 = Zernike(6, 0)
        v2 = Zernike(12, 0)
        assert self.eval(v1, v2, unit_square) == pytest.approx(0, abs=0.002)

    def test_ortho_4(self, unit_square):
        v1 = Zernike(7, -1)
        v2 = Zernike(7, 1)
        assert self.eval(v1, v2, unit_square) == pytest.approx(0, abs=0.002)

    def test_norm_1(self, unit_square):
        v1 = Zernike(7, -1)
        assert self.eval(v1, v1, unit_square) == pytest.approx(1, abs=0.002)

    def test_norm_2(self, unit_square):
        v1 = Zernike(8, 0)
        assert self.eval(v1, v1, unit_square) == pytest.approx(1, abs=0.002)

    def test_norm_2(self, unit_square):
        v1 = Zernike(11, -11)
        assert self.eval(v1, v1, unit_square) == pytest.approx(1, abs=0.002)



class TestZernikeVector():
    def test_create_wrong_n(self):
        with pytest.raises(ValueError):
            _ = ZernikeVector(-2, 7)

    def test_create_wrong_l(self):
        with pytest.raises(ValueError):
            _ = ZernikeVector(3, -2)

    def test_create_wrong_laplacian(self):
        with pytest.raises(ValueError):
            _ = ZernikeVector(4, 4, False)

    def test_create_wrong_nonlaplacian(self):
        with pytest.raises(ValueError):
            _ = ZernikeVector(4, 0)

    @staticmethod
    def eval(v1, v2, field):
        x, y = field
        z = np.ma.masked_where(Field.UnitDiskMask(x, y), (v1 * v2)(x, y))
        return np.sum(z) / z.count()

    def test_ortho_1(self, unit_square):
        v1 = ZernikeVector(3, -1, False)
        v2 = ZernikeVector(5, 3, True)
        assert self.eval(v1, v2, unit_square) == pytest.approx(0, abs=0.002)

    def test_ortho_2(self, unit_square):
        v1 = ZernikeVector(1, -1)
        v2 = ZernikeVector(5, 5)
        assert self.eval(v1, v2, unit_square) == pytest.approx(0, abs=0.002)

    def test_ortho_3(self, unit_square):
        v1 = ZernikeVector(6, 4, True)
        v2 = ZernikeVector(8, -2, True)
        assert self.eval(v1, v2, unit_square) == pytest.approx(0, abs=0.002)

    def test_ortho_4(self, unit_square):
        v1 = ZernikeVector(2, 0, True)
        v2 = ZernikeVector(4, 0, False)
        assert self.eval(v1, v2, unit_square) == pytest.approx(0, abs=0.002)

    def test_norm_1(self, unit_square):
        v1 = ZernikeVector(2, 0, True)
        assert self.eval(v1, v1, unit_square) == pytest.approx(1, abs=0.005)

    def test_norm_2(self, unit_square):
        v1 = ZernikeVector(4, 2, False)
        assert self.eval(v1, v1, unit_square) == pytest.approx(1, abs=0.005)

    def test_norm_3(self, unit_square):
        v1 = ZernikeVector(5, 5)
        assert self.eval(v1, v1, unit_square) == pytest.approx(1, abs=0.005)

    def test_norm_4(self, unit_square):
        v1 = ZernikeVector(23, 17, False)
        assert self.eval(v1, v1, unit_square) == pytest.approx(1, abs=0.005)
