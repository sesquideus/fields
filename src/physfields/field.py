import numpy as np


class Field():
    @staticmethod
    def UnitDiskMask(x, y):
        return x**2 + y**2 > 1

    @staticmethod
    def UnitSquareMask(x, y):
        return np.abs(x) > 1 or np.abs(y) > 1

    def __init__(self, *, mask=None, name=""):
        self.mask = mask
        self.name = name

    def __call__(self, x, y):
        out = self.function(x, y)
        if self.mask is None:
            return out
        else:
            return np.ma.masked_where(self.mask(x, y), out)

