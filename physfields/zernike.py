import numpy as np

from .scalarfield import ScalarField


class Zernike(ScalarField):
    def __init__(self, n, m, masked=True):
        if abs(m) > n or (m + n) % 2 != 0:
            raise NotImplementedError
        self.n = n
        self.m = m
        self.absm = abs(m)
        self.masked = masked
        self.j = n * (n + 1) // 2 + abs(m) + (1 if (m >= 0 and n % 4 in [2, 3]) or (m <= 0 and n % 4 in [0, 1]) else 0)

    @staticmethod
    def from_noll(j):
        return Zernike(0, 0)

    def radial_part(self, x, y):
        r = np.sqrt(x * x + y * y)
        return sum([
            (-1)**s * np.math.factorial(self.n - s) /
            (np.math.factorial(s) * np.math.factorial((self.n + self.absm) // 2 - s) * np.math.factorial((self.n - self.absm) // 2 - s))
            * r ** (self.n - 2 * s)
            for s in range(0, (self.n - self.absm) // 2 + 1)],
        )

    def angular_part(self, x, y):
        if self.m == 0:
            return 1

        if self.m > 0:
            return np.sqrt(2) * np.cos(self.absm * np.arctan2(y, x))
        else:
            return np.sqrt(2) * np.sin(self.absm * np.arctan2(y, x))

    def function(self, x, y):
        z = np.sqrt(self.n + 1) * self.radial_part(x, y) * self.angular_part(x, y)
        if self.masked:
            return np.ma.masked_where(x * x + y * y >= 1, z)
        else:
            return z
