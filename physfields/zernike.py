import numpy as np

from .scalarfield import ScalarField
from .vectorfield import VectorField


def triangular_row(n):
    """ Return the 0-indexed row in which nth triangular number is found """
    assert(isinstance(n, int))
    return int(np.floor(np.sqrt(1 + 8 * n) - 1) / 2)


def triangular_less(n):
    """ Return the largest triangular number less than n """
    assert(isinstance(n, int))
    row = triangular_row(n)
    return row * (row + 1) // 2


def triangular_column(n):
    """ Return the column of number n in triangle """
    assert(isinstance(n, int))
    return n - triangular_less(n)


class Zernike(ScalarField):
    """
        Lazily evaluated scalar Zernike polynomial
    """

    def __init__(self, n, l, masked=True):
        """
            A new Zernike scalar field defined in terms of radial and angular component order n and l:
            n: int, non-negative radial component order
            l: int, angular component order, |l| < n and l = n (mod 2)
            masked: mask the area outside the unit disk
        """
        if abs(l) > n or (l + n) % 2 != 0:
            raise NotImplementedError("|l| must be < n and also l = n (mod 2)")

        self.n = n
        self.l = l
        self.absl = abs(l)
        self.masked = masked

        # Set noll and ANSI indices for convenience
        self.noll = n * (n + 1) // 2 + abs(l) + (1 if (l >= 0 and n % 4 in [2, 3]) or (l <= 0 and n % 4 in [0, 1]) else 0)
        self.ansi = (n * (n + 2) + l) / 2

    @staticmethod
    def noll_to_nl(noll):
        n = triangular_row(noll - 1)
        l = (-1)**noll * ((n % 2) + 2 * int((triangular_column(noll - 1) + ((n + 1) % 2)) / 2))
        return n, l

    @staticmethod
    def ansi_to_nl(ansi):
        n = triangular_row(ansi)
        l = (ansi - triangular_less(ansi)) * 2 - n
        return n, l

    @classmethod
    def from_noll(cls, noll):
        """
            Static constructor from noll index
        """
        return cls(*cls.noll_to_nl(noll))

    @classmethod
    def from_ANSI(cls, ansi):
        """
            Static constructor from ANSI index
        """
        return cls(*cls.ansi_to_nl(ansi))

    def radial_part(self, x, y):
        r = np.sqrt(x * x + y * y)
        return sum([
            (-1)**s * np.math.factorial(self.n - s) /
            (np.math.factorial(s) * np.math.factorial((self.n + self.absl) // 2 - s) * np.math.factorial((self.n - self.absl) // 2 - s))
            * r ** (self.n - 2 * s)
            for s in range(0, (self.n - self.absl) // 2 + 1)],
        )

    def angular_part(self, x, y):
        if self.l == 0:
            return 1

        if self.l > 0:
            return np.sqrt(2) * np.cos(self.absl * np.arctan2(y, x))
        else:
            return np.sqrt(2) * np.sin(self.absl * np.arctan2(y, x))

    def function(self, x, y):
        z = np.sqrt(self.n + 1) * self.radial_part(x, y) * self.angular_part(x, y)
        if self.masked:
            return np.ma.masked_where(x * x + y * y >= 1, z)
        else:
            return z


class ZernikeVector(VectorField):
    def __init__(self, n, m, masked=True):
        pass

#    @staticmethod
