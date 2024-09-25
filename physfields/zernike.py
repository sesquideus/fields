import numpy as np
import scipy as sp

from typing import Optional

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


SR2 = np.sqrt(2)
SROH = 1 / np.sqrt(2)


class Zernike(ScalarField):
    """
    A lazily evaluated scalar Zernike polynomial
    """

    def __init__(self, n, l, *, masked=False):
        """
        A new Zernike scalar field defined in terms of radial and angular component order n and l:
        n: int, non-negative radial component order
        l: int, angular component order, |l| < n and l = n (mod 2)
        masked: mask the area outside the unit disk
        """
        if abs(l) > n or (l + n) % 2 != 0:
            raise ValueError("|l| must be <= n and also l = n (mod 2)")

        self.n = n
        self.l = l
        self.absl = abs(l)
        self.masked = masked

        # Set noll and ANSI indices for convenience
        self.noll = n * (n + 1) // 2 + abs(l) + (1 if (l >= 0 and n % 4 in [2, 3]) or (l <= 0 and n % 4 in [0, 1]) else 0)
        self.ansi = (n * (n + 2) + l) / 2
        super().__init__(lambda x, y: np.sqrt(self.n + 1) * self.radial_part(x, y) * self.angular_part(x, y))

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
        Static constructor from Noll index
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
            (-1)**s * sp.special.factorial(self.n - s) /
            (sp.special.factorial(s) * sp.special.factorial((self.n + self.absl) // 2 - s) * sp.special.factorial((self.n - self.absl) // 2 - s))
            * r ** (self.n - 2 * s)
            for s in range(0, (self.n - self.absl) // 2 + 1)],
        )

    def angular_part(self, x, y):
        if self.l == 0:
            return 1
        elif self.l > 0:
            return np.sqrt(2) * np.cos(self.absl * np.arctan2(y, x))
        else:
            return np.sqrt(2) * np.sin(self.absl * np.arctan2(y, x))


class ZernikeVector(VectorField):
    def __init__(self, n: int, l: int, r: Optional[bool]=None, *, masked: bool=True):
        if abs(l) > n or (l + n) % 2 != 0:
            raise ValueError("|l| must be <= n and also l = n (mod 2)")
        if abs(l) == n and r is not None:
            raise ValueError("Polynomials with |l| = n are always Laplacian")
        if abs(l) != n and r is None:
            raise ValueError("Polynomials with |l| != n must be rotational or diverging")

        self.n = n
        self.l = l
        self.r = r

        if masked:
            mask = lambda x, y: x**2 + y**2 > 1
        else:
            mask = None

        if n == 0:
            super().__init__(ScalarField(), ScalarField(), mask=mask)
        elif n == 1:
            if l == -1:
                super().__init__(Zernike(0, 0), ScalarField(), mask=mask)
            else:
                super().__init__(ScalarField(), Zernike(0, 0), mask=mask)
        else:
            m = n - 1
            rot = -1 if r else 1

            if n == -l:
                super().__init__(SROH * Zernike(m, -m), SROH * Zernike(m, m), mask=mask)
            elif n == l:
                super().__init__(SROH * Zernike(m, m), -SROH * Zernike(m, -m), mask=mask)
            elif l == 0:
                super().__init__(SROH * Zernike(m, rot), (SROH * rot) * Zernike(m, -rot), mask=mask)
            elif abs(l) == 1:
                if l == -1:
                    super().__init__(0.5 * Zernike(m, -2), 0.5 * ((rot * SR2) * Zernike(m, 0) - Zernike(m, 2)), mask=mask)
                else:
                    super().__init__(0.5 * (SR2 * Zernike(m, 0) + rot * Zernike(m, 2)), (0.5 * rot) * Zernike(m, -2), mask=mask)
            else:
                super().__init__(0.5 * (Zernike(m, l - 1) + rot * Zernike(m, l + 1)), 0.5 * (rot * Zernike(m, -l - 1) - Zernike(m, -l + 1)), mask=mask)


    @classmethod
    def from_index(cls, index):
        pass

#    @staticmethod
