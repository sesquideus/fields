import numpy as np
import collections


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
        Lazily evaluated scalar Zernike polynomial
    """

    def __init__(self, n, l, masked=False):
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


class ZernikeVector():
    v = collections.OrderedDict()

    v[0,  0]        = VectorField.from_uv(ScalarField(), ScalarField())                                             # S_1
    z = v[0, 0] # temporarily occupied

    v[1, -1]        = VectorField.from_uv(Zernike(0, 0), ScalarField())                                             # S_2 = (  Z_1,    0) =  S_3
    v[1,  1]        = VectorField.from_uv(ScalarField(), Zernike(0, 0))                                             # S_3 = (    0,  Z_1) = -T_2

    v[2, -2]        = SROH * VectorField.from_uv(Zernike(1, -1), Zernike(1, 1))                                     # S_5 = (  Z_2,  Z_3) = -T_6
    v[2,  0, False] = SROH * VectorField.from_uv(Zernike(1, 1), Zernike(1, -1))                                     # S_4 = (  Z_3,  Z_2)
    v[2,  0, True]  = SROH * VectorField.from_uv(Zernike(1, -1), -Zernike(1, 1))                                    #       (  Z_3, -Z_2) =  T_4
    v[2,  2]        = SROH * VectorField.from_uv(Zernike(1, 1), -Zernike(1, -1))                                    # S_6 = (  Z_2, -Z_3) =  T_5

    v[3, -3]        = SROH * VectorField.from_uv(Zernike(2, -2), Zernike(2, 2))                                     # S_9 = (  Z_5,  Z_6) = -T_10
    v[3, -1, False] =  0.5 * VectorField.from_uv(Zernike(2, -2), SR2 * Zernike(2, 0) - Zernike(2, 2))               # S_7 = (  Z_5, sqrt(2)*Z_4 - Z_6)
    v[3, -1, True]  =  0.5 * VectorField.from_uv(SR2 * Zernike(2, 0) - Zernike(2, 2), -Zernike(2, -2))
    v[3,  1, False] =  0.5 * VectorField.from_uv(SR2 * Zernike(2, 0) + Zernike(2, 2), Zernike(2, -2))               # S_8 = (sqrt(2) Z_4 + Z_6, Z_5)
    v[3,  1, True]  =  0.5 * VectorField.from_uv(Zernike(2, -2), -SR2 * Zernike(2, 0) - Zernike(2, 2))
    v[3,  3]        = SROH * VectorField.from_uv(Zernike(2, 2), -Zernike(2, -2))                                    # S_10 = ( Z_6, -Z_5) = T_9

    v[4, -4]        = SROH * VectorField.from_uv(Zernike(3, -3), Zernike(3, 3))
    v[4, -2, False] =  0.5 * VectorField.from_uv(Zernike(3, 3) + Zernike(3, 1), -Zernike(3, -1) + Zernike(3, -3))
    v[4, -2, True]  =  0.5 * VectorField.from_uv(-Zernike(3, -1) + Zernike(3, -3), -Zernike(3, 1) - Zernike(3, 3))
    v[4,  0, False] = SROH * VectorField.from_uv(Zernike(3, 1), Zernike(3, -1))
    v[4,  0, True]  = SROH * VectorField.from_uv(Zernike(3, -1), -Zernike(3, 1))
    v[4,  2, False] =  0.5 * VectorField.from_uv(Zernike(3, -3) + Zernike(3, -1), Zernike(3, 1) - Zernike(3, 3))
    v[4,  2, True]  =  0.5 * VectorField.from_uv(Zernike(3, 1) - Zernike(3, 3), -Zernike(3, -1) - Zernike(3, -3))
    v[4,  4]        = SROH * VectorField.from_uv(Zernike(3, 3), -Zernike(3, -3))

    v[5, -5]        = SROH * VectorField.from_uv(Zernike(4, -4), Zernike(4, 4))
    v[5, -3, False] = z
    v[5, -3, True]  = z
    v[5, -1, False] = z
    v[5, -1, True]  =  0.5 * VectorField.from_uv(SR2 * Zernike(4, 0) - Zernike(4, 2), -Zernike(4, -2))
    v[5,  1, False] =  0.5 * VectorField.from_uv(SR2 * Zernike(4, 0) + Zernike(4, 2), Zernike(4, -2))
    v[5,  1, True]  = z
    v[5,  3, False] = z
    v[5,  3, True]  = z
    v[5,  5]        = SROH * VectorField.from_uv(Zernike(4, 4), -Zernike(4, -4))

    v[6, -6]        = SROH * VectorField.from_uv(Zernike(5, -5), Zernike(5, 5))
    v[6, -4, False] =  0.5 * VectorField.from_uv(Zernike(5, 5) + Zernike(5, 3), -Zernike(5, -3) + Zernike(5, -5))
    v[6, -4, True]  =  0.5 * VectorField.from_uv(-Zernike(5, -3) + Zernike(5, -5), -Zernike(5, 3) - Zernike(5, 5))
    v[6, -2, False] =  0.5 * VectorField.from_uv(Zernike(5, 3) + Zernike(5, 1), -Zernike(5, -1) + Zernike(5, -3))
    v[6, -2, True]  =  0.5 * VectorField.from_uv(-Zernike(5, -1) + Zernike(5, -3), -Zernike(5, 1) - Zernike(5, 3))
    v[6,  0, False] = SROH * VectorField.from_uv(Zernike(5, 1), Zernike(5, -1))
    v[6,  0, True]  = SROH * VectorField.from_uv(Zernike(5, -1), -Zernike(5, 1))
    v[6,  2, False] =  0.5 * VectorField.from_uv(Zernike(5, -3) + Zernike(5, -1), Zernike(5, 1) - Zernike(5, 3))
    v[6,  2, True]  =  0.5 * VectorField.from_uv(Zernike(5, 1) - Zernike(5, 3), -Zernike(5, -1) - Zernike(5, -3))
    v[6,  4, False] =  0.5 * VectorField.from_uv(Zernike(5, -5) + Zernike(5, -3), Zernike(5, 3) - Zernike(5, 5))
    v[6,  4, True]  =  0.5 * VectorField.from_uv(Zernike(5, 3) - Zernike(5, 5), -Zernike(5, -3) - Zernike(5, -5))
    v[6,  6]        = SROH * VectorField.from_uv(Zernike(5, 5), -Zernike(5, -5))

    @classmethod
    def create(cls, n, l, rotational=None, masked=True):
        if abs(l) > n or (l + n) % 2 != 0:
            raise NotImplementedError("|l| must be < n and also l = n (mod 2)")
        if abs(l) == n and rotational is not None:
            raise NotImplementedError("Polynomials with |l| = n are always Laplacian")
        if abs(l) != n and rotational is None:
            raise NotImplementedError("Polynomials with |l| != n have to be rotational or diverging")

        if rotational is None:
            return cls.v[n, l]
        else:
            return cls.v[n, l, rotational]

    @classmethod
    def from_index(cls, index):
        pass

#    @staticmethod
