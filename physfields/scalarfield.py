import numpy as np
import numbers

from matplotlib import pyplot as plt
from matplotlib import colors


class ScalarField():
    def __init__(self, function=None):
        self.function = (lambda x, y: 0) if function is None else function

    def __call__(self, x, y):
        return self.function(x, y)

    def __neg__(self):
        return ScalarField(lambda x, y: -self.function(x, y))

    def __add__(self, other):
        return ScalarField(lambda x, y: self.function(x, y) + other.function(x, y))

    def __sub__(self, other):
        return ScalarField(lambda x, y: self.function(x, y) - other.function(x, y))

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return ScalarField(lambda x, y: other * self.function(x, y))
        else:
            return ScalarField(lambda x, y: self.function(x, y) * other.function(x, y))

    def __rmul__(self, value):
        if isinstance(value, numbers.Number):
            return ScalarField(lambda x, y: value * self.function(x, y))
        else:
            raise NotImplementedError

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return ScalarField(lambda x, y: self.function(x, y) / other)
        else:
            return ScalarField(lambda x, y: self.function(x, y) / other.function(x, y))

    def plot(self, x, y):
        plt.imshow(self(x, y), extent=[x[0][0], x[0][-1], x[-1][0], x[-1][-1]])
        plt.gca().set_aspect('equal')
        plt.get_current_fig_manager().window.attributes('-fullscreen', True)


class SampledScalarField():
    def __init__(self, x, y, z=None):
        self.x, self.y = x, y
        self.z = np.zeros((len(x), len(y))) if z is None else z

    def __add__(self, other):
        return SampledScalarField(self.x, self.y, self.z + other.z)

    def __mul__(self, value):
        return SampledScalarField(self.x, self.y, self.z * value)

    def __rmul__(self, value):
        return self * value

    def plot(self):
        scale = self.z.shape[0]
        x = 1 + 1 / scale
        plt.imshow(self.z, extent=[-x, x, x, -x], cmap='bwr', norm=colors.TwoSlopeNorm(vcenter=0))


