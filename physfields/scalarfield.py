import numpy as np
import numbers

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors


class ScalarField():
    def __init__(self, function=None):
        self.function = (lambda x, y: np.zeros_like(x)) if function is None else function

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

    def plot_image(self, x, y, *, file=None, limits=None, mask=None, colour=None, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 10)))
        fig.tight_layout()

        if mask is not None:
            u = np.ma.masked_where(mask(x, y), x)
            v = np.ma.masked_where(mask(x, y), y)
            x, y = u, v

        if limits is not None:
            ((xmin, xmax), (ymin, ymax)) = limits
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        if file is None:
            plt.show()
        else:
            fig.switch_backend('Agg')
            fig.savefig(file, dpi=100)

    def plot_3D(self, x, y, *, colour=None, file=None, limits=None, **kwargs):
        fig = plt.figure(figsize=kwargs.get('figsize', (10, 10)))
        ax = plt.axes(projection='3d')
        fig.tight_layout()

        norm = None
        cmap = None
        if colour is None:                                  # No colour
            clr = np.zeros_like(u)
        elif isinstance(colour, ScalarField):               # Colour by scalar field, evaluate automatically
            clr = colour(x, y)
            norm = kwargs.get('norm', mpl.colors.TwoSlopeNorm(0))
            cmap = kwargs.get('cmap', 'bwr')                # Default cmap: bwr
        else:                                               # Default: pass-through
            clr = colour

        if limits is not None:
            ((xmin, xmax), (ymin, ymax)) = limits
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        ax.set_box_aspect([1, 1, 1])
        cmap = mpl.cm.get_cmap(cmap)
        ax.plot_surface(x, y, self(x, y), facecolors=cmap(norm(clr)), norm=norm, cmap=cmap, rstride=1, cstride=1)

        if file is None:
            mpl.use('TkAgg')
            plt.show()
        else:
            mpl.use('Agg')
            fig.savefig(file, dpi=100)


class TrueScalarField():
    def __init__(self, function=None):
        self.function = (lambda x, y: np.zeros_like(x)) if function is None else function

    def __call__(self, points):
        """
            Evaluate the scalar field at selected points
            points: 2D array of shape (n, d)
        """
        return self.function(points)



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


