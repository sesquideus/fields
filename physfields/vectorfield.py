import numpy as np
import numbers
import matplotlib
from scipy.interpolate import griddata

from matplotlib import pyplot as plt, colors as clr

from .scalarfield import ScalarField


class VectorField():
    def __init__(self, function):
        self.function = (lambda x, y: (0, 0)) if function is None else function

    @staticmethod
    def from_uv(u, v):
        """ Construct from functions u(x, y) and v(x, y) """
        return __class__(lambda x, y: (u(x, y), v(x, y)))

    def __call__(self, x, y):
        return self.function(x, y)

    def __add__(self, other):
        return VectorField(lambda x, y: self.function(x, y) + other.function(x, y))

    def __sub__(self, other):
        return VectorField(lambda x, y: self.function(x, y) - other.function(x, y))

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return VectorField(lambda x, y: (other * self.function(x, y)[0], other * self.function(x, y)[1]))
        elif isinstance(other, ScalarField):
            return VectorField(lambda x, y: (self.function(x, y)[0] * other.function(x, y), self.function(x, y)[1] * other.function(x, y)))
        elif isinstance(other, VectorField):
            return ScalarField(lambda x, y: self.function(x, y)[0] * other.function(x, y)[0] + self.function(x, y)[1] * other.function(x, y)[1])

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return VectorField(lambda x, y: (other * self.function(x, y)[0], other * self.function(x, y)[1]))
        else:
            raise NotImplementedError

    def plot(self, x, y, *, file=None, limits=None, mask=None):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.tight_layout()

        u, v = self(x, y)
        if mask is not None:
            u = np.ma.masked_where(mask(x, y), u)
            v = np.ma.masked_where(mask(x, y), v)

        ax.quiver(x, y, u, v, np.arctan2(v, u), norm=clr.Normalize(vmin=-np.pi, vmax=np.pi), cmap='hsv', pivot='middle')
        ax.set_aspect('equal')

        if limits is not None:
            ((xmin, xmax), (ymin, ymax)) = limits
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        if file is None:
            matplotlib.use('TkAgg')
            plt.show()
        else:
            matplotlib.use('Agg')
            fig.savefig(file, dpi=100)


class SampledVectorField():
    def __init__(self, x, y, u, v):
        self.x, self.y = x, y
        self.u, self.v = u, v

    # Construct from a set of (x, y) -> (u, v) placed vectors
    @staticmethod
    def from_vectors(x, y, vectors):
        return SampledVectorField(x, y,
            griddata(vectors[:, 0, :], vectors[:, 1, 0], (x, y), method='cubic'),
            griddata(vectors[:, 0, :], vectors[:, 1, 1], (x, y), method='cubic')
        )

    @staticmethod
    def from_abstract(x, y, abstract):
        u, v = abstract(x, y)
        return SampledVectorField(x, y, u, v)

    def plot(self, color='black'):
        plt.quiver(self.x, self.y, self.u, self.v, width=2e-3, color=color)
        plt.gca().set_aspect('equal')
        plt.get_current_fig_manager().window.attributes('-fullscreen', True)

    def __add__(self, other):
        return SampledVectorField(self.x, self.y, self.u + other.u, self.v + other.v)

    def __neg__(self):
        return SampledVectorField(self.x, self.y, -self.u, -self.v)

    def __iadd__(self, other):
        self.u += other.u
        self.v += other.v
        return self

    def __sub__(self, other):
        return SampledVectorField(self.x, self.y, self.u - other.u, self.v - other.v)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return SampledVectorField(self.x, self.y, other * self.u, other * self.v)
        elif isinstance(other, ScalarField):
            if self.x == other.x and self.y == other.y:
                return SampledVectorField(self.x, self.y, self.u * other.z, self.v * other.z)
            else:
                raise ValueError(f"Domains do not agree ({self.x}×{self.y} vs {other.x}×{other.y}")
        elif isinstance(other, SampledVectorField):
            #if self.x == other.x and self.y == other.y:
            return ScalarField(self.x, self.y, self.u * other.u + self.v * other.v)
         #   else:
          #      raise ValueError(f"Domains do not agree ({self.x}×{self.y} vs {other.x}×{other.y}")

    def __rmul__(self, other):
        return self * other
