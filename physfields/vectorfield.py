import numpy as np
import numbers
import matplotlib as mpl
from scipy.interpolate import griddata

from matplotlib import pyplot as plt

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

    def plot(self, x, y, *, file=None, limits=None, mask=None, colour=None, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 10)))
        fig.tight_layout()

        ax.set_facecolor(kwargs.get('face_colour', 'white'))

        u, v = self(x, y)
        if mask is not None:
            u = np.ma.masked_where(mask(x, y), u)
            v = np.ma.masked_where(mask(x, y), v)

        norm = None
        cmap = None
        if colour is None:                                  # No colour
            clr = np.zeros_like(u)
        elif isinstance(colour, ScalarField):               # Colour by scalar field, evaluate automatically
            clr = colour(x, y)
            norm = mpl.colors.TwoSlopeNorm(0)
            cmap = kwargs.get('cmap', 'bwr')                # Default cmap: bwr
        elif colour == 'azimuth':                           # Colour by azimuth automatically
            clr = np.arctan2(v, u)
            norm = mpl.colors.Normalize(-np.pi, np.pi)
            cmap = kwargs.get('cmap', 'hsv')                # Default cmap: hsv
        else:                                               # Default: pass-through
            clr = colour

        ax.quiver(x, y, u, v, clr, norm=norm, cmap=cmap, pivot='middle')
        ax.set_aspect('equal')

        if limits is not None:
            ((xmin, xmax), (ymin, ymax)) = limits
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        if file is None:
            plt.switch_backend('TkAgg')
            plt.show()
        else:
            plt.switch_backend('Agg')
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
            if self.x == other.x and self.y == other.y:
                return ScalarField(self.x, self.y, self.u * other.u + self.v * other.v)
            else:
                raise ValueError(f"Domains do not agree ({self.x}×{self.y} vs {other.x}×{other.y}")

    def __rmul__(self, other):
        return self * other
