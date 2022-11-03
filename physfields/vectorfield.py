import numpy as np
import numbers
import inspect
import matplotlib as mpl
from scipy.interpolate import griddata

from matplotlib import pyplot as plt

from .scalarfield import ScalarField, SampledScalarField


class VectorField():
    @staticmethod
    def UnitDisk(x, y):
        return x**2 + y**2 > 1

    def __init__(self, function, name=""):
        self.function = (lambda x, y: (np.zeros_like(x), np.zeros_like(y))) if function is None else function
        self.name = name

    @staticmethod
    def from_uv(u, v):
        """ Construct from functions u(x, y) and v(x, y) """
        return __class__(lambda x, y: (u(x, y), v(x, y)))

    @staticmethod
    def from_rt(rho, tau):
        """ Construct from functions rho(r, phi), tau(r, phi) """
        def fun(x, y):
            r = np.sqrt(x**2 + y**2)
            phi = (2 * np.pi + np.arctan2(y, x)) % (2 * np.pi)
            erho = rho(r, phi)
            etau = tau(r, phi)
            return erho * np.cos(etau), erho * np.sin(etau)
        return __class__(fun)

    def __call__(self, x, y):
        return self.function(x, y)

    def eval(self, x):
        return np.stack(self.__call__(x[:, 0], x[:, 1]), axis=1)

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
        else:
            raise NotImplemented("Can only multiply vector field with scalars, scalar fields and vector fields")

    def __matmul__(self, other):
        if isinstance(other, VectorField):
            return ScalarField(lambda x, y: self.function(x, y)[0] * other.function(x, y)[1] - self.function(x, y)[1] * other.function(x, y)[0])
        else:
            raise NotImplemented("Can only @-multiply vector field with another vector field")

    def __rmul__(self, other):
        return self * other

    def __rmatmul__(self, other):
        return other @ self

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return VectorField(lambda x, y: (self.function(x, y)[0] / other, self.function(x, y)[1] / other))
        else:
            raise NotImplemented("Can only divide vector field by scalars or scalar felds")

    def plot(self, x, y, *, file=None, limits=None, mask=None, colour=None, **kwargs):
        print(f"Plotting to {file}")
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 10)))
        fig.tight_layout()

        ax.set_facecolor(kwargs.get('face_colour', 'white'))
        ax.set_aspect('equal')
        self.plot_data(ax, x, y, mask=mask, colour=colour, **kwargs)

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

    def plot_data(self, ax, x, y, *, mask=None, colour=None, **kwargs):
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
        elif colour == 'div':
            clr = self.div(x, y)
            norm = mpl.colors.TwoSlopeNorm(0)
            cmap = kwargs.get('cmap', 'bwr')
        elif colour == 'rot':
            clr = self.rot(x, y)
            norm = mpl.colors.TwoSlopeNorm(0)
            cmap = kwargs.get('cmap', 'bwr')
        else:                                               # Default: pass-through
            clr = colour

        ax.quiver(x, y, u, v, clr, norm=norm, cmap=cmap, pivot='middle')

    def div(self, x, y, epsilon=0.01):
        """
            Numerically evaluate the divergence of the vector field at meshgrid (x, y)
            with precision epsilon
        """
        u1, _ = self.__call__(x + epsilon, y)
        u2, _ = self.__call__(x - epsilon, y)
        _, v1 = self.__call__(x, y + epsilon)
        _, v2 = self.__call__(x, y - epsilon)
        return ((u2 - u1) + (v2 - v1)) / (2 * epsilon)

    def rot(self, x, y, epsilon=0.01):
        """
            Numerically evaluate the curl of the vector field at meshgrid (x, y)
            With precision epsilon as (x + e, y) - (x - e, y) + ()
        """
        u1, _ = self.__call__(x, y + epsilon)
        u2, _ = self.__call__(x, y - epsilon)
        _, v1 = self.__call__(x + epsilon, y)
        _, v2 = self.__call__(x - epsilon, y)
        return ((u2 - u1) - (v2 - v1)) / (2 * epsilon)


class TrueVectorField():
    def __init__(self, function, name=""):
        self.function = (lambda x: np.zeros_like(x)) if function is None else function
        self.name = name

    def __call__(self, x, y):
        return self.function(x, y)



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
            if np.array_equal(self.x, other.x) and np.array_equal(self.y, other.y):
                return SampledScalarField(self.x, self.y, self.u * other.u + self.v * other.v)
            else:
                raise ValueError(f"Domains do not agree ({self.x}×{self.y} vs {other.x}×{other.y}")

    def __rmul__(self, other):
        return self * other
