import numpy as np
from .base import InterpolatorBase

_2pi = 2*np.pi

class CartesianToSpherical(InterpolatorBase):
    """
    Wrapper class that takes cartesian coordinates, and calls the underlying
    interpolator with converted spherical coordinates.

    The polar coordinate handed to the wrapped interpolator is either the
    polar angle ``theta`` (``polar="theta"``, the default) or its cosine
    ``mu = cos(theta)`` (``polar="mu"``).  Which one to use is a property of
    the data's grid: the interpolators require the second axis to be
    uniformly spaced, so a grid built uniform in ``cos(theta)`` (as
    AthenaK's spherical output is) must be interpolated in ``mu``.
    """

    def __init__(self, interpolator: type[InterpolatorBase], *args,
                 polar: str = 'theta', **kwargs):
        if polar not in ('theta', 'mu'):
            raise ValueError(f"polar must be 'theta' or 'mu', got {polar!r}.")
        self.polar = polar
        self.interpolator = interpolator(*args, **kwargs)

    def _polar_coord(self, cos_theta: np.ndarray) -> np.ndarray:
        """Map ``cos(theta)`` to whichever polar coordinate the grid uses."""
        return cos_theta if self.polar == 'mu' else np.arccos(cos_theta)

    def __del__(self):
        # Guard against partially-constructed instances (e.g. during unpickling
        # with spawn) where __del__ may fire before interpolator is set.
        if hasattr(self, 'interpolator'):
            del self.interpolator

    def __getattr__(self, name):
        # Delegate unknown attribute lookups to the wrapped interpolator.
        # object.__getattribute__ avoids infinite recursion if self.interpolator
        # itself is not yet set (e.g. during __init__ or unpickling).
        try:
            interp = object.__getattribute__(self, 'interpolator')
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(interp, name)

    def __getstate__(self):
        return {'interpolator': self.interpolator, 'polar': self.polar}

    def __setstate__(self, state):
        self.interpolator = state['interpolator']
        self.polar = state['polar']

    def sort_tracers(self, tracers: np.ndarray) -> np.ndarray:
        """Sort tracers by their spherical (polar, phi) bin for cache locality."""
        positions = np.array([tr.positions[-1] for tr in tracers]).T  # (3, n)
        x, y, z = positions
        r = np.sqrt(x**2 + y**2 + z**2)
        safe_r = np.where(r > 0, r, 1.0)
        polar = self._polar_coord(z / safe_r)
        phi = (np.arctan2(y, x) + _2pi) % _2pi
        polar_bins = np.digitize(polar, self.interpolator._y_nodes)
        phi_bins = np.digitize(phi, self.interpolator._z_nodes)
        return tracers[np.lexsort((phi_bins, polar_bins))]

    def load(self, track: bool = True):
        self.interpolator.load(track=track)

    def unload(self):
        self.interpolator.unload()

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        x, y, z = coords
        r = np.sqrt(x**2 + y**2 + z**2)
        # Guard against division by zero at the origin
        safe_r = np.where(r > 0, r, 1.0)
        polar = self._polar_coord(z / safe_r)
        # account for scaling [0,2pi] instead of [-pi,pi]
        phi = (np.arctan2(y, x) + _2pi) % _2pi
        sph_coords = np.array([r, polar, phi])
        return self.interpolator(sph_coords)
