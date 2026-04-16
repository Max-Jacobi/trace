import numpy as np
from .base import InterpolatorBase

class CartesianToSpherical[Interpolator: type[InterpolatorBase]](InterpolatorBase):
    """
    Wrapper class that takes cartesian coordinates, and calls the underlying
    interpolator with converted spherical coordinates.
    """

    def __init__(self, interpolator: Interpolator, *args, **kwargs):
        self.interpolator = interpolator(*args, **kwargs)

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        x, y, z = coords
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        sph_coords = np.array([r, theta, phi])
        return self.interpolator(sph_coords)
