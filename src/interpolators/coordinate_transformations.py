import numpy as np
from .base import InterpolatorBase

class CartesianToSpherical(InterpolatorBase):
    """
    Wrapper class that takes cartesian coordinates, and calls the underlying
    interpolator with converted spherical coordinates.
    """

    def __init__(self, interpolator: type[InterpolatorBase], *args, **kwargs):
        self.interpolator = interpolator(*args, **kwargs)

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        x, y, z = coords
        r = np.sqrt(x**2 + y**2 + z**2)
        # Guard against division by zero at the origin
        safe_r = np.where(r > 0, r, 1.0)
        cos_theta = np.clip(z / safe_r, -1.0, 1.0)
        theta = np.where(r > 0, np.arccos(cos_theta), 0.0)
        phi = np.arctan2(y, x)
        sph_coords = np.array([r, theta, phi])
        return self.interpolator(sph_coords)
