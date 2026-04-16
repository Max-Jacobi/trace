from scipy.interpolate import RegularGridInterpolator
import numpy as np

from .base import InterpolatorBase

class LinearInterpolator3D(InterpolatorBase):
    """
      Simple wrapper around scipy RegularGridInterpolator for Cartesian grids.
        Expects data with ghost zones included.
    """
    n_ghosts = 1

    def __init__(self, *coords: np.ndarray, d: dict[str, np.ndarray], log_coords: list[int] = []):
        self.log_coords = log_coords

        transformed = []
        for i, c in enumerate(coords):
            c = np.asarray(c)
            if i in log_coords:
                if np.any(c <= 0):
                    raise ValueError(
                        f"Coordinate axis {i} contains non-positive values but log_coords={log_coords}; "
                        "log10 requires strictly positive inputs."
                    )
                transformed.append(np.log10(c))
            else:
                transformed.append(c)
        x, y, z = transformed

        d0 = d[next(iter(d))] # assume all d have same dtype and shape
        self.dtype = d0.dtype
        self.nx, self.ny, self.nz = d0.shape
        self.keys = list(d.keys())

        if x.shape[0] != self.nx or y.shape[0] != self.ny or z.shape[0] != self.nz:
            raise ValueError("x/y/z lengths must match d.shape")

        self.interp_cache = {}
        for k, dd in d.items():
            self.interp_cache[k] = RegularGridInterpolator((x, y, z), dd, bounds_error=False, fill_value=np.nan)

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        transformed = []
        for i, c in enumerate(coords):
            c = np.asarray(c)
            if i in self.log_coords:
                if np.any(c <= 0):
                    raise ValueError(
                        f"Query coordinate axis {i} contains non-positive values but log_coords={self.log_coords}."
                    )
                transformed.append(np.log10(c))
            else:
                transformed.append(c)
        xi, yi, zi = transformed
        if xi.shape != yi.shape or xi.shape != zi.shape:
            raise ValueError(
                f"Coordinate arrays must have identical shapes; got {xi.shape}, {yi.shape}, {zi.shape}."
            )
        out = [np.empty_like(xi, dtype=self.dtype) for _ in self.keys]
        flat_xi = xi.ravel(); flat_yi = yi.ravel(); flat_zi = zi.ravel()
        out_flat = [o.ravel() for o in out]

        points = np.array([flat_xi, flat_yi, flat_zi]).T
        for i, interp in enumerate(self.interp_cache.values()):
            out_flat[i][:] = interp(points)
        return np.asarray(out)
