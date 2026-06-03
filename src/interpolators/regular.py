from scipy.interpolate import RegularGridInterpolator
import numpy as np

from .base import InterpolatorBase

class RegularInterpolator3D(InterpolatorBase):
    """
      Simple wrapper around scipy RegularGridInterpolator for Cartesian grids.
        Expects data with ghost zones included.
    """
    n_ghosts = 1

    def __init__(
        self,
        *coords: np.ndarray,
        shm: dict[str, str],
        shape: tuple[int, int, int],
        log_coords: list[int] = [],
        method: str = "linear",
        ):
        super().__init__(shm_names=shm, shape=shape)
        self.log_coords = log_coords
        self.method = method

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

        self.nx = x.shape[0]
        self.ny = y.shape[0]
        self.nz = z.shape[0]
        assert self.shape == (self.nx, self.ny, self.nz)

        self._x_nodes = np.asarray(x)
        self._y_nodes = np.asarray(y)
        self._z_nodes = np.asarray(z)

        if x.shape[0] != self.nx or y.shape[0] != self.ny or z.shape[0] != self.nz:
            raise ValueError("x/y/z lengths must match d.shape")

        self.interp_cache = {}

    def load(self):
        super().load()
        for k, dd in self.data.items():
            self.interp_cache[k] = RegularGridInterpolator(
                (self._x_nodes, self._y_nodes, self._z_nodes),
                dd,
                bounds_error=False,
                fill_value=np.nan,
                method=self.method,
            )

    def unload(self):
        super().unload()
        self.interp_cache.clear()

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
        out = [np.empty_like(xi) for _ in self.keys]
        flat_xi = xi.ravel(); flat_yi = yi.ravel(); flat_zi = zi.ravel()
        out_flat = [o.ravel() for o in out]

        points = np.array([flat_xi, flat_yi, flat_zi]).T
        for i, interp in enumerate(self.interp_cache.values()):
            out_flat[i][:] = interp(points)
        return np.asarray(out)
