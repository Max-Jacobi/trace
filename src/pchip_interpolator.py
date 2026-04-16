"""
Module for Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation in uniform 3D grids.

.. deprecated::
    This module is a legacy stand-alone file and is superseded by
    ``src.interpolators.pchip.PchipInterpolator3D`` and
    ``src.interpolators.linear.LinearInterpolator3D``.
    It will be removed in a future release.  Import from ``src.interpolators`` instead.
"""

import warnings
warnings.warn(
    "src.pchip_interpolator is deprecated and will be removed in a future release. "
    "Use src.interpolators.PchipInterpolator3D or src.interpolators.LinearInterpolator3D instead.",
    DeprecationWarning,
    stacklevel=2,
)

from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from collections.abc import Callable

class LinearInterpolator3D(Callable):
    """
      Simple wrapper around scipy RegularGridInterpolator for Cartesian grids.
        Expects data with ghost zones included.
    """
    n_ghosts = (1, 1, 1)

    def __init__(self, x, y, z, d):
        d0 = d[next(iter(d))] # assume all d have same dtype and shape
        self.dtype = d0.dtype
        self.nx, self.ny, self.nz = d0.shape
        self.keys = list(d.keys())

        if x.shape[0] != self.nx or y.shape[0] != self.ny or z.shape[0] != self.nz:
            raise ValueError("x/y/z lengths must match d.shape")

        self.interp_cache = {}
        for k, dd in d.items():
            self.interp_cache[k] = RegularGridInterpolator((x, y, z), dd, bounds_error=False, fill_value=np.nan)

    def __call__(self, xi, yi, zi):
        xi = np.asarray(xi); yi = np.asarray(yi); zi = np.asarray(zi)
        assert xi.shape == yi.shape == zi.shape
        out = {k: np.empty_like(xi, dtype=self.dtype) for k in self.keys}
        flat_xi = xi.ravel(); flat_yi = yi.ravel(); flat_zi = zi.ravel()
        out_flat = {k: out[k].ravel() for k in self.keys}

        points = np.array([flat_xi, flat_yi, flat_zi]).T
        for key, interp in self.interp_cache.items():
            out_flat[key] = interp(points)
        return out

class PchipInterpolator3D(Callable):
    """
    3D PCHIP interpolator for Cartesian grids.
    Expects data with ghost zones included.
    """
    n_ghosts = (2, 2, 2)

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, d: dict[str, np.ndarray]):
        x = np.asarray(x);
        y = np.asarray(y)
        z = np.asarray(z)
        d = {k: np.asarray(d) for k, d in d.items()}

        self.keys = list(d.keys())

        d0 = d[self.keys[0]] # assume all d have same dtype and shape
        self.dtype = d0.dtype
        self.nx, self.ny, self.nz = d0.shape

        if x.shape[0] != self.nx or y.shape[0] != self.ny or z.shape[0] != self.nz:
            raise ValueError("x/y/z lengths must match d.shape")

        self.dx = float(x[1] - x[0])
        self.dy = float(y[1] - y[0])
        self.dz = float(z[1] - z[0])
        self.x0 = float(x[0])
        self.y0 = float(y[0])
        self.z0 = float(z[0])

        self.xp_cache = {}
        for jy in range(self.ny):
            for kz in range(self.nz):
                self.xp_cache[(jy,kz)] = {k: PchipInterpolator(x, dd[:, jy, kz], extrapolate=True) for k, dd in d.items()}

    @staticmethod
    def cell_index(xq, x0, dx, n):
        "helper: find cell index i so x[i] <= xq < x[i+1]"
        frac = (xq - x0) / dx
        ix = int(np.floor(frac))
        if ix < 0: ix = 0
        if ix >= n-1: ix = n-2
        return ix

    def __call__(self, coords: tuple[np.ndarray, ...]) -> dict[str, np.ndarray]:
        xi, yi, zi = coords
        xi = np.asarray(xi); yi = np.asarray(yi); zi = np.asarray(zi)
        assert xi.shape == yi.shape == zi.shape
        out = {k: np.empty_like(xi, dtype=self.dtype) for k in self.xp_cache[(0,0)].keys()}
        flat_xi = xi.ravel(); flat_yi = yi.ravel(); flat_zi = zi.ravel()
        out_flat = {k: out[k].ravel() for k in out.keys()}

        for k in range(flat_xi.size):
            xq = float(flat_xi[k]); yq = float(flat_yi[k]); zq = float(flat_zi[k])
            ix = self.cell_index(xq, self.x0, self.dx, self.nx)
            iy = self.cell_index(yq, self.y0, self.dy, self.ny)
            iz = self.cell_index(zq, self.z0, self.dz, self.nz)

            ix0 = ix - 1; iy0 = iy - 1; iz0 = iz - 1
            if ix0 < 0 or ix0+3 >= self.nx or iy0 < 0 or iy0+3 >= self.ny or iz0 < 0 or iz0+3 >= self.nz:
                raise IndexError("Stencil out of bounds; ensure you have ghost layers")

            for key in self.keys:
                V = np.empty((4,4), dtype=self.dtype)
                for jy_idx in range(4):
                    jy = iy0 + jy_idx
                    for kz_idx in range(4):
                        kz = iz0 + kz_idx
                        interp = self.xp_cache[(jy, kz)][key]
                        V[jy_idx, kz_idx] = interp(xq)

                W = np.empty(4, dtype=self.dtype)
                y_nodes = self.y0 + self.dy * np.arange(iy0, iy0+4)
                for kz_idx in range(4):
                    col = V[:, kz_idx]
                    py = PchipInterpolator(y_nodes, col, extrapolate=True)
                    W[kz_idx] = py(yq)

                z_nodes = self.z0 + self.dz * np.arange(iz0, iz0+4)
                pz = PchipInterpolator(z_nodes, W, extrapolate=True)
                out_flat[key][k] = pz(zq)
        return out
