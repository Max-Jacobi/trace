"""
3D PCHIP interpolator for uniform Cartesian grids.
Vectorized query evaluation + lazy caching of x-interpolators.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from collections import defaultdict

from .base import InterpolatorBase, OutOfBoundsError

class PchipInterpolator3D(InterpolatorBase):
    n_ghosts = 3

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

        d = {k: np.asarray(v) for k, v in d.items()}

        self.keys = list(d.keys())

        d0 = d[self.keys[0]]
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

        self._x_nodes = np.asarray(x)
        self._y_nodes_full = np.asarray(y)
        self._z_nodes_full = np.asarray(z)

        self._data = d
        self._xp_cache: dict[tuple[int, int, str], PchipInterpolator] = {}

    @staticmethod
    def cell_index(xq, x0, dx, n):
        frac = (xq - x0) / dx
        ix = int(np.floor(frac))
        if ix < 0:
            ix = 0
        if ix >= n - 1:
            ix = n - 2
        return ix

    def _get_x_interp(self, jy: int, kz: int, key: str):
        cache_key = (jy, kz, key)
        interp = self._xp_cache.get(cache_key)
        if interp is None:
            yy = self._data[key][:, jy, kz]
            interp = PchipInterpolator(self._x_nodes, yy, axis=0, extrapolate=True)
            self._xp_cache[cache_key] = interp
        return interp

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
        flat_xi = xi.ravel()
        flat_yi = yi.ravel()
        flat_zi = zi.ravel()
        n_points = flat_xi.size

        # Initialise output with NaN so out-of-domain points are returned as NaN
        out = [np.full(n_points, np.nan, dtype=self.dtype) for _ in self.keys]

        if n_points == 0:
            out_arr = np.asarray([o.reshape(xi.shape) for o in out])
            return out_arr

        frac_y = (flat_yi - self.y0) / self.dy
        frac_z = (flat_zi - self.z0) / self.dz

        iy = np.floor(frac_y).astype(int)
        iz = np.floor(frac_z).astype(int)

        iy0 = iy - 1
        iz0 = iz - 1

        groups = defaultdict(list)
        for idx in range(n_points):
            iy0_val = int(iy0[idx])
            iz0_val = int(iz0[idx])
            # Skip points whose 4-point stencil falls outside the grid
            if iy0_val < 0 or iy0_val + 4 > self.ny or iz0_val < 0 or iz0_val + 4 > self.nz:
                continue
            groups[(iy0_val, iz0_val)].append(idx)

        for (iy0_val, iz0_val), idx_list in groups.items():
            idx_arr = np.array(idx_list, dtype=int)
            xq = flat_xi[idx_arr]
            yq = flat_yi[idx_arr]
            zq = flat_zi[idx_arr]

            y_indices = np.arange(iy0_val, iy0_val + 4)
            z_indices = np.arange(iz0_val, iz0_val + 4)
            y_nodes = self.y0 + self.dy * y_indices
            z_nodes = self.z0 + self.dz * z_indices

            for k_i, key in enumerate(self.keys):
                n_group = xq.size
                V = np.empty((4, 4, n_group), dtype=self.dtype)

                for jy_idx, jy in enumerate(y_indices):
                    for kz_idx, kz in enumerate(z_indices):
                        interp_x = self._get_x_interp(jy, kz, key)
                        V[jy_idx, kz_idx, :] = interp_x(xq)

                W = np.empty((4, n_group), dtype=self.dtype)
                for kz_idx in range(4):
                    col = V[:, kz_idx, :]  # (4, n_group)
                    py = PchipInterpolator(y_nodes, col, axis=0, extrapolate=False)
                    vals = py(yq)
                    W[kz_idx, :] = np.diag(vals)

                pz = PchipInterpolator(z_nodes, W, axis=0, extrapolate=False)
                vals_z = pz(zq)
                final_vals = np.diag(vals_z)

                out[k_i][idx_arr] = final_vals

        out = [o.reshape(xi.shape) for o in out]
        return np.asarray(out)
