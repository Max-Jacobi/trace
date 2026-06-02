"""
3D PCHIP interpolator for uniform Cartesian grids.
Vectorized query evaluation + lazy caching of x-interpolators.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from collections import defaultdict
from multiprocessing.shared_memory import SharedMemory

from .base import InterpolatorBase

class PchipInterpolator3D(InterpolatorBase):
    n_ghosts = 3

    def __init__(self, *coords: np.ndarray, shm: dict[str, str], shape: tuple[int, int, int], log_coords: list[int] = []):
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

        self.keys = list(shm.keys())
        self.n_keys = len(self.keys)

        self.nx = x.shape[0]
        self.ny = y.shape[0]
        self.nz = z.shape[0]
        self.shape = shape
        assert shape == (self.nx, self.ny, self.nz)

        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        self.dx = np.average(dx)
        self.dy = np.average(dy)
        self.dz = np.average(dz)
        assert np.all(np.isclose(self.dx, dx))
        assert np.all(np.isclose(self.dy, dy))
        assert np.all(np.isclose(self.dz, dz))
        self.x0 = float(x[0])
        self.y0 = float(y[0])
        self.z0 = float(z[0])

        self._x_nodes = np.asarray(x)
        self._y_nodes = np.asarray(y)
        self._z_nodes = np.asarray(z)


        self.shm_names = shm
        self.shm: dict[str, SharedMemory] = {}

        self._xp_cache: dict[tuple[int, int], PchipInterpolator] = {}

        self.data: dict[str, np.ndarray] = {}

    def load(self):
        for key in self.keys:
            shm = SharedMemory(name=self.shm_names[key])
            self.shm[key] = shm
            self.data[key] = np.ndarray(self.shape, dtype=np.float64, buffer=shm.buf)

    def __del__(self):
        for key, shm in self.shm.items():
            shm.close()
            del self.data[key]

    @staticmethod
    def cell_index(xq, x0, dx, n):
        frac = (xq - x0) / dx
        ix = int(np.floor(frac))
        if ix < 0:
            ix = 0
        if ix >= n - 1:
            ix = n - 2
        return ix

    def _get_x_interp(self, jy: int, kz: int):
        cache_key = (jy, kz)
        interp = self._xp_cache.get(cache_key)
        if interp is None:
            data = np.empty((self.n_keys, self.nx), dtype=np.float64)
            for i_k, key in enumerate(self.keys):
                data[i_k] = self.data[key][:, jy, kz]
            data = np.transpose(data)
            interp = PchipInterpolator(self._x_nodes, data, axis=0, extrapolate=False)
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
        out = np.full((self.n_keys, n_points), np.nan)

        if n_points == 0:
            out_arr = np.asarray([o.reshape(xi.shape) for o in out])
            return out_arr

        frac_y = (flat_yi - self.y0) / self.dy
        frac_z = (flat_zi - self.z0) / self.dz

        # Guard against NaN inputs (e.g. tracers that left the domain)
        nan_yz = np.isnan(frac_y) | np.isnan(frac_z)
        iy = np.where(nan_yz, 0, np.floor(frac_y)).astype(int)
        iz = np.where(nan_yz, 0, np.floor(frac_z)).astype(int)

        iy0 = iy - 1
        iz0 = iz - 1

        groups = defaultdict(list)
        for idx in range(n_points):
            if nan_yz[idx]:
                continue  # leave output as NaN
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

            n_group = xq.size
            V = np.empty((4, 4, n_group, self.n_keys))

            for jy_idx, jy in enumerate(y_indices):
                for kz_idx, kz in enumerate(z_indices):
                    interp_x = self._get_x_interp(jy, kz)
                    V[jy_idx, kz_idx] = interp_x(xq)

            final_vals = np.full((n_group, self.n_keys), np.nan)
            for j, (y, z) in enumerate(zip(yq, zq)):
                W = np.full((4, self.n_keys), np.nan)
                for kz_idx in range(4):
                    col = V[:, kz_idx, j, :]
                    if not np.isfinite(col).all():
                        continue
                    py = PchipInterpolator(y_nodes, col, extrapolate=False)
                    W[kz_idx] = py(y)
                if not np.isfinite(W).all():
                    continue
                pz = PchipInterpolator(z_nodes, W, extrapolate=False)
                final_vals[j] = pz(z)

            out[:, idx_arr] = final_vals.T # reshape from (n_group, n_keys) to (n_keys, n_group)

        out = [o.reshape(xi.shape) for o in out]
        return np.asarray(out)
