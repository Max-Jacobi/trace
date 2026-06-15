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

    def __init__(
        self,
        *coords: np.ndarray,
        shm: dict[str, str],
        shape: tuple[int, int, int],
        log_coords: list[int] = [],
        max_cache_size_GB: float = 0.5,
        ):
        super().__init__(shm_names=shm, shape=shape)

        self.log_coords = log_coords
        self.max_cache_size_bytes = int(max_cache_size_GB * 1024**3)

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

        self.x0 = x[0]
        self.y0 = y[0]
        self.z0 = z[0]
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dz = z[1] - z[0]

        self._x_nodes = np.asarray(x)
        self._y_nodes = np.asarray(y)
        self._z_nodes = np.asarray(z)

        self._xp_cache: dict[tuple[int, int], PchipInterpolator] = {}
        self.n_evicted = 0

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

            while self._estimate_cached_memory() > self.max_cache_size_bytes:
                #print("PCHIP cache exceeded max size; evicting oldest entry.")
                self.n_evicted += 1
                oldest_key = next(iter(self._xp_cache))
                del self._xp_cache[oldest_key]
        return interp

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        if not self.loaded:
            raise RuntimeError("Interpolator data not loaded; call load() before querying.")

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


    def _estimate_cached_memory(self) -> int:
        total_bytes = 0
        for interp in self._xp_cache.values():
            total_bytes += interp.c.nbytes + interp.x.nbytes
        return total_bytes

    def sort_tracers(self, tracers: np.ndarray) -> np.ndarray:
        """
          Sort tracers by their y and z coordinates to improve cache locality of
          interpolator access. Specifically, digitise the y and z coordinates
          into bins corresponding to the yz grid cells, and sort tracers by
          their (y_bin, z_bin) pairs. This way, tracers that require the same
          x-interpolator will be grouped together, improving cache hits when
          evaluating the interpolator for multiple tracers in the same yz cell.
        """
        y_coords = np.array([tr.positions[-1][1] for tr in tracers])
        z_coords = np.array([tr.positions[-1][2] for tr in tracers])
        y_bins = np.digitize(y_coords, self._y_nodes)
        z_bins = np.digitize(z_coords, self._z_nodes)
        sort_indices = np.lexsort((z_bins, y_bins))
        return tracers[sort_indices]
