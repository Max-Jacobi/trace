"""
3D PCHIP interpolator for uniform Cartesian grids.
Vectorized query evaluation + lazy caching of x-interpolators.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from collections import defaultdict
from multiprocessing.shared_memory import SharedMemory

from .base import InterpolatorBase, transform_coords


def _pchip4_eval(nodes: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """
    Evaluate a monotone (Fritsch-Carlson) cubic Hermite interpolant through
    4 consecutive samples, at query points lying in ``[nodes[1], nodes[2]]``.

    This is the same Fritsch-Carlson derivative formula scipy's
    ``PchipInterpolator`` uses at interior nodes (see also
    ``src/integrators/rk4.py``'s ``_pchip_v_mid``, which is the same
    formula specialised to the interval midpoint for RK4's time axis), so
    results match scipy to floating-point precision.  The point of this
    function is to be fully vectorized over an arbitrary batch of points
    instead of constructing one ``scipy.interpolate.PchipInterpolator``
    object per point, which is the dominant cost of evaluating a 4-point
    stencil at many query points.

    Parameters
    ----------
    nodes : ndarray, shape (4,)
        The 4 grid coordinates the stencil is built on.
    values : ndarray, shape (4, *batch_shape)
        Sample values at ``nodes``, axis 0 paired with ``nodes``.
    query : ndarray, broadcastable against ``batch_shape``
        Coordinates to evaluate at (each lying in ``[nodes[1], nodes[2]]``).

    Returns
    -------
    ndarray, shape ``batch_shape``
        Interpolated values; NaN wherever any of the 4 stencil samples for
        that point was NaN (e.g. an out-of-domain neighbour).
    """
    v0, v1, v2, v3 = values[0], values[1], values[2], values[3]
    h0 = nodes[1] - nodes[0]
    h1 = nodes[2] - nodes[1]
    h2 = nodes[3] - nodes[2]

    s0 = (v1 - v0) / h0
    s1 = (v2 - v1) / h1
    s2 = (v3 - v2) / h2

    def _fc_deriv(sa: np.ndarray, sb: np.ndarray, ha: float, hb: float) -> np.ndarray:
        """Fritsch-Carlson derivative at the node shared by intervals ha, hb."""
        same_sign = sa * sb > 0
        safe_sa = np.where(same_sign, sa, 1.0)
        safe_sb = np.where(same_sign, sb, 1.0)
        denom = (2 * hb + ha) / safe_sa + (hb + 2 * ha) / safe_sb
        return np.where(same_sign, 3.0 * (ha + hb) / denom, 0.0)

    d1 = _fc_deriv(s0, s1, h0, h1)  # derivative at nodes[1]
    d2 = _fc_deriv(s1, s2, h1, h2)  # derivative at nodes[2]

    alpha = (query - nodes[1]) / h1
    h00 = 2 * alpha ** 3 - 3 * alpha ** 2 + 1
    h10 =     alpha ** 3 - 2 * alpha ** 2 + alpha
    h01 = -2 * alpha ** 3 + 3 * alpha ** 2
    h11 =     alpha ** 3 -     alpha ** 2

    result = h00 * v1 + h10 * h1 * d1 + h01 * v2 + h11 * h1 * d2

    # same_sign-based masking above already avoids NaN/inf from invalid-slope
    # division, but a NaN sample (out-of-domain x-stencil value) must still
    # propagate to the output rather than be silently treated as "same sign
    # is False" -> derivative 0.
    invalid = np.isnan(v0) | np.isnan(v1) | np.isnan(v2) | np.isnan(v3)
    return np.where(invalid, np.nan, result)


class PchipInterpolator3D(InterpolatorBase):
    """
    Three-dimensional PCHIP interpolator on a uniform Cartesian grid.

    The interpolator uses a lazy cache of x-direction
    ``scipy.interpolate.PchipInterpolator`` objects keyed by ``(iy, iz)``
    index pairs. Query evaluation proceeds sequentially along ``x``, then
    ``y``, then ``z``, and selected coordinate axes may be transformed
    (``log10`` or ``arcsinh``) before interpolation.
    """
    n_ghosts = 3

    def __init__(
        self,
        *coords: np.ndarray,
        shm: dict[str, str],
        shape: tuple[int, int, int],
        coord_transforms: dict | None = None,
        max_cache_size_GB: float = 0.5,
        ):
        """
        Initialize the 3-D PCHIP interpolator.

        Parameters
        ----------
        *coords : ndarray
            Coordinate arrays for the ``x``, ``y``, and ``z`` axes.
        shm : dict[str, str]
            Mapping from field keys to shared-memory segment names.
        shape : tuple[int, int, int]
            Shape of each field array stored in shared memory.
        coord_transforms : dict[int, str | tuple], optional
            Per-axis coordinate transforms applied before interpolation:
            ``"log"`` or ``("asinh", scale)``. See
            :func:`~.base.transform_coords`.
        max_cache_size_GB : float, optional
            Approximate upper bound for the cached x-interpolator memory usage.
        """
        super().__init__(shm_names=shm, shape=shape)

        self.coord_transforms = coord_transforms
        self.max_cache_size_bytes = int(max_cache_size_GB * 1024**3)

        x, y, z = transform_coords(coords, coord_transforms)

        self.nx = x.shape[0]
        self.ny = y.shape[0]
        self.nz = z.shape[0]
        if (self.nx, self.ny, self.nz) != shape:
            raise ValueError(
                f"Coordinate axis lengths ({self.nx}, {self.ny}, {self.nz}) "
                f"do not match shape {shape}."
            )

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
        self._cache_bytes = 0  # running total, kept in sync by _get_x_interp
        self.n_evicted = 0

    @staticmethod
    def cell_index(xq, x0, dx, n):
        """Return the left cell index for scalar query ``xq`` on a uniform grid."""
        frac = (xq - x0) / dx
        ix = int(np.floor(frac))
        if ix < 0:
            ix = 0
        if ix >= n - 1:
            ix = n - 2
        return ix

    def _get_x_interp(self, jy: int, kz: int):
        """Return the cached x-direction interpolator for grid column ``(jy, kz)``."""
        cache_key = (jy, kz)
        interp = self._xp_cache.get(cache_key)
        if interp is None:
            data = np.empty((self.n_keys, self.nx), dtype=np.float64)
            for i_k, key in enumerate(self.keys):
                data[i_k] = self.data[key][:, jy, kz]
            data = np.transpose(data)
            interp = PchipInterpolator(self._x_nodes, data, axis=0, extrapolate=False)
            self._xp_cache[cache_key] = interp
            self._cache_bytes += interp.c.nbytes + interp.x.nbytes

            while self._cache_bytes > self.max_cache_size_bytes and self._xp_cache:
                self.n_evicted += 1
                oldest_key = next(iter(self._xp_cache))
                oldest = self._xp_cache.pop(oldest_key)
                self._cache_bytes -= oldest.c.nbytes + oldest.x.nbytes
        return interp

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate the loaded PCHIP interpolator at the requested coordinates."""
        if not self.loaded:
            raise RuntimeError("Interpolator data not loaded; call load() before querying.")

        xi, yi, zi = transform_coords(coords, self.coord_transforms, context="Query coordinate")

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

            # Interpolate along y: contract V's axis 0 (jy), broadcasting
            # each point's own yq across the kz and key axes -> (4, n_group, n_keys).
            W = _pchip4_eval(y_nodes, V, yq[None, :, None])

            # Interpolate along z: contract W's axis 0 (kz) -> (n_group, n_keys).
            final_vals = _pchip4_eval(z_nodes, W, zq[:, None])

            out[:, idx_arr] = final_vals.T # reshape from (n_group, n_keys) to (n_keys, n_group)

        out = [o.reshape(xi.shape) for o in out]
        return np.asarray(out)


    def _estimate_cached_memory(self) -> int:
        """
        Return the total number of bytes used by the x-interpolator cache.

        ``_get_x_interp`` maintains ``self._cache_bytes`` incrementally on
        insert/evict so it doesn't need to rescan the whole cache on every
        single insertion (a full rescan per insert made cache maintenance
        scale with cache size, dominating runtime once the cache grows
        large).  This method recomputes from scratch and is kept only as a
        cheap consistency check / for any external callers.
        """
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
