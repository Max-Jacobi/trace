"""Regular-grid interpolator wrappers."""

from scipy.interpolate import RegularGridInterpolator
import numpy as np

from .base import InterpolatorBase, transform_coords

class RegularInterpolator3D(InterpolatorBase):
    """
    Wrap ``scipy.interpolate.RegularGridInterpolator`` for 3-D Cartesian data.

    Linear interpolation is used by default, and selected coordinate axes can
    be transformed (``log10`` or ``arcsinh``) before interpolation.
    """
    n_ghosts = 1

    def __init__(
        self,
        *coords: np.ndarray,
        shm: dict[str, str],
        shape: tuple[int, int, int],
        coord_transforms: dict | None = None,
        method: str = "linear",
        ):
        """
        Initialize the regular-grid interpolator.

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
        method : str, optional
            Interpolation method passed to ``RegularGridInterpolator``.
        """
        super().__init__(shm_names=shm, shape=shape)
        self.coord_transforms = coord_transforms
        self.method = method
        self.interp_cache = {}  # initialised early so __del__ is safe on failed init

        x, y, z = transform_coords(coords, coord_transforms)

        self.nx = x.shape[0]
        self.ny = y.shape[0]
        self.nz = z.shape[0]
        if (self.nx, self.ny, self.nz) != shape:
            raise ValueError(
                f"Coordinate axis lengths ({self.nx}, {self.ny}, {self.nz}) "
                f"do not match shape {shape}."
            )

        self._x_nodes = np.asarray(x)
        self._y_nodes = np.asarray(y)
        self._z_nodes = np.asarray(z)

    def load(self, track: bool = True):
        """Build cached SciPy interpolators for the loaded field arrays."""
        super().load(track=track)
        for k, dd in self.data.items():
            self.interp_cache[k] = RegularGridInterpolator(
                (self._x_nodes, self._y_nodes, self._z_nodes),
                dd,
                bounds_error=False,
                fill_value=np.nan,
                method=self.method,
            )

    def unload(self):
        """Release shared-memory data and clear cached interpolator objects."""
        super().unload()
        self.interp_cache.clear()

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate the loaded interpolators at the requested coordinates."""
        if not self.loaded:
            raise RuntimeError("Interpolator data not loaded; call load() before querying.")

        xi, yi, zi = transform_coords(coords, self.coord_transforms, context="Query coordinate")
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
