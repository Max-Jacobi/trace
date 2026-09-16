"""
Reduced Surface Module

  This module provides file handling functionality for reduced/transformed
  GR-Athena++ surface files.
"""

import signal
from typing import Any
from multiprocessing.shared_memory import SharedMemory

from h5py import File
import numpy as np

from .file import FileHandler
from .utils import glob_files, tensor_cell_bounds
from .gra_surface import _fill_with_ghosts
from .interpolators.base import InterpolatorBase
from .interpolators.coordinate_transformations import CartesianToSpherical


class ReducedSurfaceFileHandler(FileHandler):
    """
    FileHandler implementation for reduced/transformed GR-Athena++ surface files.
    """

    grid_geometry = 'spherical'
    extra_data: dict[str, Any]

    def __init__(
        self,
        interpolator: type[InterpolatorBase],
        *args,
        rad_transform: str | tuple | None = "log",
        file_pattern: str = "*.hdf5",
        **kwargs,
    ) -> None:
        self.file_pattern = file_pattern
        self.n_ghosts = interpolator.n_ghosts
        super().__init__(interpolator, *args, **kwargs)
        # Transform spec for the radial axis: "log", ("asinh", scale), or None.
        self.extra_data["rad_transform"] = rad_transform
        signal.signal(signal.SIGINT, self.handler)

    def list_files(self, directory: str) -> list[str]:
        files = glob_files(directory, self.file_pattern)
        self.load_grid(files[0])
        return files

    def load_grid(self, file_path: str) -> None:
        """
        Load the grid from a reduced/transformed surface file.
        """
        with File(file_path, "r") as f:
            radii = np.array(f["coordinates/r"][()])
            th = np.array(f["coordinates/th"][()])
            ph = np.array(f["coordinates/ph"][()])

        if len(th) < 2:
            raise ValueError(
                f"Grid theta coordinate has fewer than 2 points ({len(th)}); cannot compute spacing."
            )
        if len(ph) < 2:
            raise ValueError(
                f"Grid phi coordinate has fewer than 2 points ({len(ph)}); cannot compute spacing."
            )

        dth = th[1] - th[0]
        dphi = ph[1] - ph[0]
        g_th = np.arange(1, self.n_ghosts + 1) * dth
        g_ph = np.arange(1, self.n_ghosts + 1) * dphi
        th = np.concatenate((th[0] - g_th[::-1], th, th[-1] + g_th))
        ph = np.concatenate((ph[0] - g_ph[::-1], ph, ph[-1] + g_ph))

        self.extra_data["r"] = radii
        self.extra_data["th"] = th
        self.extra_data["ph"] = ph
        self.extra_data["shape"] = (len(radii), len(th), len(ph))
        self.extra_data["mem_size"] = len(radii) * len(th) * len(ph) * 8

    @staticmethod
    def parse_file(
        file_path: str,
        keys: list[str],
        extra_data: Any = None,
    ) -> tuple[float, str, list[str], int]:
        with File(file_path, "r") as f:
            avail_keys = [key for key in keys if key in f]
            if not avail_keys:
                return 0.0, "", [], 0
            time = float(f["coordinates/time"][()])

        return time, file_path, avail_keys, extra_data["mem_size"]

    def _native_layout(
        self,
        surface_radius: float | None = None,
        ) -> tuple[np.ndarray, np.ndarray, Any, float | None]:
        """
        Where this format's cells are: their bounds, the slice of one snapshot's
        array that holds them, and the radius a surface value was sampled at.
        Split out of :meth:`native_cell_weights` so that reading a field off
        the same cells -- see :meth:`native_cell_values` -- shares it.

        This format is a single global (r, theta, phi) grid, so the cells are
        the dumped samples themselves and the bounds are one
        ``tensor_cell_bounds`` call. Edges follow the layout documented in
        ``docs/formats/gr_athena.md``: ``r`` is geometric, so a sample sits at
        the geometric centre of ``[r/sqrt(q), r*sqrt(q)]``; ``theta`` and
        ``phi`` are uniform and cell-centred, so a sample spans half a spacing
        either side, and their edges tile ``[0, pi]`` and ``[0, 2pi]`` exactly.
        Ghost zones are stripped before anything is returned.
        """
        ng = self.n_ghosts
        r = np.asarray(self.extra_data["r"], dtype=float)
        th = np.asarray(self.extra_data["th"], dtype=float)[ng:-ng]
        ph = np.asarray(self.extra_data["ph"], dtype=float)[ng:-ng]

        dth = th[1] - th[0]
        dph = ph[1] - ph[0]
        cth_edges = np.concatenate((np.cos(th - dth / 2), [np.cos(th[-1] + dth / 2)]))
        ph_edges = np.concatenate((ph - dph / 2, [ph[-1] + dph / 2]))

        q = float(r[1] / r[0])              # constant, the grid is geometric
        r_edges = np.concatenate((r / np.sqrt(q), [r[-1] * np.sqrt(q)]))

        if surface_radius is None:
            i_r = slice(None)
            lo, hi = tensor_cell_bounds(r_edges, cth_edges, ph_edges)
            r_sample = None
        else:
            # The shell whose cell the sphere passes through, half-open so a
            # sphere on a face takes exactly one. Not the nearest shell: the
            # sphere stays at exactly the radius asked for, so it is the outer
            # boundary of a volume seeded to the same radius, and this shell's
            # value stands for the flux through it.
            j = int(np.searchsorted(r_edges, surface_radius, 'right')) - 1
            if not 0 <= j < len(r):
                raise ValueError(
                    f"r = {surface_radius:g} lies outside the dumped shells, whose "
                    f"cells span [{r_edges[0]:g}, {r_edges[-1]:g})."
                )
            i_r = slice(j, j + 1)
            lo, hi = tensor_cell_bounds(cth_edges, ph_edges)
            r_sample = float(r[j])

        return lo, hi, i_r, r_sample

    def _read_cells(self, slot: int, keys: tuple[str, ...], i_r) -> np.ndarray:
        ng = self.n_ghosts
        return self.read_shm_cells(slot, keys,
                                   lambda buf: buf[i_r, ng:-ng, ng:-ng])

    def native_cell_weights(
        self,
        slot: int,
        surface_radius: float | None = None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """See :meth:`~src.file.FileHandler.native_cell_weights`."""
        lo, hi, i_r, r_sample = self._native_layout(surface_radius)
        keys = (self.mass_density.density_keys if surface_radius is None
                else self.mass_density.flux_keys)
        values = self._read_cells(slot, keys, i_r)
        return lo, hi, self.cell_weights_from_values(lo, hi, values, r_sample)

    def native_cell_values(
        self,
        slot: int,
        keys: tuple[str, ...],
        surface_radius: float | None = None,
        ) -> np.ndarray:
        """See :meth:`~src.file.FileHandler.native_cell_values`."""
        _, _, i_r, _ = self._native_layout(surface_radius)
        return self._read_cells(slot, keys, i_r)

    @staticmethod
    def load_step_to_memory(
        metadata_dict: dict[str, list[str]],
        shared_memory: dict[str, str],
        extra_data: Any = None,
    ) -> None:
        nghosts = extra_data["interpolator"].n_ghosts
        shape = extra_data["shape"]

        for path, keys in metadata_dict.items():
            with File(path, "r") as f:
                for key in keys:
                    shm = SharedMemory(name=shared_memory[key])
                    try:
                        buf = np.ndarray(shape=shape, dtype=np.float64, buffer=shm.buf)
                        data = f[key]
                        for ir in range(shape[0]):
                            _fill_with_ghosts(buf[ir], {key: data[ir]}, key, ng=nghosts)
                    finally:
                        shm.close()

    @staticmethod
    def setup_interpolator(
        shared_memory: dict[str, str],
        extra_data: Any = None,
    ) -> InterpolatorBase:
        r = extra_data["r"]
        th = extra_data["th"]
        phi = extra_data["ph"]
        interpolator = extra_data["interpolator"]
        spec = extra_data["rad_transform"]
        coord_transforms = {0: spec} if spec else {}
        interpolator = CartesianToSpherical(
            interpolator,
            r,
            th,
            phi,
            shm=shared_memory,
            coord_transforms=coord_transforms,
            shape=extra_data["shape"],
            **extra_data.get("interpolator_kwargs", {}),
        )
        return interpolator

    def handler(self, signum, frame):
        """
        Signal handler for graceful shutdown on interrupt signal.
        """
        if signum == signal.SIGINT:
            print("Received interrupt signal. Exiting gracefully...")
            self.free_shared_memory()
