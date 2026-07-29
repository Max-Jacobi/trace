"""
Reduced Surface Module

  This module provides file handling functionality for reduced/transformed
  GR-Athena++ surface files.
"""

import signal
from pathlib import Path
from typing import Any
from multiprocessing.shared_memory import SharedMemory

from h5py import File
import numpy as np

from .file import FileHandler
from .gra_surface import _fill_with_ghosts
from .interpolators.base import InterpolatorBase
from .interpolators.coordinate_transformations import CartesianToSpherical


class ReducedSurfaceFileHandler(FileHandler):
    """
    FileHandler implementation for reduced/transformed GR-Athena++ surface files.
    """

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
        path = Path(directory)
        has_glob = any(ch in self.file_pattern for ch in "*?[]")
        if has_glob:
            files = sorted(str(f) for f in path.glob(self.file_pattern) if f.is_file())
        else:
            files = sorted(str(f) for f in path.iterdir() if f.is_file() and f.name.endswith(self.file_pattern))

        if not files:
            raise FileNotFoundError(
                f"No files matching pattern '{self.file_pattern}' found in directory: {directory}"
            )

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
