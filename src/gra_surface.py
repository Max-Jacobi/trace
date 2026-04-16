"""
GRA Surface Module

  This modules provides the file handling functionality for GR-Athena++ surface files.
  See github.com/computationalrelativity/gr-athena for more details.
"""

from pathlib import Path
from typing import Any, Callable
from multiprocessing.shared_memory import SharedMemory
from h5py import File
import numpy as np

from .file import FileHandler
from .interpolators.base import InterpolatorBase
from .interpolators.coordinate_transformations import CartesianToSpherical


def _fill_with_ghosts(
    buf: np.ndarray,
    h5f: File,
    key: str,
    ng: int,
    ):
    """
    Fill buffer with data from h5 file, adding spherical ghost zones.
    """
    try:
        ar = np.array(h5f[key][:])
    except KeyError:
        print(f"Key {key} not found in file.")
        print(h5f.keys())
        print(h5f.file)
        raise
    nphi = ar.shape[1]
    buf[ng:-ng, ng:-ng] = ar[:, :]
    for ig in range(ng):
        ign = -ig-1
        buf[ ig, ng:-ng] = np.roll(ar[ ig, :], nphi//2)
        buf[ign, ng:-ng] = np.roll(ar[ign, :], nphi//2)
        buf[ng:-ng,  ig] = ar[:, ign]
        buf[ng:-ng, ign] = ar[:,  ig]
        buf[ig, ig] = ar[ig, ig]
        buf[ign, ig] = ar[ign, ig]
        buf[ig, ign] = ar[ig, ign]
        buf[ign, ign] = ar[ign, ign]


class GRASurfaceFileHandler(FileHandler):
    """
    FileHandler implementation for GR-Athena++ surface files.
    """
    extra_data: dict[str, Any]

    def __init__(
        self,
        interpolator: type[InterpolatorBase],
        *args,
        log_rad: bool = True,
        surface_num: int = 1,
        **kwargs
        ) -> None:
        self.extra_data = {'interpolator': interpolator}
        self.n_ghosts = interpolator.n_ghosts
        self.surface_num = surface_num
        self.log_rad = log_rad
        super().__init__(*args, **kwargs)

    def list_files(self, directory: str) -> list[str]:
        path = Path(directory)
        files = [str(path/f.name) for f in path.iterdir() if f'.surface{self.surface_num}' in f.name]
        self.load_grid(files[0])
        return files

    def load_grid(
        self,
        file_path: str,
        ) -> None:
        """
        Loads the grid from a GR-Athena++ surface file.
        """
        with File(file_path, 'r') as f:
            n_radii = len(f['coordinates'].keys())
            th = np.array(f['coordinates/00/th'][()])
            ph = np.array(f['coordinates/00/ph'][()])
            try:
                radii = np.array([float(f[f'coordinates/{ir:02d}/R'][0]) for ir in range(n_radii)])
            except Exception as e:
                print(f"Error loading radii: {e}")
                print(f.keys())
                print(f.file)
                raise
        dth = th[1] - th[0]
        dphi = ph[1] - ph[0]
        g_th = np.arange(1, self.n_ghosts+1)*dth
        g_ph = np.arange(1, self.n_ghosts+1)*dphi
        th = np.concatenate((th[0]-g_th[::-1], th, th[-1]+g_th))
        ph = np.concatenate((ph[0]-g_ph[::-1], ph, ph[-1]+g_ph))
        self.extra_data['r'] = radii
        self.extra_data['th'] = th
        self.extra_data['ph'] = ph
        self.extra_data['shape'] = (len(radii), len(th), len(ph))
        self.extra_data['mem_size'] = len(radii)*len(th)*len(ph)*8

    @staticmethod
    def parse_file(
        file_path: str,
        keys: list[str],
        extra_data: Any = None,
        ) -> tuple[float, str, list[str], int]:
        with File(file_path, 'r') as f:
            file_keys = [key for grp in f['fields/00/'].keys()
                         for key in f[f'fields/00/{grp}/'].keys()]
            avail_keys = [key for key in keys if key in file_keys]
            if not avail_keys:
                return 0.0, file_path, [], 0
            time = float(f['coordinates/00/T'][0])
        return time, file_path, avail_keys, extra_data['mem_size']

    @staticmethod
    def load_step_to_memory(
        metadata_dict: dict[str, list[str]],
        shared_memory: dict[str, str],
        extra_data: Any = None,
        ) -> None:

        nghosts = extra_data['interpolator'].n_ghosts
        shape = extra_data['shape']
        for path, keys in metadata_dict.items():
            with File(path, 'r') as f:
                for key in keys:
                    shm = SharedMemory(name=shared_memory[key])
                    grp = key.split('.')[0]
                    buf = np.ndarray(
                        shape=shape,
                        dtype=np.float64,
                        buffer=shm.buf
                    )
                    for ir in range(shape[0]):
                        _fill_with_ghosts(buf[ir], f, f'fields/{ir:02d}/{grp}/{key}', ng=nghosts)

    def setup_interpolator(
        self,
        shared_memory: dict[str, str],
        extra_data: Any = None,
        ) -> InterpolatorBase:

        data = {}
        shm = {}
        for key, shm_name in shared_memory.items():
            shm[key] = SharedMemory(name=shm_name)
            data[key] = np.ndarray(
                shape=extra_data['shape'],
                dtype=np.float64,
                buffer=shm[key].buf
                )

        r = extra_data['r']
        th = extra_data['th']
        phi = extra_data['ph']

        interpolator = extra_data['interpolator']
        log_coords = [0] if self.log_rad else []
        interpolator = CartesianToSpherical(interpolator, r, th, phi, d=data, log_coords=log_coords)
        interpolator.shared_memory = shm # Store shared memory references to prevent premature cleanup
        return interpolator
