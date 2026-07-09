from abc import ABC, abstractmethod
import numpy as np
from multiprocessing.shared_memory import SharedMemory

class InterpolatorBase(ABC):
    """
    Base class for interpolators backed by shared-memory field arrays.

    Subclasses are constructed with shared-memory names and array metadata,
    then follow the lifecycle ``construct -> load() -> call -> unload()``.
    The ``n_ghosts`` attribute declares how many ghost cells are required on
    each coordinate axis by the interpolation stencil.
    """
    n_ghosts: int # number of ghost zones required per dimension
    keys: list[str] # list of data keys required for interpolation
    n_keys: int # number of data keys
    data: dict[str, np.ndarray] # dictionary of data arrays for each key
    shm_names: dict[str, str] # dictionary linking data keys with shared memory names
    shm: dict[str, 'SharedMemory'] # dictionary of SharedMemory objects for each key
    shape: tuple[int, ...] # shape of the data arrays

    def __init__(self, shm_names: dict[str, str], shape: tuple[int, ...]):
        self.shm_names = shm_names
        self.shape = shape
        self.keys = list(shm_names.keys())
        self.n_keys = len(self.keys)
        self.shm: dict[str, SharedMemory] = {}
        self.data: dict[str, np.ndarray] = {}
        self.loaded = False

    @abstractmethod
    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """
        Evaluate the interpolator at the requested coordinates.

        Parameters
        ----------
        coords : ndarray
            Coordinate array describing the query points.

        Returns
        -------
        ndarray
            Interpolated values with shape ``(n_keys, n_points)``.

        Notes
        -----
        This is an abstract method that must be implemented by subclasses.
        """
        pass

    def __del__(self):
        self.unload()

    def load(self, track: bool = True):
        """
        Open the shared-memory segments and expose their data arrays.

        Parameters
        ----------
        track : bool, optional
            Whether to register the opened segments with the multiprocessing
            resource tracker.  Pass ``False`` when this instance only
            attaches to memory owned (created and eventually unlinked) by
            another process -- e.g. a worker process that just reads a
            snapshot built by the main process -- so the resource tracker
            doesn't report a spurious "leak" for handles this process was
            never responsible for unlinking.
        """
        for key in self.keys:
            shm = SharedMemory(name=self.shm_names[key], track=track)
            self.shm[key] = shm
            self.data[key] = np.ndarray(self.shape, dtype=np.float64, buffer=shm.buf)
        self.loaded = True

    def unload(self):
        """
        Close shared-memory handles and release the cached data arrays.
        """
        for key, shm in self.shm.items():
            shm.close()
            del self.data[key]
        self.shm.clear()
        self.loaded = False
