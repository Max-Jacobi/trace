from abc import ABC, abstractmethod
import numpy as np
from multiprocessing.shared_memory import SharedMemory

class InterpolatorBase(ABC):
    """
    Base class for interpolators.
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
        pass

    def __del__(self):
        self.unload()

    def load(self):
        for key in self.keys:
            shm = SharedMemory(name=self.shm_names[key])
            self.shm[key] = shm
            self.data[key] = np.ndarray(self.shape, dtype=np.float64, buffer=shm.buf)
        self.loaded = True

    def unload(self):
        for key, shm in self.shm.items():
            shm.close()
            del self.data[key]
        self.shm.clear()
        self.loaded = False
