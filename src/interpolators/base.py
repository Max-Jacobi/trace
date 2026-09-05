import inspect
from abc import ABC, abstractmethod
import numpy as np
from multiprocessing.shared_memory import SharedMemory

# SharedMemory's `track` kwarg was added in Python 3.13 (bpo-82300); guard it
# so this still runs under older interpreters (e.g. 3.11), just without the
# ability to opt out of resource-tracker registration on those versions.
_SHM_SUPPORTS_TRACK = "track" in inspect.signature(SharedMemory.__init__).parameters


def transform_coords(coords, coord_transforms, context="Coordinate"):
    """
    Apply per-axis forward transforms to a sequence of coordinate arrays.

    Parameters
    ----------
    coords : sequence of ndarray
        One coordinate array per axis.
    coord_transforms : dict[int, str | tuple] or None
        Mapping from axis index to a transform spec:
        ``"log"`` for ``log10``, or ``("asinh", scale)`` for
        ``arcsinh(c / scale)`` where ``scale`` is the approximate
        lin-log transition point.
    context : str, optional
        Prefix used in error messages (e.g. "Query coordinate").

    Returns
    -------
    list of ndarray
        Transformed coordinate arrays, same order as ``coords``.
    """
    out = []
    for i, c in enumerate(coords):
        c = np.asarray(c)
        spec = (coord_transforms or {}).get(i)
        if spec is None:
            out.append(c)
            continue
        name, *params = (spec,) if isinstance(spec, str) else spec
        if name == "log":
            if np.any(c <= 0):
                raise ValueError(
                    f"{context} axis {i} contains non-positive values; "
                    "log10 requires strictly positive inputs."
                )
            out.append(np.log10(c))
        elif name == "asinh":
            (scale,) = params
            out.append(np.arcsinh(c / scale))
        else:
            raise ValueError(f"Unknown coordinate transform {name!r} for axis {i}.")
    return out


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
            kwargs = {"track": track} if _SHM_SUPPORTS_TRACK else {}
            shm = SharedMemory(name=self.shm_names[key], **kwargs)
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
