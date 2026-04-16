from abc import ABC, abstractmethod
import numpy as np

class InterpolatorBase(ABC):
    """
    Base class for interpolators.
    """
    n_ghosts: int # number of ghost zones required per dimension

    @abstractmethod
    def __init__(self, *coords: np.ndarray, d: dict[str, np.ndarray], log_coords: list[int] = []):
        pass

    @abstractmethod
    def __call__(self, coords: np.ndarray) -> np.ndarray:
        pass


class UnloadedInterpolator(InterpolatorBase):
    """
    Placeholder interpolator for unloaded data.
    raise an error when called.
    """
    n_ghosts = 0

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        raise RuntimeError("Interpolator called before data was loaded.")


class OutOfBoundsError(Exception):
    pass
