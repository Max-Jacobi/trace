from abc import ABC, abstractmethod
import numpy as np

class InterpolatorBase(ABC):
    """
    Base class for interpolators.
    """
    n_ghosts: int # number of ghost zones required per dimension

    @abstractmethod
    def __init__(self, *coords: np.ndarray, d: dict[str, str], shape: tuple[int, ...], log_coords: list[int] = []):
        pass

    @abstractmethod
    def __call__(self, coords: np.ndarray) -> np.ndarray:
        pass
