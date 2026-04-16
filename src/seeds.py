"""
Class to seed initial position and times for tracers.
"""

from abc import ABC, abstractmethod
import numpy as np

class SeedsBase(ABC):
    n_tracers: int  # number of tracers

    @abstractmethod
    def get_positions(self) -> np.ndarray:
        pass


    @abstractmethod
    def get_times(self) -> np.ndarray:
        pass


class SimpleSeeds(SeedsBase):
    def __init__(
        self,
        positions: np.ndarray, # shape (n_tracers, n_dim)
        times: np.ndarray,     # shape (n_tracers, )
        ):
        isort = np.argsort(times)
        self.positions = positions[isort]
        self.times = times[isort]
        self.n_tracers = len(times)


    def get_positions(self) -> np.ndarray:
        return self.positions

    def get_times(self) -> np.ndarray:
        return self.times
