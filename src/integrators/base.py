from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

InterpolatorCallable = Callable[[np.ndarray], np.ndarray]

class IntegratorBase(ABC):
    """
    Base class for time integrators.
    """

    @abstractmethod
    def __call__(
        self,
        xn: np.ndarray,
        dt: float,
        interps: tuple[InterpolatorCallable, InterpolatorCallable],
    ) -> np.ndarray:
        """
        Perform a single time step update.
        Args:
            xn: array shape (d {,n}) current position(s)
            dt: float,
            interps: tuple interpolator callables of length n_steps
        """
        pass
