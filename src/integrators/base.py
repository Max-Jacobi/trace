from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

InterpolatorCallable = Callable[[np.ndarray], np.ndarray]

class IntegratorBase(ABC):
    """
    Base class for time integrators.

    Subclasses must set ``n_snapshots`` to the number of consecutive velocity
    snapshots required per integration step.  The default of 2 matches the
    two-level schemes (explicit / implicit trapezoid).  Higher-order schemes
    such as RK4 with cubic time interpolation require more snapshots.
    """

    n_snapshots: int = 2  # number of velocity snapshots consumed per step

    @abstractmethod
    def __call__(
        self,
        xn: np.ndarray,
        dt: float,
        interps: tuple[InterpolatorCallable, ...],
    ) -> np.ndarray:
        """
        Perform a single time step update.
        Args:
            xn: array shape (d {,n}) current position(s)
            dt: float,
            interps: tuple of interpolator callables, length == n_snapshots
        """
        pass
