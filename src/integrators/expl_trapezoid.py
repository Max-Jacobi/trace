"""
  Explicit Trapezoidal Method for solving ODEs

   This script implements the explicit trapezoidal method (also known as the
    modified Euler method) for numerically solving ordinary differential equations (ODEs).
    The method is a two-step process that first predicts the next value using
    Euler's method and then corrects it by averaging the slopes at the current
    and predicted points.
"""

import numpy as np

from .base import InterpolatorCallable, IntegratorBase

class ExplicitTrapezoid(IntegratorBase):

    def __call__(
        self,
        xn: np.ndarray,
        dt: float,
        interps: tuple[InterpolatorCallable, ...],
        snap_times: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Perform a single explicit trapezoidal update for one tracer.

        xn: array shape (d {,n}) current position(s)
        dt: float
        interps: tuple of two callables (interp_n, interp_n1)
        snap_times: unused (two-snapshot scheme needs only dt)

        Returns:
        - x_new: array shape (d, {n}) new position(s)
        """

        vn = interps[0](xn)
        x_predict = xn + dt * vn
        v_predict = interps[1](x_predict)

        x_new = xn + 0.5 * dt * (vn + v_predict)

        return x_new
