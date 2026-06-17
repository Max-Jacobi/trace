"""Explicit trapezoidal integrator module."""

import numpy as np

from .base import InterpolatorCallable, IntegratorBase

class ExplicitTrapezoid(IntegratorBase):
    """Advance tracer positions with the explicit trapezoidal method."""

    def __call__(
        self,
        xn: np.ndarray,
        dt: float,
        interps: tuple[InterpolatorCallable, ...],
        snap_times: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Perform a single explicit trapezoidal update for a batch of tracers.

        Parameters
        ----------
        xn : ndarray
            Current position array with shape ``(d,)`` or ``(d, n)``.
        dt : float
            Time step from ``t_n`` to ``t_{n+1}``.
        interps : tuple of callable
            Pair of interpolators ``(interp_n, interp_n1)`` evaluated at the
            start and end snapshots.
        snap_times : ndarray or None, optional
            Unused, provided for interface compatibility.

        Returns
        -------
        ndarray
            Updated position array with the same shape as ``xn``.
        """

        vn = interps[0](xn)
        x_predict = xn + dt * vn
        v_predict = interps[1](x_predict)

        x_new = xn + 0.5 * dt * (vn + v_predict)

        return x_new
