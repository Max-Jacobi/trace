"""
Implicit trapezoid (Crank-Nicolson) fixed-step integrator.

Implements a Picard (fixed-point) iteration per step.

This is a standalone integrator that assumes separate interpolation functions
are provided for the two time levels.

Notes:
- No temporal interpolation at intermediate times is required because the
  scheme only needs velocities at the endpoints t_n and t_{n+1}.
"""
from typing import Sequence, Callable
import numpy as np

from .base import InterpolatorCallable, IntegratorBase

class ImplicitTrapezoid(IntegratorBase):
    converged: bool = False
    n_iter: int = 0

    def __init__(self, tol: float = 1e-8, max_iter: int = 20, relax: float = 1.0):
        """
        Initialize the implicit trapezoid integrator.

        tol: relative tolerance for convergence (Euclidean norm)
        max_iter: max Picard iterations before fallback
        relax: relaxation factor in (0,1] applied to Picard updates
        """
        self.tol = tol
        self.max_iter = max_iter
        self.relax = relax

    def __call__(
        self,
        xn: np.ndarray,
        dt: float,
        interps: tuple[InterpolatorCallable, ...],
        snap_times: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Perform a single implicit trapezoid update for one tracer.

        xn: array shape (d {,n}) current position(s)
        dt: array shape (1,) time step
        interps: tuple of two callables (interp_n, interp_n1)
        snap_times: unused (two-snapshot scheme needs only dt)
        tol: relative tolerance for convergence (Euclidean norm)
        max_iter: max Picard iterations before fallback
        relax: relaxation factor in (0,1] applied to Picard updates

        Returns:
        - x_new: array shape (d, {n}) new position(s)
        - converged: bool
        - n_iter: number of iterations used
        """

        self.converged = False

        vn = interps[0](xn)

        xk = xn + dt * vn

        # Picard iterations: x_{k+1} = xn + 0.5*dt*(vn + v(tn1, x_k))
        for self.n_iter in range(1, self.max_iter + 1):
            v_k = interps[1](xk)
            x_candidate = xn + 0.5 * dt * (vn + v_k)
            delta = x_candidate - xk
            x_new = xk + self.relax * delta

            if np.linalg.norm(delta) <= self.tol * max(1.0, np.linalg.norm(x_new)):
                self.converged = True
                return x_new
            xk = x_new
        return xk
